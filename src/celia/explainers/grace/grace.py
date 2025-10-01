from typing import Tuple, List, Optional, TYPE_CHECKING, Union
import numpy as np
import copy
import pandas as pd

from celia.data import BaseData, PublicData
from celia.model import BaseModel, TorchModel
from celia.explainers import ClassifierExplainer
from celia._errors import ConfigurationError
from celia._utils.dependecies import requires_torch_class
from celia.counterfactuals import Counterfactual

if TYPE_CHECKING:
    import torch
    from torch import Tensor

@requires_torch_class
class GRACE:
    def __init__(self, model: "torch.nn.Module"):
        self.model = model

    @staticmethod
    def _place_perturbation(perturbation: np.ndarray, feature_indices: np.ndarray | List[int],
                           input_shape: Tuple[int]) -> np.ndarray:
        """
        Insert perturbation values at the specified feature indices for an input of the given shape.
        """
        perturbation_mask = np.zeros(input_shape)
        perturbation_mask[0, feature_indices] = perturbation
        return perturbation_mask

    def _naive_gradient_attack(self,
            original_instance: "Tensor",
            num_features: int,
            num_classes: int = 2,
            overshoot: float = 0.0001,
            max_iterations: int = 50,
            feature_min_bounds: List[float] = None,
            feature_max_bounds: List[float] = None,
            feature_is_integer: List[bool] = None,
            feature_alphas: List[float] | np.ndarray = None,
            feature_selector: Optional[object] = None
    ) -> Tuple[int, np.ndarray, np.ndarray, List[int], int]:
        """
        Generate a counterfactual example by iteratively applying gradient-based perturbations.

        Parameters
        ----------
        original_instance : Tensor
            The original input sample.
        num_features : int
            Number of features to perturb.
        num_classes : int, optional
            Number of output classes of the model (default=2).
        overshoot : float, optional
            Factor to scale perturbation size (default=0.0001).
        max_iterations : int, optional
            Maximum number of perturbation iterations (default=50).
        feature_min_bounds : list of float, optional
            Minimum feasible values for each feature.
        feature_max_bounds : list of float, optional
            Maximum feasible values for each feature.
        feature_is_integer : list of bool, optional
            Flags indicating which features are integer-valued.
        feature_alphas : list of float, optional
            Scaling factors used when rounding integer-valued features.
        feature_selector : object, optional
            Custom feature selector with a `.select(indices, num_features)` method.

        Returns
        -------
        original_class : int
            Predicted class of the unperturbed input.
        counterfactual_class : int
            Predicted class of the counterfactual input.
        cumulative_perturbation : np.ndarray
            Total perturbation applied across all iterations.
        counterfactual_input : np.ndarray
            Final counterfactual example as a NumPy array.
        selected_features : np.ndarray
            Indices of the features perturbed.
        iteration_count : int
            Number of iterations performed.
        """
        import torch

        predictions = self.model.predict_proba(original_instance)
        sorted_class_indices: List[int] = predictions.argsort()[::-1]

        original_class = self.model.predict(original_instance)

        input_clone = copy.deepcopy(original_instance)
        cumulative_perturbation = np.zeros(num_features)
        minimal_perturbation_direction = np.zeros(num_features)

        perturbed_input = input_clone.clone().detach().requires_grad_(True)
        outputs = self.model(perturbed_input)

        tolerance = 1e-8
        iteration_count = 0
        continue_search = True

        outputs[0, original_class].backward(retain_graph=True)
        selected_features = []

        counterfactual_array : np.ndarray = input_clone.detach().numpy().astype(np.float32, copy=True)

        while continue_search:
            outputs[0, original_class].backward(retain_graph=True)
            gradients = perturbed_input.grad.detach().numpy().copy()

            if len(selected_features) == 0:
                sorted_features = np.argsort(gradients)[::-1][0]
                if feature_selector:
                    selected_features: List[int] = feature_selector.select(sorted_features, num_features)
                else:
                    selected_features = sorted_features[:num_features].tolist()

            base_gradient = gradients[0, selected_features]
            best_step_size = np.inf

            for class_index in range(1, num_classes):
                if perturbed_input.grad is not None:
                    perturbed_input.grad.zero_()
                outputs[0, sorted_class_indices[class_index]].backward(retain_graph=True)

                current_gradient = perturbed_input.grad.detach().numpy().copy()[0, selected_features]
                gradient_difference = current_gradient - base_gradient
                logit_difference = (
                        outputs[0, sorted_class_indices[class_index]] - outputs[0, original_class]
                ).detach().numpy()

                step_size = abs(logit_difference) / (np.linalg.norm(gradient_difference.flatten()) + tolerance)
                if step_size < best_step_size:
                    best_step_size = step_size
                    minimal_perturbation_direction = gradient_difference

            scaled_step = (best_step_size + 1e-4) * minimal_perturbation_direction / (
                    np.linalg.norm(minimal_perturbation_direction) + tolerance
            )
            cumulative_perturbation = np.float32(cumulative_perturbation + scaled_step)

            perturbation_mask = self._place_perturbation(scaled_step, selected_features, perturbed_input.shape)

            counterfactual_array += (1 + overshoot) * perturbation_mask


            counterfactual_array = self._maintain_domain(
                counterfactual_array,
                selected_features,
                feature_alphas,
                feature_is_integer,
                feature_min_bounds,
                feature_max_bounds
            )

            counterfactual_tensor = torch.from_numpy(counterfactual_array).float()
            perturbed_input = counterfactual_tensor.clone().detach().requires_grad_(True)
            outputs = self.model(perturbed_input)
            counterfactual_class = np.argmax(outputs.detach().numpy().flatten())

            iteration_count += 1
            if counterfactual_class != original_class or iteration_count > max_iterations:
                continue_search = False

        final_counterfactual_array = counterfactual_array.copy()
        final_counterfactual_tensor = torch.from_numpy(final_counterfactual_array).float()
        final_input = final_counterfactual_tensor.clone().detach().requires_grad_(True)

        outputs = self.model(final_input)
        counterfactual_class = np.argmax(outputs.detach().numpy().flatten())

        return (
            counterfactual_class,
            cumulative_perturbation,
            final_counterfactual_array.flatten(),
            selected_features,
            iteration_count,
        )

    @staticmethod
    def _maintain_domain(counterfactual_array: np.ndarray,
                       selected_features: List[int],
                       feature_alphas: List[float],
                       feature_is_integer: List[bool] = None,
                       feature_min_bounds: List[float] = None,
                       feature_max_bounds: List[float] = None
                       ) -> np.ndarray:
        """
        Adjust counterfactual values to respect integer constraints and feature bounds.
        """

        if feature_is_integer and any(feature_is_integer):
            selected_feature_set = set(selected_features)
            for feature_idx, is_integer in enumerate(feature_is_integer):
                if not is_integer or feature_idx not in selected_feature_set:
                    continue

                step_size = float(feature_alphas[feature_idx])
                if not np.isfinite(step_size) or step_size <= 0.0:
                    continue

                counterfactual_array[:, feature_idx] = (
                        np.round(counterfactual_array[:, feature_idx] / step_size) * step_size
                )


        if feature_min_bounds is not None:
            counterfactual_array = np.maximum(counterfactual_array, feature_min_bounds)
        if feature_max_bounds is not None:
            counterfactual_array = np.maximum(counterfactual_array, feature_max_bounds)

        return counterfactual_array

    def generate_counterfactuals(self, original_instance: Union["Tensor", np.ndarray, pd.DataFrame, pd.Series],
                                 max_features_to_perturb: int,
                                 overshoot: float = 0.0001,
                                 max_iterations: int = 50,
                                 feature_min_bounds: List[float] = None,
                                 feature_max_bounds: List[float] = None,
                                 feature_is_integer: List[bool] = None,
                                 feature_alphas: List[float] | np.ndarray = None,
                                 feature_selector: Optional[object] = None,
                                 feature_names: Optional[List[str]] = None,
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
            Generate counterfactuals for a given instance and return them as a pandas DataFrame.

            Parameters
            ----------
            original_instance : Tensor | np.ndarray | pd.DataFrame | pd.Series
                The original input instance.
            max_features_to_perturb : int
                Maximum number of features to perturb.
            overshoot : float, optional
                Overshoot factor for perturbation, by default 0.0001.
            max_iterations : int, optional
                Maximum number of iterations for counterfactual generation, by default 50.
            feature_min_bounds : List[float], optional
                Minimum bounds for features.
            feature_max_bounds : List[float], optional
                Maximum bounds for features.
            feature_is_integer : List[bool], optional
                Whether each feature is integer-valued.
            feature_alphas : List[float], optional
                Alpha weights per feature.
            feature_selector : object, optional
                Custom feature selection logic.
            feature_names : List[str], optional
                Names of features for the dataframe.

            Returns
            -------
            Tuple[pd.DataFrame, pd.DataFrame]
                The first dataframe is the original instance and the second is the counterfactual instance.
            """
        import torch

        if isinstance(original_instance, pd.DataFrame):
            original_instance = torch.from_numpy(original_instance.to_numpy()).float()
        elif isinstance(original_instance, np.ndarray):
            original_instance = torch.from_numpy(original_instance).float()
        elif isinstance(original_instance, pd.Series):
            original_instance = torch.from_numpy(original_instance.to_numpy().reshape(1, -1)).float()

        original_class = self.model.predict(original_instance)
        for num_features in range(1, max_features_to_perturb + 1):
            (
                counterfactual_class,
                cumulative_perturbation,
                counterfactual_array,
                selected_features,
                iteration_count,
            ) = self._naive_gradient_attack(
                original_instance=original_instance,
                num_features = num_features,
                overshoot=overshoot,
                max_iterations=max_iterations,
                feature_min_bounds=feature_min_bounds,
                feature_max_bounds=feature_max_bounds,
                feature_is_integer=feature_is_integer,
                feature_alphas=feature_alphas,
                feature_selector=feature_selector
            )

            if original_class != counterfactual_class:
                break

        original_instance_numpy = original_instance.detach().numpy().flatten()
        counterfactual_instance = counterfactual_array.flatten()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(original_instance_numpy))]

        original_row = {
            feature_names[idx]: float(original_instance_numpy[idx])
            for idx in range(len(original_instance_numpy))
        }
        original_row["Prediction"] = int(original_class)

        counterfactual_row = {
            **{
                feature_names[idx]: f"{counterfactual_instance[idx]:.3f}"
                for idx in range(len(counterfactual_instance))
            },
            "Prediction": counterfactual_class,
        }

        original_instance_df = pd.DataFrame([original_row])
        counterfactual_instance_df = pd.DataFrame([counterfactual_row])

        return original_instance_df, counterfactual_instance_df

class GRACEClassifierExplainer(ClassifierExplainer):
    """
    GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model’s Prediction.
    Thai Le, Suhang Wang, Dongwon Lee. 26th ACM SIGKDD Int’l Conf. on Knowledge Discovery and Data Mining (KDD), Virtual. August 2020.
    """

    def __init__(self, model: BaseModel, data: BaseData, *args, **kwargs):
        if not isinstance(model, TorchModel):
            raise ConfigurationError(
                message="GRACEClassifierExplainer requires a TorchModel wrapping a torch.nn.Module.",
                param="model",
                config={"expected": "TorchModel", "received": str(type(model))},
                hint="Wrap your torch.nn.Module with `TorchModel(model=...)`",
                source="GRACEClassifierExplainer.__init__",
            )
        if not isinstance(data, PublicData):
            raise ConfigurationError(
                message="GRACEClassifierExplainer requires data to be an instance of PublicData.",
                param="data",
                config={"expected": "PublicData", "received": str(type(data))},
                source="GRACEClassifierExplainer.__init__",
            )
        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model: TorchModel, data: BaseData, *args, **kwargs):
        return GRACE(model.raw_model)

    def _generate_counterfactuals(self, sample, *args, **kwargs) -> List[Counterfactual] | Counterfactual:
        feature_min_bounds, feature_max_bounds = self._obtain_feasible_ranges()
        feature_names = self.data.column_names
        feature_is_integer = [pd.api.types.is_integer_dtype(self.data.data[feature]) for feature in self.data.column_names]

        # Check if user provided any additional parameters
        max_features_to_perturb = kwargs.get('max_features_to_perturb', 3)
        overshoot = kwargs.get('overshoot', 0.0001)
        max_iterations = kwargs.get('max_iterations', 50)
        feature_selector = kwargs.get('feature_selector', None)
        feature_alphas = kwargs.get('feature_alphas', np.ones(len(self.data.column_names)))

        #Check how many instances are in the sample
        if isinstance(sample, pd.DataFrame) and sample.shape[0] > 1:
            counterfactuals = []
            for index, row in sample.iterrows():
                original_instance_df, counterfactuals_df = self.explainer.generate_counterfactuals(original_instance = row,
                                             max_features_to_perturb = max_features_to_perturb,
                                             overshoot = overshoot,
                                             max_iterations = max_iterations,
                                             feature_min_bounds = feature_min_bounds,
                                             feature_max_bounds = feature_max_bounds,
                                             feature_is_integer = feature_is_integer,
                                             feature_alphas = feature_alphas,
                                             feature_selector = feature_selector,
                                             feature_names = feature_names)
                counterfactual = Counterfactual(original_instance=original_instance_df,
                                                counterfactual_instance=counterfactuals_df)
                counterfactuals.append(counterfactual)
            return counterfactuals
        else:
            original_instance_df, counterfactuals_df = self.explainer.generate_counterfactuals(original_instance = sample,
                                         max_features_to_perturb = max_features_to_perturb,
                                         overshoot = overshoot,
                                         max_iterations = max_iterations,
                                         feature_min_bounds = feature_min_bounds,
                                         feature_max_bounds = feature_max_bounds,
                                         feature_is_integer = feature_is_integer,
                                         feature_alphas = feature_alphas,
                                         feature_selector = feature_selector,
                                         feature_names = feature_names)
            counterfactual = Counterfactual(original_instance=original_instance_df,
                                            counterfactual_instance=counterfactuals_df)
            return counterfactual

    def _obtain_feasible_ranges(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Extract per-feature minimum and maximum bounds from the data's feasible_values.

        Returns
        -------
        Tuple[Optional[List[float]], Optional[List[float]]]
            A tuple containing:
            - feature_min_bounds: list of minimum values for each feature, or None if not provided
            - feature_max_bounds: list of maximum values for each feature, or None if not provided

            Both lists are ordered according to self.data.column_names.
            If self.data.feasible_values is None, returns (None, None).
        """
        if not self.data.feasible_values:
            return None, None
        else:
            feature_min_bounds = []
            feature_max_bounds = []
            for feature in self.data.column_names:
                if feature in self.data.feasible_values:
                    feasible_value = self.data.feasible_values[feature]
                    feature_min_bounds.append(float(feasible_value[0]))
                    feature_max_bounds.append(float(feasible_value[1]))
                else:
                    try:
                        feature_min_bounds.append(float(self.data.data[feature].min()))
                        feature_max_bounds.append(float(self.data.data[feature].max()))
                    except Exception as e:
                        raise ConfigurationError(
                            message=f"It was not possible to infer min/max bounds for feature '{feature}'. ",
                            param=feature,
                            source="PublicData"
                        ) from e
            return feature_min_bounds, feature_max_bounds
