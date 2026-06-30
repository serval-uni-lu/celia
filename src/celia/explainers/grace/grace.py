import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from celia._utils.dependencies import requires_torch_class
from celia.counterfactuals import Counterfactual
from celia.data import Data
from celia.data._base import BaseData
from celia.errors import ConfigurationError
from celia.explainers import ClassifierExplainer
from celia.model import BaseModel, TorchModel

if TYPE_CHECKING:
    from torch import Tensor


@requires_torch_class
class GRACE:
    def __init__(self, model: TorchModel) -> None:
        self.model = model

    @staticmethod
    def _place_perturbation(
        perturbation: np.ndarray,
        feature_indices: np.ndarray | list[int],
        input_shape: tuple[int, ...],
    ) -> np.ndarray:
        """
        Insert perturbation values at the specified feature indices for an input of the given shape.
        """
        mask = np.zeros(input_shape)
        mask[0, feature_indices] = perturbation
        return mask

    def _naive_gradient_attack(
        self,
        original_instance: "Tensor",
        num_features: int,
        num_classes: int = 2,
        overshoot: float = 0.0001,
        max_iterations: int = 50,
        feature_min_bounds: list[float] | None = None,
        feature_max_bounds: list[float] | None = None,
        feature_is_integer: list[bool] | None = None,
        feature_alphas: list[float] | np.ndarray | None = None,
        feature_selector: object | None = None,
    ) -> tuple[int, np.ndarray, np.ndarray, list[int], int]:
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

        predictions = self.model.predict_proba(original_instance).flatten()
        sorted_class_indices: list[int] = predictions.argsort()[::-1]

        original_class = int(self.model.predict(original_instance).item())

        input_clone = copy.deepcopy(original_instance)
        cumulative_perturbation = np.zeros(num_features, dtype=np.float32)
        minimal_direction = np.zeros(num_features, dtype=np.float32)

        perturbed_input = input_clone.clone().detach().requires_grad_(True)
        outputs = self.model.raw_model(perturbed_input)

        tolerance = 1e-8
        iteration_count = 0
        continue_search = True

        # Initial backward call
        outputs[0, original_class].backward(retain_graph=True)
        selected_features: list[int] = []

        counterfactual_array = input_clone.detach().numpy().astype(np.float32, copy=True)

        while continue_search:
            outputs[0, original_class].backward(retain_graph=True)
            gradients = cast("Tensor", perturbed_input.grad).detach().numpy().copy()

            if len(selected_features) == 0:
                sorted_features = np.argsort(gradients)[::-1][0]
                if feature_selector:
                    if not hasattr(feature_selector, "select"):
                        message = "feature_selector must implement a `.select(indices, num_features)` method."
                        raise ConfigurationError(
                            message=message,
                            param="feature_selector",
                            hint="Provide an object with a `select` method that returns a list of feature indices.",
                        )

                    selector: Any = feature_selector
                    selected_features = selector.select(sorted_features, num_features)  # type: ignore[call-non-callable]
                else:
                    selected_features = sorted_features[:num_features].tolist()

            base_gradient = gradients[0, selected_features]
            best_step_size = np.inf

            for class_index in range(1, num_classes):
                if perturbed_input.grad is not None:
                    perturbed_input.grad.zero_()

                outputs[0, sorted_class_indices[class_index]].backward(retain_graph=True)
                current_grad = cast("Tensor", perturbed_input.grad).detach().numpy().copy()[0, selected_features]

                gradient_diff = current_grad - base_gradient
                logit_diff = (
                    (outputs[0, sorted_class_indices[class_index]] - outputs[0, original_class]).detach().numpy()
                )

                step_size = abs(logit_diff) / (np.linalg.norm(gradient_diff.flatten()) + tolerance)

                if step_size < best_step_size:
                    best_step_size = step_size
                    minimal_direction = gradient_diff

            scaled_step = (best_step_size + 1e-4) * minimal_direction / (np.linalg.norm(minimal_direction) + tolerance)
            cumulative_perturbation = cumulative_perturbation + scaled_step.astype(np.float32)

            mask = self._place_perturbation(scaled_step, selected_features, perturbed_input.shape)
            counterfactual_array += (1 + overshoot) * mask

            counterfactual_array = self._maintain_domain(
                counterfactual_array,
                selected_features,
                feature_alphas,
                feature_is_integer,
                feature_min_bounds,
                feature_max_bounds,
            )

            counterfactual_tensor = torch.from_numpy(counterfactual_array).float()
            perturbed_input = counterfactual_tensor.clone().detach().requires_grad_(True)
            outputs = self.model.raw_model(perturbed_input)

            counterfactual_class = int(np.argmax(outputs.detach().numpy().flatten()))
            iteration_count += 1

            if counterfactual_class != original_class or iteration_count > max_iterations:
                continue_search = False

        final_cf_array = counterfactual_array.copy()
        final_cf_tensor = torch.from_numpy(final_cf_array).float()
        final_input = final_cf_tensor.clone().detach().requires_grad_(True)

        outputs = self.model.raw_model(final_input)
        counterfactual_class = int(np.argmax(outputs.detach().numpy().flatten()))

        return (
            counterfactual_class,
            cumulative_perturbation,
            final_cf_array.flatten(),
            selected_features,
            iteration_count,
        )

    @staticmethod
    def _maintain_domain(
        counterfactual_array: np.ndarray,
        selected_features: list[int],
        feature_alphas: list[float],
        feature_is_integer: list[bool] | None = None,
        feature_min_bounds: list[float] | None = None,
        feature_max_bounds: list[float] | None = None,
    ) -> np.ndarray:
        """
        Adjust counterfactual values to respect integer constraints and feature bounds.
        """

        if feature_is_integer and any(feature_is_integer):
            selected_set = set(selected_features)

            for idx, is_integer in enumerate(feature_is_integer):
                if not is_integer or idx not in selected_set:
                    continue

                step = float(feature_alphas[idx])
                if not np.isfinite(step) or step <= 0.0:
                    continue

                counterfactual_array[:, idx] = np.round(counterfactual_array[:, idx] / step) * step

        if feature_min_bounds is not None:
            counterfactual_array = np.maximum(counterfactual_array, feature_min_bounds)

        if feature_max_bounds is not None:
            counterfactual_array = np.minimum(counterfactual_array, feature_max_bounds)

        return counterfactual_array

    def generate_counterfactuals(
        self,
        original_instance: "Tensor | np.ndarray | pd.DataFrame | pd.Series",
        max_features_to_perturb: int,
        overshoot: float = 0.0001,
        max_iterations: int = 50,
        feature_min_bounds: list[float] | None = None,
        feature_max_bounds: list[float] | None = None,
        feature_is_integer: list[bool] | None = None,
        feature_alphas: list[float] | np.ndarray | None = None,
        feature_selector: object | None = None,
        feature_names: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        # Normalize incoming instance to Tensor
        if isinstance(original_instance, pd.DataFrame):
            original_tensor = torch.tensor(original_instance.to_numpy(), dtype=torch.float32)
        elif isinstance(original_instance, pd.Series):
            original_tensor = torch.as_tensor(original_instance.to_numpy(), dtype=torch.float32).reshape(1, -1)
        elif isinstance(original_instance, np.ndarray):
            original_tensor = torch.tensor(original_instance, dtype=torch.float32)
        else:
            original_tensor = original_instance  # already Tensor

        original_class = int(self.model.predict(original_tensor).item())

        for k in range(1, max_features_to_perturb + 1):
            (
                counterfactual_class,
                cumulative,
                counterfactual_array,
                selected_features,
                iteration_count,
            ) = self._naive_gradient_attack(
                original_instance=original_tensor,
                num_features=k,
                overshoot=overshoot,
                max_iterations=max_iterations,
                feature_min_bounds=feature_min_bounds,
                feature_max_bounds=feature_max_bounds,
                feature_is_integer=feature_is_integer,
                feature_alphas=feature_alphas,
                feature_selector=feature_selector,
            )

            if counterfactual_class != original_class:
                break

        original_np = original_tensor.detach().numpy().flatten()
        cf_np = counterfactual_array.flatten()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(original_np))]

        original_row = {feature_names[i]: float(original_np[i]) for i in range(len(original_np))}
        original_row["Prediction"] = original_class

        cf_row = {feature_names[i]: float(cf_np[i]) for i in range(len(cf_np))}
        cf_row["Prediction"] = counterfactual_class

        return (
            pd.DataFrame([original_row]),
            pd.DataFrame([cf_row]),
        )


class GRACEClassifierExplainer(ClassifierExplainer):
    """
    GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model’s Prediction.
    Thai Le, Suhang Wang, Dongwon Lee. 26th ACM SIGKDD Int’l Conf. on Knowledge Discovery and Data Mining (KDD), Virtual. August 2020.
    """

    def __init__(self, model: BaseModel, data: BaseData, *args: object, **kwargs: object) -> None:
        if not isinstance(model, TorchModel):
            raise ConfigurationError(
                message="GRACEClassifierExplainer requires a TorchModel wrapping a torch.nn.Module.",
                param="model",
                config={"expected": "TorchModel", "received": type(model).__name__},
                hint="Wrap your torch.nn.Module with `TorchModel(model=...)`",
                source="GRACEClassifierExplainer.__init__",
            )

        if not isinstance(data, Data):
            raise ConfigurationError(
                message="GRACEClassifierExplainer requires data to be an instance of Data.",
                param="data",
                config={"expected": "Data", "received": type(data).__name__},
                source="GRACEClassifierExplainer.__init__",
            )

        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(
        self,
        model: TorchModel,
        data: BaseData,
        *args: object,
        **kwargs: object,
    ) -> GRACE:
        return GRACE(model)

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        feature_min_bounds, feature_max_bounds = self._obtain_feasible_ranges()
        feature_names = self.data.column_names
        feature_is_integer = [
            pd.api.types.is_integer_dtype(self.data.data[feature]) for feature in self.data.column_names
        ]

        max_features_to_perturb = kwargs.get("max_features_to_perturb", 3)
        overshoot = kwargs.get("overshoot", 0.0001)
        max_iterations = kwargs.get("max_iterations", 50)
        feature_selector = kwargs.get("feature_selector")
        feature_alphas = kwargs.get(
            "feature_alphas",
            np.ones(len(self.data.column_names)),
        )

        # Multiple rows
        if isinstance(sample, pd.DataFrame) and sample.shape[0] > 1:
            cf_list: list[Counterfactual] = []

            for _, row in sample.iterrows():
                original_df, cf_df = self.explainer.generate_counterfactuals(
                    original_instance=row,
                    max_features_to_perturb=max_features_to_perturb,
                    overshoot=overshoot,
                    max_iterations=max_iterations,
                    feature_min_bounds=feature_min_bounds,
                    feature_max_bounds=feature_max_bounds,
                    feature_is_integer=feature_is_integer,
                    feature_alphas=feature_alphas,
                    feature_selector=feature_selector,
                    feature_names=feature_names,
                )
                cf_list.append(
                    Counterfactual(
                        original_instance=original_df.drop(columns=["Prediction"]),
                        counterfactual_instance=cf_df.drop(columns=["Prediction"]),
                        original_prediction=original_df["Prediction"].iloc[0],
                        counterfactual_prediction=cf_df["Prediction"].iloc[0],
                    )
                )
            return cf_list

        # Single instance
        original_df, cf_df = self.explainer.generate_counterfactuals(
            original_instance=sample,
            max_features_to_perturb=max_features_to_perturb,
            overshoot=overshoot,
            max_iterations=max_iterations,
            feature_min_bounds=feature_min_bounds,
            feature_max_bounds=feature_max_bounds,
            feature_is_integer=feature_is_integer,
            feature_alphas=feature_alphas,
            feature_selector=feature_selector,
            feature_names=feature_names,
        )

        return Counterfactual(
            original_instance=original_df.drop(columns=["Prediction"]),
            counterfactual_instance=cf_df.drop(columns=["Prediction"]),
            original_prediction=original_df["Prediction"].iloc[0],
            counterfactual_prediction=cf_df["Prediction"].iloc[0],
        )

    def _obtain_feasible_ranges(
        self,
    ) -> tuple[list[float] | None, list[float] | None]:
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

        feature_min_bounds: list[float] = []
        feature_max_bounds: list[float] = []

        for feature in self.data.column_names:
            if feature in self.data.feasible_values:
                feasible_val = self.data.feasible_values[feature]
                feature_min_bounds.append(float(feasible_val[0]))
                feature_max_bounds.append(float(feasible_val[1]))
            else:
                try:
                    feature_min_bounds.append(float(self.data.data[feature].min()))
                    feature_max_bounds.append(float(self.data.data[feature].max()))
                except Exception as e:
                    message = f"It was not possible to infer min/max bounds for feature '{feature}'."
                    raise ConfigurationError(
                        message=message,
                        param=feature,
                        source="Data",
                    ) from e

        return feature_min_bounds, feature_max_bounds
