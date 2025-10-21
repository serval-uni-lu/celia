import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Union, Tuple

from celia.counterfactuals import Counterfactual
from celia.explainers import RegressorExplainer
from celia.model import BaseModel
from celia.data import PublicData
from celia._errors import (ConfigurationError,
                           MethodValueError,
                           NoCounterfactualsFound)

class NearestNeighborCE:
    """Generate counterfactuals using a Nearest Neighbor search.

        Parameters
        ----------
        train_data : pd.DataFrame
            The training data used to find nearest neighbors. Must contain the same
            feature columns expected by `model.predict`. We assume train_data contains the column with the target variable.
        model : BaseModel
            The predictive model used to compute outputs.
        target_name : str, optional
            The target variable name, by default "prediction".
        task_type : str, optional
            Either "classification" or "regression", by default "classification".
        verbose : bool, optional
            Whether to print intermediate steps, by default False.

        Raises
        ------
        MethodValueError

        """
    def __init__(self,
                 train_data: pd.DataFrame,
                 model,
                 target_name: str = "prediction",
                 task_type: str = 'classification',
                 verbose=False):

        self._validate_init_dtypes(train_data, target_name, task_type, verbose)

        self.train_data = train_data.reset_index(drop=True)
        self.model = model
        self.target_name = target_name
        self.task_type = task_type
        self.verbose = verbose

    def nnce_generate_counterfactuals(self,
                                 instance: Union[pd.Series, pd.DataFrame],
                                 desired_output: Union[int, float, List[float]],
                                 n_counterfactuals: int = 1,
                                 mutable_features: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame] | None:

        instance, mutable_features, desired_output, n_counterfactuals = self._validate_generate_counterfactuals_args(
            instance, desired_output, n_counterfactuals, mutable_features)

        # Step 1: Current prediction
        instance_df = pd.DataFrame([instance])
        current_pred = self.model.predict(instance_df)[0]
        if self.verbose:
            print("current_pred", current_pred)

        # Step 2: Identify candidates
        immutable_features = [col for col in self.train_data.columns if col not in mutable_features]

        # Select candidates where immutable features match
        mask = (self.train_data[immutable_features] == instance[immutable_features]).all(axis=1)
        candidates = self.train_data[mask].copy()
        if self.verbose:
            print(f'Available Candidates in Mutable Features: {len(candidates)}')

        if candidates.empty:
            raise NoCounterfactualsFound(message="No candidates found with matching immutable features.")

        # Step 3: Predict candidate outputs
        candidate_preds = self.model.predict(candidates)

        # Step 4: Filter candidates based on desired output
        if self.task_type == 'classification':
            valid_idx = np.where(candidate_preds == desired_output)[0]
        else:
            min_val, max_val = desired_output
            if self.verbose:
                print(f'Desired output: {min_val} - {max_val}')
            valid_idx = np.where((candidate_preds >= min_val) & (candidate_preds <= max_val))[0]

        valid_candidates = candidates.iloc[valid_idx]

        if valid_candidates.empty:
            raise NoCounterfactualsFound(message="No candidates found matching the desired output.")

        # Step 5: Distance calculation in mutable feature space
        instance_mutable = instance[mutable_features].values.reshape(1, -1)
        candidates_mutable = valid_candidates[mutable_features].values

        distances = pairwise_distances(instance_mutable, candidates_mutable)[0]

        nearest_indices = np.argsort(distances)[:n_counterfactuals]
        counterfactuals: pd.DataFrame = valid_candidates.iloc[nearest_indices].copy()

        # Add a column with the model's prediction for each counterfactual
        counterfactual_preds = self.model.predict(counterfactuals)
        counterfactuals[self.target_name] = counterfactual_preds
        instance_df[self.target_name] = current_pred

        return instance_df, counterfactuals.reset_index(drop=True)

    def _validate_init_dtypes(self, train_data: pd.DataFrame, target_name: str, task_type: str, verbose: bool):
        """Validate the initialization parameters for NearestNeighborCE."""

        if not isinstance(train_data, pd.DataFrame):
            raise MethodValueError(
                message="Invalid train_data. Expected a pandas DataFrame.",
                config={"train_data_type": type(train_data)},
                param="train_data",
                hint="Ensure train_data is a pandas DataFrame.",
                source="NearestNeighborCE.__init__",
            )

        if not isinstance(target_name, str):
            raise MethodValueError(
                message="Invalid target_name. Expected a string.",
                config={"target_name_type": type(target_name)},
                param="target_name",
                hint="Ensure target_name is a string.",
                source="NearestNeighborCE.__init__",
            )

        if task_type not in ["classification", "regression"]:
            raise MethodValueError(
                message="Invalid task_type. Expected 'classification' or 'regression'.",
                config={"task_type": task_type},
                param="task_type",
                hint="Use task_type='classification' or task_type='regression'.",
                source="NearestNeighborCE.__init__",
            )

        if not isinstance(verbose, bool):
            raise MethodValueError(
                message="Invalid verbose flag. Expected a boolean.",
                config={"verbose_type": type(verbose)},
                param="verbose",
                hint="Ensure verbose is a boolean value (True or False).",
                source="NearestNeighborCE.__init__",
            )

    def _validate_generate_counterfactuals_args(self, instance: Union[pd.Series, pd.DataFrame],
                                 desired_output: Union[int, float, List[float]],
                                 n_counterfactuals: int = 1,
                                 mutable_features: List[str] = None) -> Tuple[pd.Series, List[str], Union[int, float, List[float]], int]:
        """Validate and normalize the arguments for nnce_generate_counterfactuals.

        Returns
        -------
        Tuple[pd.Series, List[str], Union[int, float, List[float]], int]
            Validated instance, mutable feature list, desired output, and number of counterfactuals.
        """

        if isinstance(instance, pd.DataFrame):
            if instance.shape[0] != 1:
                raise MethodValueError(
                    message="`instance` must be a single-row DataFrame or a Series.",
                    config={"rows": int(instance.shape[0])},
                    param="instance",
                    hint="Select exactly one row (e.g., df.iloc[[idx]] or df.loc[[id]]).",
                    source="NearestNeighborCE.nnce_generate_counterfactuals",
                )
            instance = instance.iloc[0]
        elif not isinstance(instance, pd.Series):
            raise MethodValueError(
                message="`instance` must be a pandas Series or single-row DataFrame.",
                config={"received_type": type(instance).__name__},
                param="instance",
                hint="Provide a pandas Series or df.iloc[[i]] for a single row.",
                source="NearestNeighborCE.nnce_generate_counterfactuals",
            )

        if mutable_features is None:
            mutable_features = list(self.train_data.columns)

        missing = [feature for feature in mutable_features if feature not in self.train_data.columns]
        if missing:
            raise MethodValueError(
                message="All mutable features must exist in the training data.",
                config={"mutable_features": mutable_features, "missing_in_train": missing},
                param="mutable_features",
                hint="Remove unknown columns or align train_data columns with `mutable_features`.",
                source="NearestNeighborCE.nnce_generate_counterfactuals",
            )

        if n_counterfactuals <= 0 or not isinstance(n_counterfactuals, int):
            raise MethodValueError(
                message="`n_counterfactuals` must be a positive integer.",
                config={"n_counterfactuals": n_counterfactuals},
                param="n_counterfactuals",
                hint="Set n_counterfactuals to a positive integer (e.g., 1, 2, 3...).",
                source="NearestNeighborCE.nnce_generate_counterfactuals",
            )

        if not isinstance(desired_output, (int, float, list)):
            raise MethodValueError(
                message="`desired_output` must be an int, float, or list of two floats for regression.",
                config={"desired_output_type": type(desired_output)},
                param="desired_output",
                hint="For classification, use an int or float. For regression, use a list [min, max].",
                source="NearestNeighborCE.nnce_generate_counterfactuals",
            )
        else:
            if self.task_type == 'regression' and not (isinstance(desired_output, list) and len(desired_output) == 2 and all(isinstance(x, (int, float)) for x in desired_output)):
                raise MethodValueError(
                    message="For regression, `desired_output` must be a list of two floats [min, max].",
                    config={"desired_output": desired_output},
                    param="desired_output",
                    hint="Use a list [min_value, max_value] to specify the desired output range.",
                    source="NearestNeighborCE.nnce_generate_counterfactuals",
                )
            if self.task_type == 'classification':
                train_classes = np.unique(self.train_data[self.target_name])
                if desired_output not in train_classes:
                    raise MethodValueError(
                        message="For classification, `desired_output` must be a valid class present in training data predictions.",
                        config={"desired_output": desired_output, "valid_classes": train_classes.tolist()},
                        param="desired_output",
                        hint="Set desired_output to one of the valid classes from training data predictions.",
                        source="NearestNeighborCE.nnce_generate_counterfactuals",
                    )

        return instance, mutable_features, desired_output, n_counterfactuals


class NNCERegressorExplainer(RegressorExplainer):
    """
    Concrete implementation of RegressorExplainer for Nearest Neighbor-based Counterfactual Explanations.
    This class wraps the NearestNeighborCE method to generate counterfactual explanations
    for regression tasks using public training data. It enforces the use of PublicData
    and validates that the input data meets the expected structure required by the explainer.

    Parameters
    ----------
    model : BaseModel
        The predictive regression model to be explained. Must implement the BaseModel interface
        with a `predict` method.

    data : PublicData
        The public dataset object, containing the training data, target labels, and metadata
        such as feature types and feasible values.

    *args : Any
        Additional positional arguments passed to the parent RegressorExplainer class.

    **kwargs : Any
        Additional keyword arguments. Supports the following optional keys:
        - verbose (bool): If True, enables verbose output during CE generation.

    Raises
    ------
    ConfigurationError

    Attributes
    ----------
    model : BaseModel
        The regression model to be explained.

    data : PublicData
        The dataset used to generate counterfactual explanations.

    explainer : NearestNeighborCE
        Instance of the NearestNeighborCE class initialized with training data, model,
        and target variable for regression tasks.
    """

    def __init__(self, model: BaseModel, data: PublicData, *args, **kwargs):
        if not isinstance(data, PublicData):
            raise ConfigurationError(
                message="`data` must be an instance of PublicData.",
                config={"received_type": type(data).__name__},
                param="data",
                hint="Instantiate and pass celia.data.PublicData(...).",
                source="NNCERegressorExplainer.__init__",
            )
        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model: BaseModel, data: PublicData, *args, **kwargs):
        target_name = data.target_name
        train_data = data.data
        verbose = kwargs.pop("verbose", False)
        return NearestNeighborCE(train_data, model, target_name, task_type='regression', verbose=verbose)

    def _generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series],
                                  target_range: Union[List[float], Tuple[float, float]],
                                 *args, **kwargs) -> Counterfactual:

        # Check if user provided n_counterfactuals in kwargs, else default to 1
        n_counterfactuals = kwargs.pop("n_counterfactuals", 1)

        # Check if self.data has mutable_features, else default to None
        if self.data.immutable_column_names is None:
            mutable = self.data.column_names
        else:
            mutable = [col for col in self.data.column_names if col not in self.data.immutable_column_names]

        instance_df, results = self.explainer.nnce_generate_counterfactuals(instance=sample,
                                                     desired_output=target_range,
                                                     n_counterfactuals=n_counterfactuals,
                                                     mutable_features=mutable)

        counterfactual = Counterfactual(original_instance=instance_df, counterfactual_instance=results)
        return counterfactual

    def _validate_sample(self, sample: Union[pd.DataFrame, pd.Series],
                         target_range: Union[List[float], Tuple[float, float]] = None, *args, **kwargs) -> None:

        """Validate that NNCE receives exactly one instance at a time.

       Parameters
       ----------
       sample : Union[pd.DataFrame, pd.Series]
           The input sample to validate.
       target_range : Union[List[float], Tuple[float, float]], optional
           Desired prediction interval [min_value, max_value] for regression.

        Raises
        ------
        ConfigurationError
       """
        if not isinstance(sample, (pd.DataFrame, pd.Series)):
            raise ConfigurationError(
                message="Invalid input type for `sample`. Expected a pandas DataFrame or Series.",
                config={"received_type": type(sample).__name__},
                param="sample",
                hint="Pass either a pandas Series or a single-row DataFrame (e.g., df.iloc[[i]]).",
                source="NNCERegressorExplainer._validate_sample",
            )

        if isinstance(sample, pd.DataFrame) and sample.shape[0] > 1:
            raise ConfigurationError(
                message="`sample` must contain exactly one instance.",
                config={
                    "rows_provided": int(sample.shape[0]),
                    "expected_rows": 1
                },
                param="sample",
                hint="Select a single instance (e.g., df.iloc[[i]] or df.head(1)).",
                source="NNCERegressorExplainer._validate_sample",
            )

