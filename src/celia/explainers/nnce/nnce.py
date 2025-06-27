import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Union, Tuple
from celia.explainers import RegressorExplainer
from celia.model import BaseModel
from celia.data import PublicData
from celia.errors.user_configuration_erros import CELIAConfigurationError


class NearestNeighborCE:
    def __init__(self,
                 train_data: pd.DataFrame,
                 model,
                 target_name: str = "prediction",
                 task_type: str = 'classification',
                 verbose=False):
        assert task_type in ['classification', 'regression'], "task_type must be 'classification' or 'regression'"
        self.train_data = train_data.reset_index(drop=True)
        self.model = model
        self.target_name = target_name
        self.task_type = task_type
        self.verbose = verbose

    def nnce_generate_counterfactuals(self,
                                 instance: pd.Series,
                                 desired_output: Union[int, float, List[float]],
                                 n_counterfactuals: int = 1,
                                 mutable_features: List[str] = None):

        if isinstance(instance, pd.DataFrame):
            # Convert single-row DataFrame to Series
            instance = instance.iloc[0]
        # If mutable_features is not given, assume all features are mutable
        if mutable_features is None:
            mutable_features = list(self.train_data.columns)

        assert set(mutable_features).issubset(
            set(self.train_data.columns)), "All mutable features must exist in training data"

        # Step 1: Current prediction
        instance_df = pd.DataFrame([instance])
        current_pred = self.model.predict(instance_df)[0]
        if self.verbose: print("current_pred", current_pred)
        # Step 2: Identify candidates
        immutable_features = [col for col in self.train_data.columns if col not in mutable_features]

        # Select candidates where immutable features match
        mask = (self.train_data[immutable_features] == instance[immutable_features]).all(axis=1)
        candidates = self.train_data[mask].copy()
        if self.verbose: print(f'Available Candidates in Mutable Features: {len(candidates)}')

        if candidates.empty:
            raise ValueError("No candidates with matching immutable features found.")

        # Step 3: Predict candidate outputs
        candidate_preds = self.model.predict(candidates)

        # Step 4: Filter candidates based on desired output
        if self.task_type == 'classification':
            valid_idx = np.where(candidate_preds == desired_output)[0]
        else:  # Regression
            assert isinstance(desired_output, list) and len(desired_output) == 2, \
                "For regression, desired_output must be a list [min_value, max_value]"
            min_val, max_val = desired_output
            if self.verbose: print(f'Desired output: {min_val} - {max_val}')
            valid_idx = np.where((candidate_preds >= min_val) & (candidate_preds <= max_val))[0]

        valid_candidates = candidates.iloc[valid_idx]

        if valid_candidates.empty:
            return None

        # Step 5: Distance calculation in mutable feature space
        instance_mutable = instance[mutable_features].values.reshape(1, -1)
        candidates_mutable = valid_candidates[mutable_features].values

        distances = pairwise_distances(instance_mutable, candidates_mutable)[0]

        nearest_indices = np.argsort(distances)[:n_counterfactuals]
        counterfactuals = valid_candidates.iloc[nearest_indices].copy()

        # Add a column with the model's prediction for each counterfactual
        counterfactual_preds = self.model.predict(counterfactuals)
        counterfactuals[self.target_name] = counterfactual_preds

        return counterfactuals.reset_index(drop=True)

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
    ValueError
        If the provided data is not an instance of PublicData.

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
        #Assert that data is an instance of PublicData
        if not isinstance(data, PublicData):
            raise CELIAConfigurationError("data must be an instance of PublicData")
        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model: BaseModel, data: PublicData, *args, **kwargs):
        target_name = data.target_name
        train_data = data.data
        verbose = kwargs.pop("verbose", False)
        return NearestNeighborCE(train_data, model, target_name, task_type='regression', verbose=verbose)

    def _generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series],
                                  target_range: Union[List[float], Tuple[float, float]],
                                 *args, **kwargs) -> pd.DataFrame:

        # Check if user provided n_counterfactuals in kwargs, else default to 1
        n_counterfactuals = kwargs.pop("n_counterfactuals", 1)

        # Check if self.data has mutable_features, else default to None
        if self.data.immutable_column_names is None:
            mutable = self.data.column_names
        else:
            mutable = [col for col in self.data.column_names if col not in self.data.immutable_column_names]

        return self.explainer.nnce_generate_counterfactuals(instance=sample,
                                                     desired_output=target_range,
                                                     n_counterfactuals=n_counterfactuals,
                                                     mutable_features=mutable)

    def _validate_sample(self, sample: Union[pd.DataFrame, pd.Series],
                         target_range: Union[List[float], Tuple[float, float]] = None, *args, **kwargs) -> None:

        # Raise error if sample has more than one row if it's a DataFrame
        if isinstance(sample, pd.DataFrame) and sample.shape[0] > 1:
            raise CELIAConfigurationError("NNCE only explains one instance at a time."
                             "Sample must be a single row DataFrame or Series")
