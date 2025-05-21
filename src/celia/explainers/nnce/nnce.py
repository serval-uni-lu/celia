import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Union

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

    def generate_counterfactuals(self,
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