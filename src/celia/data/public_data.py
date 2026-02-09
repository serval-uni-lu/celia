from typing import Any

import numpy as np
import pandas as pd

from celia.data._base import BaseData
from celia.errors import ConfigurationError


class PublicData(BaseData):
    """
    Concrete implementation of BaseData for public data.
    This class wraps the training data, labels, and metadata required by CELIA explainers.
    It provides structured access to feature types, immutability constraints, and
    feasible values, and performs internal consistency validation upon instantiation.

    Parameters
    ----------
    data : pd.DataFrame
        The feature matrix used to train the model, with shape (n_samples, n_features).

    targets :  pd.Series | np.ndarray
        The target labels corresponding to the training data, with shape (n_samples,).

    target_name : str | None = None
        The name of the target variable. It should be a single string representing the target column in `labels`.

    column_names : list[str] | set[str] | None = None
        A list or set of column names to be considered as features. If not provided, all columns in `data` are used.

    continuous_column_names : list[str] | None = None
        Names of features considered continuous (i.e., real-valued and bounded by a range).

    categorical_column_names : list[str] | None = None
        Names of features considered categorical (i.e., discrete values or classes).

    immutable_column_names : list[str] | None = None
        Names of features that are immutable (i.e., cannot be changed in counterfactuals).

    feasible_values : dict[str, Any] | None = None
        Dictionary mapping each feature to its feasible values:
        - Continuous features: a tuple (min, max) or a List[int | float] of length 2
        - Categorical features: a list of allowed values

    Raises
    ------
    ConfigurationError
        If any consistency check fails (e.g., overlapping feature types, missing values, invalid ranges).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        targets: pd.Series | np.ndarray,
        target_name: str | None = None,
        column_names: list[str] | set[str] | None = None,
        continuous_column_names: list[str] | None = None,
        categorical_column_names: list[str] | None = None,
        immutable_column_names: list[str] | None = None,
        feasible_values: dict[str, Any] | None = None,
    ) -> None:
        self._data = data
        self._column_names = list(data.columns) if column_names is None else list(column_names)
        self._targets = targets
        self._target_name = target_name
        self._continuous_column_names = continuous_column_names
        self._categorical_column_names = categorical_column_names
        self._immutable_column_names = immutable_column_names
        self._feasible_values = feasible_values

        self._validate_data()

    @property
    def data(self) -> pd.DataFrame:
        """The stored feature matrix."""
        return self._data

    @property
    def column_names(self) -> list[str]:
        """the set of column names in the dataset. If not provided, it defaults to all columns in `data`."""
        return self._column_names

    @property
    def targets(self) -> pd.Series:
        """The stored target labels."""
        return self._targets

    @property
    def target_name(self) -> str:
        """The target label name"""
        return self._target_name

    @property
    def continuous_column_names(self) -> list[str] | None:
        """The list of continuous feature names."""
        return self._continuous_column_names

    @property
    def categorical_column_names(self) -> list[str] | None:
        """The list of categorical feature names."""
        return self._categorical_column_names

    @property
    def immutable_column_names(self) -> list[str] | None:
        """The list of immutable feature names."""
        return self._immutable_column_names

    @property
    def feasible_values(self) -> dict[str, Any] | None:
        """The feasible values for all relevant features."""
        return self._feasible_values

    def _check_feature_names_exist(self, feature_list: list[str], name: str) -> None:
        """
        Validate that all feature names in a list exist in the dataset.

        Parameters
        ----------
        feature_list : List[str]
            A list of feature names to validate.

        name : str
            A human-readable label for the feature type (e.g. 'continuous', 'categorical')
            used in error messages.

        Raises
        ------
        ValueError
            If any feature in the list is not found in self.data.columns.
        """
        missing = set(feature_list) - set(self.data.columns)
        if missing:
            message = f"The following {name} features are not in the dataset: {sorted(missing)}"
            raise ConfigurationError(
                message=message,
                param=name,
                config={"expected_columns": list(self.data.columns), "missing_features": list(missing)},
                hint=f"Ensure all {name} features exist as columns in the provided dataset.",
            )

    @staticmethod
    def _check_data_label_alignment(data: pd.DataFrame, targets: pd.Series) -> None:
        """
        Ensure that the number of samples in data and targets match.

        Parameters
        ----------
        data : pd.DataFrame
            The feature matrix used to train the model.
        targets : pd.Series
            The target values corresponding to the data.

        Raises
        ------
        ConfigurationError
            If the number of rows in data does not match the number of targets.
        """
        if len(data) != len(targets):
            message = (
                "Data and targets must have the same number of instances: "
                f"{len(data)} rows in data vs {len(targets)} targets."
            )
            raise ConfigurationError(
                message=message,
                param="targets",
                config={"data_rows": len(data), "target_rows": len(targets)},
                hint="Ensure targets contain exactly one entry per row in the DataFrame.",
            )

    def _validate_inputs(self) -> None:
        """
        Validate the type correctness of all provided inputs.
        """
        if not isinstance(self.data, pd.DataFrame):
            message = "'data' must be a pandas DataFrame, got "
            raise ConfigurationError(
                message=f"{message}{type(self.data).__name__}",
                param="data",
                config={"provided_type": type(self.data).__name__},
                hint="Ensure you pass a pandas DataFrame containing your features.",
            )

        if not isinstance(self.targets, (pd.Series, np.ndarray)):
            message = "'targets' must be a pandas Series or numpy ndarray, got "
            raise ConfigurationError(
                message=f"{message}{type(self.targets).__name__}",
                param="targets",
                config={"provided_type": type(self.targets).__name__},
                hint="Use a pandas Series or numpy ndarray for targets.",
            )

        if self.target_name is not None and not isinstance(self.target_name, str):
            message = "'target_name' must be a string, got "
            raise ConfigurationError(
                message=f"{message}{type(self.target_name).__name__}",
                param="target_name",
                config={"provided_type": type(self.target_name).__name__},
                hint="Provide the name of the target column as a string.",
            )

        if self.continuous_column_names is not None and not isinstance(self.continuous_column_names, list):
            message = "'continuous_column_names' must be a list of strings or None, got "
            raise ConfigurationError(
                message=f"{message}{type(self.continuous_column_names).__name__}",
                param="continuous_column_names",
                config={"provided_type": type(self.continuous_column_names).__name__},
                hint="Pass a list of continuous feature names, or None.",
            )

        if self.categorical_column_names is not None and not isinstance(self.categorical_column_names, list):
            message = "'categorical_column_names' must be a list of strings or None, got "
            raise ConfigurationError(
                message=f"{message}{type(self.categorical_column_names).__name__}",
                param="categorical_column_names",
                config={"provided_type": type(self.categorical_column_names).__name__},
                hint="Pass a list of categorical feature names, or None.",
            )

        if self.immutable_column_names is not None and not isinstance(self.immutable_column_names, list):
            message = "'immutable_column_names' must be a list of strings or None, got "
            raise ConfigurationError(
                message=f"{message}{type(self.immutable_column_names).__name__}",
                param="immutable_column_names",
                config={"provided_type": type(self.immutable_column_names).__name__},
                hint="Pass a list of immutable feature names, or None.",
            )

        if self.feasible_values is not None and not isinstance(self.feasible_values, dict):
            message = "'feasible_values' must be a dictionary or None, got "
            raise ConfigurationError(
                message=f"{message}{type(self.feasible_values).__name__}",
                param="feasible_values",
                config={"provided_type": type(self.feasible_values).__name__},
                hint="Provide feasible values as a dictionary mapping feature names to valid ranges or categories.",
            )

    def _validate_data(self) -> None:
        """
        Validate the dataset and its properties.

        This method checks that:
        - All feature names in continuous_column_names, categorical_column_names, and immutable_column_names lists exist in the data.
        - The number of samples in data matches the number of targets.
        - Feasible values for features are valid.
        - No features are shared between continuous_column_names and categorical_column_names lists.

        Raises
        ------
        ConfigurationError
            If any validation check fails.
        """
        self._validate_inputs()

        if self.target_name is not None:
            if isinstance(self.targets, pd.Series):
                self._targets.name = self.target_name
            else:
                self._targets = pd.Series(self.targets, name=self.target_name)
        else:
            if isinstance(self.targets, pd.Series):
                if self.targets.name is not None:
                    self._target_name = self.targets.name
                else:
                    self._target_name = "target"
                    self._targets.name = "target"
            else:
                message = "When 'targets' is provided as a numpy array, you must also supply a 'target_name'."
                raise ConfigurationError(
                    message=message,
                    param="target_name",
                    config={"targets_type": "ndarray"},
                    hint="Pass a string to 'target_name' so CELIA can build a named Series from it.",
                )

        self._check_data_label_alignment(self.data, self.targets)

        if self.continuous_column_names is not None and self.categorical_column_names is not None:
            self._check_feature_overlap(self.continuous_column_names, self.categorical_column_names)

        if self.continuous_column_names is not None:
            self._check_feature_names_exist(self.continuous_column_names, "continuous")

        if self.categorical_column_names is not None:
            self._check_feature_names_exist(self.categorical_column_names, "categorical")

        if self.immutable_column_names is not None:
            self._check_feature_names_exist(self.immutable_column_names, "immutable")

        if self.feasible_values is not None:
            self._check_range_dict_validity(self.feasible_values)
