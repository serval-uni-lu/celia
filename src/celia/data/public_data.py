import pandas as pd
from celia.errors.data_handling_errors import UnSupportedDataTypeError
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from celia.data._base import BaseData


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

    targets : pd.Series
        The target labels corresponding to the training data, with shape (n_samples,).

    target_name : str
        The name of the target variable. It should be a single string representing the target column in `labels`.

    column_names : Optional[Union[List[str], Set[str]]]
        A list or set of column names to be considered as features. If not provided, all columns in `data` are used.

    continuous : List[str]
        Names of features considered continuous (i.e., real-valued and bounded by a range).

    categorical : List[str]
        Names of features considered categorical (i.e., discrete values or classes).

    immutable : List[str]
        Names of features that are immutable (i.e., cannot be changed in counterfactuals).

    feasible_values : Dict[str, Any]
        Dictionary mapping each feature to its feasible values:
        - Continuous features: a tuple (min, max)
        - Categorical features: a list of allowed values


    Raises
    ------
    ValueError
        If any consistency check fails (e.g., overlapping feature types, missing values, invalid ranges).

    Attributes
    ----------
    data : pd.DataFrame
        Returns the stored feature matrix.

    targets : pd.Series
        Returns the stored target labels.

    target_name : str
        Returns the name of target variable.

    column_names: Optional[Union[List[str], Set[str]]]
        Returns the set of column names in the dataset. If not provided, it defaults to all columns in `data`.

    continuous : List[str]
        Returns the list of continuous feature names.

    categorical : List[str]
        Returns the list of categorical feature names.

    immutable : List[str]
        Returns the list of immutable feature names.

    feasible_values : Dict[str, Any]
        Returns the feasible values for all relevant features.
    """

    def __init__(
        self,
        data: pd.DataFrame, # NOTE: To accept other data types later e.g dict
        targets: pd.Series | List[str] | Tuple[str], # NOTE: Not a list?
        target_name: str,
        column_names: Optional[Union[List[str], Set[str]]] = None,
        continuous: Optional[List[str]] = None,
        categorical: Optional[List[str]] = None,
        immutable: Optional[List[str]] = None,
        feasible_values: Optional[Dict[str, Any]] = None,
    ):
        self._data = data
        self._column_names = list(data.columns) if column_names is None else list(column_names)
        self._targets = targets
        self._target_name = target_name
        self._continuous = continuous
        self._categorical = categorical
        self._immutable = immutable
        self._feasible_values = feasible_values
        
        #NOTE: Convert the data to a pandas dataframe with polars too 
        
        
        self.validate_data()

    @property
    def data(self) -> pd.DataFrame:
        """Getter for the data"""
        return self._data

    @property
    def column_names(self) -> List[str]:
        return self._column_names

    @property
    def targets(self) -> pd.Series:
        """Getter for target labels"""
        return self._targets

    @property
    def target_name(self) -> str:
        """Getter for the target name"""
        return self._target_name

    @property
    def continuous(self) -> List[str]:
        """Getter for the continuous variables 

        Returns:
            List[str]: _description_
        """
        return self._continuous

    @property
    def categorical(self) -> List[str]:
        """Getter for categorical variables
        """
        return self._categorical

    @property
    def immutable(self) -> List[str]:
        """Getter for immutable features"""
        return self._immutable

    @property
    def feasible_values(self) -> Dict[str, Any]:
        """Getter for feasible values"""
        return self._feasible_values

    def _check_feature_names_exist(self, feature_list: List[str], name: str) -> None:
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
        # if not hasattr(self, 'data') or self.data is None:
        #     raise AttributeError("Attribute 'data' must be defined before validation.")

        missing = set(feature_list) - set(self.data.columns)
        if missing:
            raise ValueError(f"The following {name} features are not in the dataset: {missing}")

    @staticmethod
    def _check_data_label_alignment(data: pd.DataFrame, targets: pd.Series) -> None:
        """
        Ensure that the number of samples in data and targets match.

        Parameters
        ----------
        data : pd.DataFrame
            The feature matrix used to train the model.
        targets : pd.Series
            The target targets corresponding to the data.

        Raises
        ------
        ValueError
            If the number of rows in data does not match the number of targets.
        """
        if len(data) != len(targets):
            raise ValueError(
                f"Data and targets must have the same number of instances: "
                f"{len(data)} rows in data vs {len(targets)} targets."
            )

    def _validate_inputs(self):
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(f"'data' must be a pandas DataFrame, got {type(self.data).__name__}")
        if not isinstance(self.targets, pd.Series):
            raise TypeError(f"'targets' must be a pandas Series, got {type(self.targets).__name__}")
        if not isinstance(self.target_name, str):
            raise TypeError(f"'target_name' must be a string, got {type(self.target_name).__name__}")
        if self.continuous is not None and not isinstance(self.continuous, list):
            raise TypeError(f"'continuous' must be a list of strings or None, got {type(self.continuous).__name__}")
        if self.categorical is not None and not isinstance(self.categorical, list):
            raise TypeError(f"'categorical' must be a list of strings or None, got {type(self.categorical).__name__}")
        if self.immutable is not None and not isinstance(self.immutable, list):
            raise TypeError(f"'immutable' must be a list of strings or None, got {type(self.immutable).__name__}")
        if self.feasible_values is not None and not isinstance(self.feasible_values, dict):
            raise TypeError(f"'feasible_values' must be a dictionary or None, got {type(self.feasible_values).__name__}")

    def validate_data(self) -> None:
        """
        Validate the dataset and its properties.

        This method checks that:
        - All feature names in continuous, categorical, and immutable lists exist in the data.
        - The number of samples in data matches the number of targets.
        - Feasible values for features are valid.
        - No features are shared between continuous and categorical lists.
        """
        self._validate_inputs()
        self._check_data_label_alignment(data=self.data, targets=self.targets)
        self._check_feature_overlap(self.continuous, self.categorical)
        if self.continuous is not None:
            self._check_feature_names_exist(self.continuous, "continuous")
        if self.categorical is not None:
            self._check_feature_names_exist(self.categorical, "categorical")
        if self.immutable is not None:
            self._check_feature_names_exist(self.immutable, "immutable")
        if self.feasible_values is not None:
            self._check_range_dict_validity(self.feasible_values)



if __name__ == "__main__":
    PublicData("hello",["label1"],target_name="target")