import pandas as pd
from celia.errors.data_handling_errors import UnSupportedDataTypeError
from typing import List, Dict, Any, Tuple, Optional
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

    labels : pd.Series
        The target labels corresponding to the training data, with shape (n_samples,).

    target_names : List[str]
        The name(s) of the target variable(s). Typically, a single string for most use cases.

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

    labels : pd.Series
        Returns the stored target labels.

    target_name : List[str]
        Returns the list of target variable names.

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
        labels: pd.Series | List[str] | Tuple[str], # NOTE: Not a list?
        target_name: str,
        continuous: Optional[List[str]] = None,
        categorical: Optional[List[str]] = None,
        immutable: Optional[List[str]] = None,
        feasible_values: Optional[Dict[str, Any]] = None,
    ):
        try:
            self._data = pd.DataFrame(data)
        except Exception as e:
            raise UnSupportedDataTypeError


        self._labels = labels
        self._target_name = target_name
        self._continuous = continuous # 
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
    def labels(self) -> pd.Series:
        """Getter for the label(s)"""
        return self._labels

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
        return self._immutable

    @property
    def feasible_values(self) -> Dict[str, Any]:
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

    def _check_data_label_alignment(self, labels: pd.Series) -> None:
        """
        Ensure that the number of samples in data and labels match.

        Parameters
        ----------
        data : pd.DataFrame
            The feature matrix used to train the model.
        labels : pd.Series
            The target labels corresponding to the data.

        Raises
        ------
        ValueError
            If the number of rows in data does not match the number of labels.
        """
        if len(self._data) != len(labels):
            raise ValueError(
                f"Data and labels must have the same number of instances: "
                f"{len(self._data)} rows in self._data vs {len(labels)} labels."
            )

    def validate_data(self) -> None:
        """
        Validate the dataset and its properties.

        This method checks that:
        - All feature names in continuous, categorical, and immutable lists exist in the data.
        - The number of samples in data matches the number of labels.
        - Feasible values for features are valid.
        - No features are shared between continuous and categorical lists.
        """
        self._check_feature_names_exist(self.continuous, "continuous")
        self._check_feature_names_exist(self.categorical, "categorical")
        self._check_feature_names_exist(self.immutable, "immutable")
        self._check_data_label_alignment(self.data, self.labels)
        self._check_range_dict_validity(self.feasible_values)
        self._check_feature_overlap(self.continuous, self.categorical)



if __name__ == "__main__":
    PublicData("hello",["label1"],target_name="target")