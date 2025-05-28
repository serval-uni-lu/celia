from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional


class BaseData(ABC):
    """
    Abstract base class for data handling in Celia.
    All data classes should inherit from this class and implement the required properties.
    """
    @property
    @abstractmethod
    def target_name(self) -> str:
        """
        Return the name(s) of the target variable(s).

        Returns
        -------
        str
            Target variable name.
        """
        pass

    @property
    @abstractmethod
    def continuous(self) -> List[str]:
        """
        Return the names of continuous features.

        Returns
        -------
        List[str]
            List of continuous feature names.
        """
        pass

    @property
    @abstractmethod
    def categorical(self) -> List[str]:
        """
        Return the names of categorical features.

        Returns
        -------
        List[str]
            List of categorical feature names.
        """
        pass

    @property
    @abstractmethod
    def immutable(self) -> List[str]:
        """
        Return the names of immutable features (those that cannot change in counterfactuals).

        Returns
        -------
        List[str]
            List of immutable feature names.
        """
        pass

    @property
    @abstractmethod
    def feasible_values(self) -> Dict[str, Any]:
        """
        Return the feasible ranges or sets of values for each feature.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping feature names to feasible values:
            - Continuous features → tuple of (min, max)
            - Categorical features → list of allowed categories
        """
        pass

    def _check_range_dict_validity(self, ranges: Dict[str, Any]) -> None:
        """
        Validate structure of the feasible_values dictionary.

        Parameters
        ----------
        ranges : Dict[str, Any]
            Dictionary mapping feature names to feasible values.
            - For continuous features, values must be tuples: (min, max)
            - For categorical features, values must be lists of allowed values

        Raises
        ------
        ValueError
            If feature is not found in the data, or if values are not valid.
        TypeError
            If the type of feasible value is unsupported.
        """
        for feat, val in ranges.items():
            if feat not in self.data.columns:
                raise ValueError(f"Feature '{feat}' in feasible_values is not in data.")

            if isinstance(val, tuple):
                if len(val) != 2 or not all(isinstance(v, (int, float)) for v in val):
                    raise ValueError(f"Invalid range tuple for feature '{feat}': {val}")
                if val[0] >= val[1]:
                    raise ValueError(
                        f"Invalid range for feature '{feat}': min must be less than max, got {val}"
                    )

            elif isinstance(val, list):
                if not all(isinstance(v, (str, int)) for v in val):
                    raise ValueError(
                        f"Invalid list of values for categorical feature '{feat}': {val}"
                    )

            else:
                raise TypeError(
                    f"Unsupported feasible value type for '{feat}': {type(val)}"
                )

    def _check_feature_overlap(self, continuous: List[str], categorical: List[str]) -> None:
        """
        Ensure that no features are shared between continuous and categorical lists.

        Parameters
        ----------
        continuous : List[str]
            List of continuous feature names.

        categorical : List[str]
            List of categorical feature names.

        Raises
        ------
        ValueError
            If any feature is present in both lists.
        """
        overlap = set(continuous).intersection(categorical)
        if overlap:
            raise ValueError(
                f"The following features are defined as both continuous and categorical: {overlap}"
            )

    def validate_data(self) -> None:
        """
        Run a series of validation checks to ensure the integrity of the data interface.

        Raises
        ------
        ValueError or TypeError
            If any of the internal consistency checks fail.
        """
        # Ensure continuous and categorical are disjoint
        self._check_feature_overlap(self.continuous, self.categorical)

        # Ensure feasible_values is valid
        self._check_range_dict_validity(self.feasible_values)
