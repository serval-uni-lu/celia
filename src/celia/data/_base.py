from abc import ABC, abstractmethod
from typing import List, Dict, Any
from celia._errors import ConfigurationError


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
    def continuous_column_names(self) -> List[str]:
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
    def categorical_column_names(self) -> List[str]:
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
    def immutable_column_names(self) -> List[str]:
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

    @abstractmethod
    def _validate_inputs(self, *args, **kwargs):
        """
        Abstract method to validate inputs. This should be implemented in subclasses depending on
        the extra arguments they receive.
        """
        raise NotImplementedError("Subclasses must implement _validate_inputs method.")

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
                raise ConfigurationError(
                    message=f"Feature '{feat}' in feasible_values is not in data.",
                    param=feat,
                    config={feat: val},
                    hint="Ensure all keys in feasible_values match feature names in the dataset."
                )

            if isinstance(val, tuple): # Continuous feature range
                if len(val) != 2 or not all(isinstance(v, (int, float)) for v in val):
                    raise ConfigurationError(
                        message=f"Invalid range tuple for feature '{feat}': {val}",
                        param=feat,
                        config={feat: val},
                        hint="Ranges must be tuples of two numeric values, e.g., (min, max)."
                    )
                if val[0] >= val[1]:
                    raise ConfigurationError(
                        message=f"Invalid range for feature '{feat}': min must be less than max, got {val}",
                        param=feat,
                        config={feat: val},
                        hint="Provide a tuple where the first element is strictly less than the second."
                    )

            elif isinstance(val, list): # Categorical feature values
                if not val:
                    raise ConfigurationError(
                        message=f"Categorical feature '{feat}' has an empty feasible_values list.",
                        param=feat,
                        config={feat: val},
                        hint="Provide at least one valid category."
                    )

                first_type = type(val[0])
                if not all(isinstance(v, first_type) for v in val):
                    raise ConfigurationError(
                        message=f"All feasible values for categorical feature '{feat}' must share the same type.",
                        param=feat,
                        config={feat: val},
                        hint=f"Ensure all values are of type {first_type.__name__}."
                    )

            else:
                raise ConfigurationError(
                    message=f"Unsupported feasible value type for '{feat}': {type(val).__name__}",
                    param=feat,
                    config={feat: val},
                    hint="Use tuple for continuous ranges or list for categorical values."
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
            raise ConfigurationError(
                message=(
                    "Some features are defined as both continuous and categorical, "
                    f"which is not allowed: {sorted(overlap)}"
                ),
                param="feature_overlap",
                config={
                    "continuous": continuous,
                    "categorical": categorical,
                    "overlap": list(overlap),
                },
                hint="Remove overlapping features so that each feature is either continuous or categorical, not both."
            )

    def validate_data(self) -> None:
        """
        Run a series of validation checks to ensure the integrity of the data interface.
        1. Ensures continuous and categorical features are disjoint.
        2. Ensures the target variable is not listed as a continuous feature.
        3. Validates the structure of the feasible_values dictionary.

        Raises
        ------
        CELIAConfigurationError
            If any of the internal consistency checks fail.
        """

        if self.continuous_column_names is not None and self.categorical_column_names is not None:
            self._check_feature_overlap(self.continuous_column_names, self.categorical_column_names)

        for feature_list, param_name in [
            (self.continuous_column_names, "continuous_column_names"),
            (self.categorical_column_names, "categorical_column_names"),
        ]:
            if self.target_name in (feature_list or []):
                raise ConfigurationError(
                    message=f"Target column '{self.target_name}' cannot be listed as a feature in {param_name}.",
                    param=param_name,
                    config={"target": self.target_name, param_name: feature_list},
                    hint="Remove the target column from the feature list.",
                )

        self._check_range_dict_validity(self.feasible_values)
