from abc import ABC, abstractmethod
from typing import Any, NoReturn

from celia.errors import ConfigurationError


class BaseData(ABC):
    """
    Abstract base class for data handling in Celia.
    All data classes should inherit from this class and implement the required properties.
    """

    @property
    @abstractmethod
    def target_name(self) -> str:
        """
        Return the name of the target variable.

        Returns
        -------
        str
            Target variable name.
        """

    @property
    @abstractmethod
    def continuous_column_names(self) -> list[str]:
        """
        Return the names of continuous features.

        Returns
        -------
        list[str]
            List of continuous feature names.
        """

    @property
    @abstractmethod
    def categorical_column_names(self) -> list[str]:
        """
        Return the names of categorical features.

        Returns
        -------
        list[str]
            List of categorical feature names.
        """

    @property
    @abstractmethod
    def immutable_column_names(self) -> list[str]:
        """
        Return the names of immutable features.

        Returns
        -------
        list[str]
            List of immutable feature names.
        """

    @property
    @abstractmethod
    def feasible_values(self) -> dict[str, Any]:
        """
        Return feasible ranges or allowed values for each feature.

        Returns
        -------
        dict[str, Any]
            - Continuous features → tuple(min, max)
            - Categorical features → list of valid categories
        """

    @property
    @abstractmethod
    def monotonic_increasing_column_names(self) -> list[str]:
        """
        Return the names of features that may only increase, never decrease.

        Returns
        -------
        list[str]
            List of monotonic increasing feature names.
        """

    @property
    @abstractmethod
    def correlated_features(self) -> list[tuple[str, str, float]]:
        """
        Return correlated feature triples of (cause, effect, delta).

        Returns
        -------
        list[tuple[str, str, float]]
            Each tuple describes a mechanical coupling: whenever ``cause``
            is increased, ``effect`` is also increased by ``delta``.
        """

    @abstractmethod
    def _validate_inputs(self, *args: object, **kwargs: object) -> NoReturn:
        """
        Abstract method to validate arbitrary inputs.
        """
        message = "Subclasses must implement _validate_inputs method."
        raise NotImplementedError(message)

    def _check_range_dict_validity(self, ranges: dict[str, Any]) -> None:
        """
        Validate the structure of the feasible_values dictionary.

        Parameters
        ----------
        ranges : dict[str, Any]
            Mapping of feature name to feasible range or allowed categories.

        Raises
        ------
        ConfigurationError
        """
        for feat, val in ranges.items():
            if feat not in self.data.columns:
                message = f"Feature '{feat}' in feasible_values is not in data."
                raise ConfigurationError(
                    message=message,
                    param=feat,
                    config={feat: val},
                    hint="Ensure all keys in feasible_values match feature names in the dataset.",
                )

            # Continuous ranges
            if isinstance(val, tuple):
                if len(val) != 2 or not all(isinstance(v, (int, float)) for v in val):
                    message = f"Invalid range tuple for feature '{feat}': {val}"
                    raise ConfigurationError(
                        message=message,
                        param=feat,
                        config={feat: val},
                        hint="Ranges must be tuples of two numeric values, e.g., (min, max).",
                    )
                if val[0] >= val[1]:
                    message = f"Invalid range for feature '{feat}': min must be less than max, got {val}"
                    raise ConfigurationError(
                        message=message,
                        param=feat,
                        config={feat: val},
                        hint="Provide a tuple where the first element is strictly less than the second.",
                    )

            # Categorical values
            elif isinstance(val, list):
                if not val:
                    message = f"Categorical feature '{feat}' has an empty feasible_values list."
                    raise ConfigurationError(
                        message=message,
                        param=feat,
                        config={feat: val},
                        hint="Provide at least one valid category.",
                    )

                first_type = type(val[0])
                if not all(isinstance(v, first_type) for v in val):
                    message = f"All feasible values for categorical feature '{feat}' must share the same type."
                    raise ConfigurationError(
                        message=message,
                        param=feat,
                        config={feat: val},
                        hint=f"Ensure all values are of type {first_type.__name__}.",
                    )

            # Unsupported type
            else:
                message = f"Unsupported feasible value type for '{feat}': {type(val).__name__}"
                raise ConfigurationError(
                    message=message,
                    param=feat,
                    config={feat: val},
                    hint="Use tuple for continuous ranges or list for categorical values.",
                )

    @staticmethod
    def _check_feature_overlap(continuous: list[str], categorical: list[str]) -> None:
        """
        Ensure features are not simultaneously continuous and categorical.

        Raises
        ------
        ConfigurationError
        """
        overlap = set(continuous).intersection(categorical)
        if overlap:
            message = (
                f"Some features are defined as both continuous and categorical, which is not allowed: {sorted(overlap)}"
            )
            raise ConfigurationError(
                message=message,
                param="feature_overlap",
                config={
                    "continuous": continuous,
                    "categorical": categorical,
                    "overlap": list(overlap),
                },
                hint=("Remove overlapping features so each feature is either continuous or categorical, but not both."),
            )

    def _validate_data(self) -> None:
        """
        Run validation checks for data integrity.

        Raises
        ------
        ConfigurationError
        """
        if self.continuous_column_names is not None and self.categorical_column_names is not None:
            self._check_feature_overlap(self.continuous_column_names, self.categorical_column_names)

        for feature_list, param_name in [
            (self.continuous_column_names, "continuous_column_names"),
            (self.categorical_column_names, "categorical_column_names"),
        ]:
            if self.target_name in (feature_list or []):
                message = f"Target column '{self.target_name}' cannot be listed as a feature in {param_name}."
                raise ConfigurationError(
                    message=message,
                    param=param_name,
                    config={"target": self.target_name, param_name: feature_list},
                    hint="Remove the target column from the feature list.",
                )

        self._check_range_dict_validity(self.feasible_values)
