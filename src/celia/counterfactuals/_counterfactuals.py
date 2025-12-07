import pandas as pd

from celia._errors import MethodError, MethodValueError


class Counterfactual:
    """
    A class representing counterfactual explanation(s) for a single instance.

    Parameters
    ----------
    original_instance : pd.Series | pd.DataFrame
        The original instance for which counterfactuals were generated.
    counterfactual_instance : pd.Series | pd.DataFrame
        The generated counterfactual instance(s).

    Attributes
    ----------
    original_instance : pd.DataFrame
        The original instance as a DataFrame.
    counterfactuals : pd.DataFrame
        The generated counterfactual instance(s) as a DataFrame.
    highlighted_counterfactuals : pd.DataFrame
        A DataFrame indicating which values changed relative to the original instance.
    """

    def __init__(
        self,
        original_instance: pd.Series | pd.DataFrame,
        counterfactual_instance: pd.Series | pd.DataFrame,
    ) -> None:
        (
            self._original_instance,
            self._counterfactuals,
        ) = self._validate_init(original_instance, counterfactual_instance)

        self._highlighted_counterfactuals = self._create_highlighted_counterfactuals()

    @property
    def original_instance(self) -> pd.DataFrame:
        return self._original_instance

    @property
    def counterfactuals(self) -> pd.DataFrame:
        return self._counterfactuals

    @property
    def highlighted_counterfactuals(self) -> pd.DataFrame:
        return self._highlighted_counterfactuals

    def _create_highlighted_counterfactuals(self) -> pd.DataFrame:
        """
        Create a DataFrame marking changes from the original instance.

        Returns
        -------
        pd.DataFrame
            DataFrame where changed values are shown and unchanged values are set to '-'.
        """
        highlighted = self.counterfactuals.copy()

        for col in highlighted.columns:
            original_value = self.original_instance.iloc[0][col]
            highlighted[col] = highlighted[col].where(highlighted[col] != original_value, "-")

        return highlighted

    def _validate_init(
        self,
        original_instance: pd.Series | pd.DataFrame,
        counterfactual_instance: pd.Series | pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate input types and ensure counterfactuals are non-empty.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Validated original and counterfactual instances as DataFrames.

        Raises
        ------
        MethodValueError
        """
        if not isinstance(original_instance, (pd.Series, pd.DataFrame)):
            message = "Original Instance must be a pandas Series or DataFrame."
            raise MethodValueError(
                message=message,
                config={"type": type(original_instance).__name__},
                param="original_instance",
                hint="Ensure the original instance is a pandas Series or DataFrame.",
                source="Counterfactual._validate_init",
            )

        if not isinstance(counterfactual_instance, (pd.Series, pd.DataFrame)):
            message = "Counterfactual Instance must be a pandas Series or DataFrame."
            raise MethodValueError(
                message=message,
                config={"type": type(counterfactual_instance).__name__},
                param="counterfactual_instance",
                hint="Ensure the counterfactual instance is a pandas Series or DataFrame.",
                source="Counterfactual._validate_init",
            )

        if counterfactual_instance.empty:
            message = "Counterfactual Instance is empty."
            raise MethodValueError(
                message=message,
                config={},
                param="counterfactual_instance",
                hint="Drop unsuccessful instances before creating a Counterfactual object.",
                source="Counterfactual._validate_init",
            )

        # Validate matching columns
        self._validate_counterfactuals(original_instance, counterfactual_instance)

        # Convert Series to a 1-row DataFrame
        if isinstance(original_instance, pd.Series):
            original_instance = original_instance.to_frame().T

        if isinstance(counterfactual_instance, pd.Series):
            counterfactual_instance = counterfactual_instance.to_frame().T

        return original_instance, counterfactual_instance

    @staticmethod
    def _validate_counterfactuals(
        original_instance: pd.Series | pd.DataFrame,
        counterfactual_instance: pd.Series | pd.DataFrame,
    ) -> None:
        """
        Ensure counterfactuals have the same columns as the original instance.

        Raises
        ------
        MethodError
        """
        if isinstance(original_instance, pd.Series):
            original_columns = list(original_instance.index)
        else:
            original_columns = list(original_instance.columns)

        if isinstance(counterfactual_instance, pd.Series):
            counter_columns = list(counterfactual_instance.index)
        else:
            counter_columns = list(counterfactual_instance.columns)

        if set(counter_columns) != set(original_columns):
            message = "Counterfactual instance must have the same columns as the original instance."
            raise MethodError(
                message=message,
                config={"expected": original_columns, "received": counter_columns},
                param="columns",
                hint="Ensure the counterfactual generator preserves feature names.",
                source="Counterfactual._validate_counterfactuals",
            )

        if len(counter_columns) != len(original_columns):
            message = "Counterfactual Instance must have the same number of columns as the original instance."
            raise MethodError(
                message=message,
                config={
                    "expected": len(original_columns),
                    "received": len(counter_columns),
                },
                param="columns",
                hint="Ensure the counterfactual generator preserves the number of features.",
                source="Counterfactual._validate_counterfactuals",
            )
