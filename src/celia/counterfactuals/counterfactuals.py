from numbers import Real
from typing import TypeAlias

import pandas as pd

from celia.errors import MethodError, MethodValueError

Prediction: TypeAlias = Real | str
Predictions: TypeAlias = Prediction | list[Real] | list[str]


class Counterfactual:
    """
    A class representing counterfactual explanation(s) for a single instance.

    Parameters
    ----------
    original_instance : pd.Series | pd.DataFrame
        The original instance for which counterfactuals were generated.
    counterfactual_instance : pd.Series | pd.DataFrame
        The generated counterfactual instance(s).
    """

    def __init__(
        self,
        original_instance: pd.Series | pd.DataFrame,
        counterfactual_instance: pd.Series | pd.DataFrame,
        original_prediction: Prediction,
        counterfactual_prediction: Predictions,
    ) -> None:
        (
            self._original_instance,
            self._counterfactuals,
            self._original_prediction,
            self._counterfactual_prediction,
        ) = self._validate_init(
            original_instance, counterfactual_instance, original_prediction, counterfactual_prediction
        )

        self._highlighted_counterfactuals = self._create_highlighted_counterfactuals()

    @property
    def original_instance(self) -> pd.DataFrame:
        """The original instance as a pandas DataFrame."""
        return self._original_instance

    @property
    def counterfactuals(self) -> pd.DataFrame:
        """The generated counterfactual instance(s) as a pandas DataFrame."""
        return self._counterfactuals

    @property
    def highlighted_counterfactuals(self) -> pd.DataFrame:
        """A pandas DataFrame indicating which values changed relative to the original instance."""
        return self._highlighted_counterfactuals

    @property
    def original_prediction(self) -> Prediction:
        """The model's prediction for the original instance."""
        return self._original_prediction

    @property
    def counterfactual_prediction(self) -> Predictions:
        """The model's prediction for the counterfactual instance(s)."""
        return self._counterfactual_prediction

    def _create_highlighted_counterfactuals(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame marking changes from the original instance.

        Returns
        -------
        pd.DataFrame
            pandas DataFrame where changed values are shown and unchanged values are set to '-'.
        """
        highlighted = self.counterfactuals.copy()

        for col in highlighted.columns:
            original_value = self.original_instance.iloc[0][col]
            highlighted[col] = highlighted[col].where(highlighted[col] != original_value, "-")

        prediction_cells: list[str]
        if isinstance(self.counterfactual_prediction, list):
            prediction_cells = [f"{self.original_prediction} → {pred}" for pred in self.counterfactual_prediction]
        else:
            prediction_cells = [f"{self.original_prediction} → {self.counterfactual_prediction}"] * len(highlighted)

        highlighted["Prediction"] = prediction_cells
        return highlighted

    def _validate_init(
        self,
        original_instance: pd.Series | pd.DataFrame,
        counterfactual_instance: pd.Series | pd.DataFrame,
        original_prediction: Prediction,
        counterfactual_prediction: Predictions,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Prediction, Predictions]:
        """
        Validate input types, predictions, and ensure counterfactuals are non-empty.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, int | float | str, int | float | str | list[int] | list[float] | list[str]]
            A tuple containing:
                - original_instance: validated original instance as a DataFrame
                - counterfactual_instance: validated counterfactual instance(s) as a DataFrame
                - original_prediction: validated prediction for the original instance
                - counterfactual_prediction: validated prediction for the counterfactual instance(s)


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

        if not _is_prediction(original_prediction):
            message = "Original Prediction must be an int, float, or str."
            raise MethodValueError(
                message=message,
                config={"type": type(original_prediction).__name__},
                param="original_prediction",
                hint="Ensure the original prediction is a valid type (int, float, or str).",
                source="Counterfactual._validate_init",
            )

        if not _is_predictions(counterfactual_prediction):
            message = "Counterfactual Prediction must be an int, float, str, or a list of those types."
            raise MethodValueError(
                message=message,
                config={"type": type(counterfactual_prediction).__name__},
                param="counterfactual_prediction",
                hint="Ensure the counterfactual prediction is a valid type (int, float, str, list).",
                source="Counterfactual._validate_init",
            )

        if isinstance(counterfactual_prediction, list):
            if not counterfactual_prediction:
                message = "Counterfactual Prediction list cannot be empty."
                raise MethodValueError(
                    message=message,
                    config={},
                    param="counterfactual_prediction",
                    hint="Ensure the counterfactual prediction list contains at least one element.",
                    source="Counterfactual._validate_init",
                )

            n_counterfactuals = len(counterfactual_instance)
            if len(counterfactual_prediction) != n_counterfactuals:
                message = "Counterfactual Prediction list length must match number of counterfactual rows."
                raise MethodValueError(
                    message=message,
                    config={"n_rows": n_counterfactuals, "len_predictions": len(counterfactual_prediction)},
                    param="counterfactual_prediction",
                    hint="Provide one prediction per counterfactual row.",
                    source="Counterfactual._validate_init",
                )

            first_type = type(counterfactual_prediction[0])
            if not all(isinstance(pred, first_type) for pred in counterfactual_prediction):
                message = "All elements in Counterfactual Prediction list must be of the same type."
                raise MethodValueError(
                    message=message,
                    config={"types": [type(pred).__name__ for pred in counterfactual_prediction]},
                    param="counterfactual_prediction",
                    hint="Ensure all predictions in the list are of the same type (int, float, or str).",
                    source="Counterfactual._validate_init",
                )

        # Validate matching columns
        self._validate_counterfactuals(original_instance, counterfactual_instance)

        # Convert Series to a 1-row DataFrame
        if isinstance(original_instance, pd.Series):
            original_instance = original_instance.to_frame().T

        if isinstance(counterfactual_instance, pd.Series):
            counterfactual_instance = counterfactual_instance.to_frame().T

        return original_instance, counterfactual_instance, original_prediction, counterfactual_prediction

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


def _is_prediction(value: object) -> bool:
    """Check if *value* is a valid single prediction (Real or str, but not bool)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (Real, str))


def _is_predictions(value: object) -> bool:
    """Check if *value* is a valid Predictions (scalar or list of homogeneous predictions)."""
    if _is_prediction(value):
        return True
    if isinstance(value, list):
        return all(_is_prediction(item) for item in value)
    return False
