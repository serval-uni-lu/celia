import pandas as pd
from typing import Union
from celia._errors import MethodError, MethodValueError

__all__ = ["Counterfactual"]

class Counterfactual:
    """A class representing (a) counterfactual explanation(s) for a single instance.

    Parameters
    ----------
    original_instance: Union[pd.Series, pd.DataFrame]
        The original instance for which counterfactuals were generated. Must be a pandas Series or DataFrame.
    counterfactual_instance: Union[pd.Series, pd.DataFrame]
        The generated counterfactual instance(s). Must be a pandas Series or DataFrame.

    Attributes
    ----------
    original_instance: pd.DataFrame
        The original instance as a DataFrame.
    counterfactuals: pd.DataFrame
        The generated counterfactual instance(s) as a DataFrame.
    highlighted_counterfactuals: pd.DataFrame
        A DataFrame showing only which columns have a new value in the counterfactual(s).
    """
    def __init__(self, original_instance: Union[pd.Series, pd.DataFrame],
                 counterfactual_instance: Union[pd.Series, pd.DataFrame],):

        self._original_instance, self._counterfactuals =\
            self._validate_init(original_instance, counterfactual_instance)
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
        """Creates a DataFrame of the counterfactual(s) showing only which columns have a new value.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the same shape as the counterfactuals, where changed values
            are shown and unchanged values are assigned '-' .
        """

        highlighted: pd.DataFrame = self.counterfactuals.copy()

        # Check if the value in each cell is the same as in the original instance
        for col in highlighted.columns:
            highlighted[col] = highlighted[col].where(
                highlighted[col] != self.original_instance.iloc[0][col], '-'
            )

        return highlighted

    def _validate_init(self,original_instance: Union[pd.Series, pd.DataFrame],
                        counterfactual_instance: Union[pd.Series, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validates the dtypes of the original and counterfactual instances, and checks if
        the counterfactual is empty.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The original and counterfactual instances as DataFrames.

        Raises
        ------
        MethodValueError
        """
        if not isinstance(original_instance, (pd.Series, pd.DataFrame)):
            raise MethodValueError(
                message="Original Instance must be a pandas Series or DataFrame.",
                config={"type": str(type(original_instance))},
                param="original_instance",
                hint="Ensure the original instance is a pandas Series or DataFrame.",
                source="Counterfactual.__validate_init",
            )
        if not isinstance(counterfactual_instance, (pd.Series, pd.DataFrame)):
            raise MethodValueError(
                message="Counterfactual Instance must be a pandas Series or DataFrame.",
                config={"type": str(type(counterfactual_instance))},
                param="counterfactual_instance",
                hint="Ensure the counterfactual instance is a pandas Series or DataFrame.",
                source="Counterfactual.__validate_init",
            )
        if counterfactual_instance.empty:
            raise MethodValueError(
                message="Counterfactual Instance is empty.",
                config={},
                param="counterfactual_instance",
                hint="Be sure to first drop any unsuccessful instances before creating a Counterfactual object.",
                source="Counterfactual.__validate_init",
            )

        self._validate_counterfactuals(counterfactual_instance, original_instance)

        if isinstance(original_instance, pd.Series):
            original_instance : pd.DataFrame = original_instance.to_frame().T

        if isinstance(counterfactual_instance, pd.Series):
            counterfactual_instance : pd.DataFrame = counterfactual_instance.to_frame().T

        return original_instance, counterfactual_instance

    @staticmethod
    def _validate_counterfactuals(original_instance: Union[pd.Series, pd.DataFrame],
                                   counterfactual_instance: Union[pd.Series, pd.DataFrame]):
        """Validates that the single or multiple counterfactuals have the necessary columns to create
        a Counterfactual object.

        Raises
        ------
        MethodError
        """
        if isinstance(original_instance, pd.Series):
            original_columns = original_instance.index.tolist()
        else:
            original_columns = original_instance.columns.tolist()

        if isinstance(counterfactual_instance, pd.Series):
            counterfactual_columns = counterfactual_instance.index.tolist()
        else:
            counterfactual_columns = counterfactual_instance.columns.tolist()


        if set(counterfactual_columns) != set(original_columns):
            raise MethodError(
                message=(
                    "Counterfactual instance must have the same columns as the original instance. "
                ),
                config={"expected": original_columns, "received": counterfactual_columns},
                param="columns",
                hint="Ensure the counterfactual generator preserves feature order and names.",
                source="Counterfactual.__validate_counterfactuals",
            )

        if len(counterfactual_columns) != len(original_columns):
            raise MethodError(
                message=(
                    "Counterfactual Instance must have the same number of columns as the original instance. "
                ),
                config={"expected": len(original_columns), "received": len(counterfactual_columns)},
                param="columns",
                hint="Ensure the counterfactual generator preserves the number of features",
                source="Counterfactual.__validate_counterfactuals",
            )





