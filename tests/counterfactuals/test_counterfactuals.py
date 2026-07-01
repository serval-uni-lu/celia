import numpy as np
import pandas as pd
import pytest

from celia.counterfactuals import Counterfactual
from celia.errors import MethodError, MethodValueError


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_series(values: dict | None = None) -> pd.Series:
    if values is None:
        values = {"a": 1.0, "b": 2.0, "c": "x"}
    return pd.Series(values)


def _make_df(values: dict | None = None, n_rows: int = 1) -> pd.DataFrame:
    if values is None:
        values = {"a": [1.0] * n_rows, "b": [2.0] * n_rows, "c": ["x"] * n_rows}
    return pd.DataFrame(values)


def _make_cf_df(values: dict | None = None, n_rows: int = 1) -> pd.DataFrame:
    if values is None:
        values = {"a": [3.0] * n_rows, "b": [4.0] * n_rows, "c": ["y"] * n_rows}
    return pd.DataFrame(values)


# ── Happy path: __init__ ────────────────────────────────────────────────────


class TestCounterfactualInitHappyPaths:
    """Test that Counterfactual can be constructed with valid inputs."""

    @pytest.mark.parametrize(
        "original_instance",
        [
            pytest.param(_make_series(), id="series"),
            pytest.param(_make_df(), id="dataframe"),
        ],
    )
    def test_original_instance_series_or_dataframe(self, original_instance: pd.Series | pd.DataFrame):
        cf = Counterfactual(
            original_instance=original_instance,
            counterfactual_instance=_make_cf_df(),
            original_prediction=1,
            counterfactual_prediction=0,
        )
        assert isinstance(cf.original_instance, pd.DataFrame)
        assert cf.original_instance.shape[0] == 1

    @pytest.mark.parametrize(
        "counterfactual_instance",
        [
            pytest.param(_make_cf_df(), id="dataframe_1_row"),
            pytest.param(_make_cf_df(n_rows=3), id="dataframe_3_rows"),
            pytest.param(pd.Series({"a": 3.0, "b": 4.0, "c": "y"}), id="series"),
        ],
    )
    def test_counterfactual_instance_series_or_dataframe(self, counterfactual_instance: pd.Series | pd.DataFrame):
        n_rows = 1 if isinstance(counterfactual_instance, pd.Series) else len(counterfactual_instance)
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=counterfactual_instance,
            original_prediction=1,
            counterfactual_prediction=[0] * n_rows if n_rows > 1 else 0,
        )
        assert isinstance(cf.counterfactuals, pd.DataFrame)
        assert cf.counterfactuals.shape[0] == n_rows

    @pytest.mark.parametrize(
        "original_prediction",
        [
            pytest.param(1, id="int"),
            pytest.param(2.5, id="float"),
            pytest.param("class_a", id="str"),
        ],
    )
    def test_original_prediction_real_or_str(self, original_prediction: int | float | str):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(),
            original_prediction=original_prediction,
            counterfactual_prediction=0,
        )
        assert cf.original_prediction == original_prediction

    @pytest.mark.parametrize(
        "counterfactual_prediction",
        [
            pytest.param(1, id="int"),
            pytest.param(2.5, id="float"),
            pytest.param("class_b", id="str"),
        ],
    )
    def test_counterfactual_prediction_scalar(self, counterfactual_prediction: int | float | str):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(),
            original_prediction=0,
            counterfactual_prediction=counterfactual_prediction,
        )
        assert cf.counterfactual_prediction == counterfactual_prediction

    def test_counterfactual_prediction_list_of_ints(self):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(n_rows=3),
            original_prediction=0,
            counterfactual_prediction=[1, 2, 3],
        )
        assert cf.counterfactual_prediction == [1, 2, 3]

    def test_counterfactual_prediction_list_of_floats(self):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(n_rows=2),
            original_prediction=0,
            counterfactual_prediction=[1.1, 2.2],
        )
        assert cf.counterfactual_prediction == [1.1, 2.2]

    def test_counterfactual_prediction_list_of_strs(self):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(n_rows=2),
            original_prediction="a",
            counterfactual_prediction=["b", "c"],
        )
        assert cf.counterfactual_prediction == ["b", "c"]


# ── Happy path: numpy scalar predictions ────────────────────────────────────


class TestCounterfactualNumpyPredictions:
    """Numpy int/float types should be accepted as valid predictions."""

    @pytest.mark.parametrize(
        "original_prediction",
        [
            pytest.param(np.int32(1), id="np_int32"),
            pytest.param(np.int64(1), id="np_int64"),
            pytest.param(np.float32(1.5), id="np_float32"),
            pytest.param(np.float64(1.5), id="np_float64"),
        ],
    )
    def test_original_prediction_numpy_scalars(self, original_prediction):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(),
            original_prediction=original_prediction,
            counterfactual_prediction=0,
        )
        assert cf.original_prediction == original_prediction

    @pytest.mark.parametrize(
        "counterfactual_prediction",
        [
            pytest.param(np.int32(1), id="np_int32"),
            pytest.param(np.int64(1), id="np_int64"),
            pytest.param(np.float32(1.5), id="np_float32"),
            pytest.param(np.float64(1.5), id="np_float64"),
        ],
    )
    def test_counterfactual_prediction_numpy_scalars(self, counterfactual_prediction):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(),
            original_prediction=0,
            counterfactual_prediction=counterfactual_prediction,
        )
        assert cf.counterfactual_prediction == counterfactual_prediction

    def test_counterfactual_prediction_list_of_numpy_ints(self):
        preds = [np.int64(1), np.int64(2), np.int64(3)]
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(n_rows=3),
            original_prediction=0,
            counterfactual_prediction=preds,
        )
        assert cf.counterfactual_prediction == preds

    def test_counterfactual_prediction_list_of_numpy_floats(self):
        preds = [np.float64(1.1), np.float64(2.2)]
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(n_rows=2),
            original_prediction=0,
            counterfactual_prediction=preds,
        )
        assert cf.counterfactual_prediction == preds


# ── Happy path: _validate_counterfactuals ────────────────────────────────────


class TestValidateCounterfactualsHappyPaths:
    """_validate_counterfactuals should pass when columns match."""

    def test_series_vs_series_same_columns(self):
        original = pd.Series({"a": 1, "b": 2})
        counterfactual = pd.Series({"a": 3, "b": 4})
        Counterfactual._validate_counterfactuals(original, counterfactual)

    def test_dataframe_vs_dataframe_same_columns(self):
        original = pd.DataFrame({"a": [1], "b": [2]})
        counterfactual = pd.DataFrame({"a": [3], "b": [4]})
        Counterfactual._validate_counterfactuals(original, counterfactual)

    def test_series_vs_dataframe_same_columns(self):
        original = pd.Series({"a": 1, "b": 2})
        counterfactual = pd.DataFrame({"a": [3, 4], "b": [5, 6]})
        Counterfactual._validate_counterfactuals(original, counterfactual)

    def test_dataframe_vs_series_same_columns(self):
        original = pd.DataFrame({"a": [1], "b": [2]})
        counterfactual = pd.Series({"a": 3, "b": 4})
        Counterfactual._validate_counterfactuals(original, counterfactual)

    def test_multi_row_counterfactual_dataframe(self):
        original = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        counterfactual = pd.DataFrame({"x": [4, 5], "y": [6, 7], "z": [8, 9]})
        Counterfactual._validate_counterfactuals(original, counterfactual)


# ── Error cases: original_instance type ──────────────────────────────────────


class TestOriginalInstanceTypeErrors:
    """original_instance must be pd.Series or pd.DataFrame."""

    @pytest.mark.parametrize(
        "bad_input",
        [
            pytest.param([1, 2, 3], id="list"),
            pytest.param(np.array([1, 2, 3]), id="ndarray"),
            pytest.param({"a": 1}, id="dict"),
            pytest.param(42, id="int"),
            pytest.param("string", id="str"),
            pytest.param(None, id="none"),
        ],
    )
    def test_invalid_original_instance_type(self, bad_input):
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=bad_input,
                counterfactual_instance=_make_cf_df(),
                original_prediction=1,
                counterfactual_prediction=0,
            )
        err = exc_info.value
        assert err.param == "original_instance"
        assert "pandas Series or DataFrame" in err.message
        assert err.source == "Counterfactual._validate_init"


# ── Error cases: counterfactual_instance type ────────────────────────────────


class TestCounterfactualInstanceTypeErrors:
    """counterfactual_instance must be pd.Series or pd.DataFrame."""

    @pytest.mark.parametrize(
        "bad_input",
        [
            pytest.param([1, 2, 3], id="list"),
            pytest.param(np.array([1, 2, 3]), id="ndarray"),
            pytest.param({"a": 1}, id="dict"),
            pytest.param(42, id="int"),
            pytest.param("string", id="str"),
            pytest.param(None, id="none"),
        ],
    )
    def test_invalid_counterfactual_instance_type(self, bad_input):
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=_make_df(),
                counterfactual_instance=bad_input,
                original_prediction=1,
                counterfactual_prediction=0,
            )
        err = exc_info.value
        assert err.param == "counterfactual_instance"
        assert "pandas Series or DataFrame" in err.message
        assert err.source == "Counterfactual._validate_init"


# ── Error cases: empty counterfactual_instance ───────────────────────────────


class TestEmptyCounterfactualInstance:
    def test_empty_dataframe_raises(self):
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=_make_df(),
                counterfactual_instance=pd.DataFrame(),
                original_prediction=1,
                counterfactual_prediction=0,
            )
        err = exc_info.value
        assert err.param == "counterfactual_instance"
        assert "empty" in err.message.lower()

    def test_empty_series_raises(self):
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=_make_df(),
                counterfactual_instance=pd.Series(dtype=float),
                original_prediction=1,
                counterfactual_prediction=0,
            )
        err = exc_info.value
        assert err.param == "counterfactual_instance"
        assert "empty" in err.message.lower()


# ── Error cases: original_prediction type ────────────────────────────────────


class TestOriginalPredictionTypeErrors:
    """original_prediction must be Real | str (no lists, dicts, None, etc.)."""

    @pytest.mark.parametrize(
        "bad_prediction",
        [
            pytest.param([1, 2], id="list"),
            pytest.param({"a": 1}, id="dict"),
            pytest.param(None, id="none"),
            pytest.param((1, 2), id="tuple"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_invalid_original_prediction_type(self, bad_prediction):
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=_make_df(),
                counterfactual_instance=_make_cf_df(),
                original_prediction=bad_prediction,
                counterfactual_prediction=0,
            )
        err = exc_info.value
        assert err.param == "original_prediction"
        assert err.source == "Counterfactual._validate_init"


# ── Error cases: counterfactual_prediction type ──────────────────────────────


class TestCounterfactualPredictionTypeErrors:
    """counterfactual_prediction must be Real | str | list[Real] | list[str]."""

    @pytest.mark.parametrize(
        "bad_prediction",
        [
            pytest.param({"a": 1}, id="dict"),
            pytest.param(None, id="none"),
            pytest.param((1, 2), id="tuple"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_invalid_counterfactual_prediction_type(self, bad_prediction):
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=_make_df(),
                counterfactual_instance=_make_cf_df(),
                original_prediction=0,
                counterfactual_prediction=bad_prediction,
            )
        err = exc_info.value
        assert err.param == "counterfactual_prediction"
        assert err.source == "Counterfactual._validate_init"


# ── Error cases: empty counterfactual_prediction list ────────────────────────


class TestEmptyCounterfactualPredictionList:
    def test_empty_list_raises(self):
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=_make_df(),
                counterfactual_instance=_make_cf_df(),
                original_prediction=0,
                counterfactual_prediction=[],
            )
        err = exc_info.value
        assert err.param == "counterfactual_prediction"
        assert "empty" in err.message.lower()


# ── Error cases: prediction list length mismatch ─────────────────────────────


class TestCounterfactualPredictionListLengthMismatch:
    def test_length_mismatch_raises(self):
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=_make_df(),
                counterfactual_instance=_make_cf_df(n_rows=2),
                original_prediction=0,
                counterfactual_prediction=[1, 2, 3],
            )
        err = exc_info.value
        assert err.param == "counterfactual_prediction"
        assert "length" in err.message.lower() or "match" in err.message.lower()
        assert err.config["n_rows"] == 2
        assert err.config["len_predictions"] == 3


# ── Error cases: mixed types in prediction list ─────────────────────────────


class TestCounterfactualPredictionListMixedTypes:
    @pytest.mark.parametrize(
        "mixed_predictions",
        [
            pytest.param([1, "a"], id="int_and_str"),
            pytest.param([1.5, "b"], id="float_and_str"),
            pytest.param(["a", 1], id="str_and_int"),
        ],
    )
    def test_mixed_types_raises(self, mixed_predictions):
        n = len(mixed_predictions)
        with pytest.raises(MethodValueError) as exc_info:
            Counterfactual(
                original_instance=_make_df(),
                counterfactual_instance=_make_cf_df(n_rows=n),
                original_prediction=0,
                counterfactual_prediction=mixed_predictions,
            )
        err = exc_info.value
        assert err.param == "counterfactual_prediction"
        assert "same type" in err.message.lower()


# ── Error cases: _validate_counterfactuals column mismatch ───────────────────


class TestValidateCounterfactualsErrors:
    def test_different_columns_raises(self):
        original = pd.DataFrame({"a": [1], "b": [2]})
        counterfactual = pd.DataFrame({"a": [1], "z": [2]})
        with pytest.raises(MethodError) as exc_info:
            Counterfactual._validate_counterfactuals(original, counterfactual)
        err = exc_info.value
        assert err.param == "columns"
        assert "same columns" in err.message.lower()
        assert err.source == "Counterfactual._validate_counterfactuals"

    def test_different_columns_series_raises(self):
        original = pd.Series({"a": 1, "b": 2})
        counterfactual = pd.Series({"a": 1, "z": 2})
        with pytest.raises(MethodError) as exc_info:
            Counterfactual._validate_counterfactuals(original, counterfactual)
        err = exc_info.value
        assert err.param == "columns"

    def test_extra_column_in_counterfactual_raises(self):
        original = pd.DataFrame({"a": [1], "b": [2]})
        counterfactual = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        with pytest.raises(MethodError) as exc_info:
            Counterfactual._validate_counterfactuals(original, counterfactual)
        err = exc_info.value
        assert err.param == "columns"

    def test_missing_column_in_counterfactual_raises(self):
        original = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        counterfactual = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(MethodError) as exc_info:
            Counterfactual._validate_counterfactuals(original, counterfactual)
        err = exc_info.value
        assert err.param == "columns"

    def test_duplicate_columns_different_count_raises(self):
        """Same column set but different column count (duplicates) raises MethodError."""
        original = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        counterfactual = pd.DataFrame([[4, 5]], columns=["a", "b"])
        with pytest.raises(MethodError) as exc_info:
            Counterfactual._validate_counterfactuals(original, counterfactual)
        err = exc_info.value
        assert err.param == "columns"
        assert "same number of columns" in err.message


# ── Happy path: properties and highlighted_counterfactuals ───────────────────


class TestCounterfactualProperties:
    def test_properties_are_accessible(self):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(),
            original_prediction=1,
            counterfactual_prediction=0,
        )
        assert isinstance(cf.original_instance, pd.DataFrame)
        assert isinstance(cf.counterfactuals, pd.DataFrame)
        assert isinstance(cf.highlighted_counterfactuals, pd.DataFrame)
        assert cf.original_prediction == 1
        assert cf.counterfactual_prediction == 0

    def test_highlighted_counterfactuals_marks_changes(self):
        original = pd.DataFrame({"a": [1.0], "b": [2.0]})
        counterfactual = pd.DataFrame({"a": [1.0], "b": [9.0]})
        cf = Counterfactual(
            original_instance=original,
            counterfactual_instance=counterfactual,
            original_prediction=0,
            counterfactual_prediction=1,
        )
        highlighted = cf.highlighted_counterfactuals
        assert highlighted.iloc[0]["a"] == "-"
        assert highlighted.iloc[0]["b"] == 9.0
        assert "Prediction" in highlighted.columns

    def test_highlighted_counterfactuals_prediction_column_scalar(self):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(),
            original_prediction=0,
            counterfactual_prediction=1,
        )
        assert cf.highlighted_counterfactuals["Prediction"].iloc[0] == "0 → 1"

    def test_highlighted_counterfactuals_prediction_column_list(self):
        cf = Counterfactual(
            original_instance=_make_df(),
            counterfactual_instance=_make_cf_df(n_rows=2),
            original_prediction=0,
            counterfactual_prediction=[1, 2],
        )
        preds = cf.highlighted_counterfactuals["Prediction"].tolist()
        assert preds == ["0 → 1", "0 → 2"]
