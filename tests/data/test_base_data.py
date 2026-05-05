import pytest
import pandas as pd
from celia.data import BaseData
from celia.errors import ConfigurationError


class DummyData(BaseData):
    """Concrete subclass of BaseData """
    def __init__(self, data: pd.DataFrame, target: str,
                 continuous: list[str], categorical: list[str],
                 immutable: list[str], feasible_values: dict):
        self._data = data
        self._target_name = target
        self._continuous = continuous
        self._categorical = categorical
        self._immutable = immutable
        self._feasible_values = feasible_values

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def continuous_column_names(self) -> list[str]:
        return self._continuous

    @property
    def categorical_column_names(self) -> list[str]:
        return self._categorical

    @property
    def immutable_column_names(self) -> list[str]:
        return self._immutable

    @property
    def feasible_values(self) -> dict:
        return self._feasible_values

    @property
    def monotonic_increasing_column_names(self) -> list[str]:
        return []

    @property
    def correlated_features(self) -> list[tuple[str, str, float]]:
        return []

    def _validate_inputs(self, *args, **kwargs):
        return True  # For testing only

@pytest.fixture
def simple_dataframe() -> pd.DataFrame:
    """A simple dataset with continuous and categorical columns."""
    return pd.DataFrame({
        "age": [25, 30, 35],
        "income": [40000, 50000, 60000],
        "gender": ["M", "F", "M"]
    })

@pytest.fixture
def valid_feasible_values(simple_dataframe) -> dict:
    """Valid feasible values dictionary aligned with simple_dataframe."""
    return {
        "age": (18, 65),
        "income": (20000, 100000),
        "gender": ["M", "F"]
    }

@pytest.fixture
def valid_dummy_data(simple_dataframe, valid_feasible_values) -> DummyData:
    """A valid DummyData object ready for testing BaseData logic."""
    return DummyData(
        data=simple_dataframe,
        target="income",
        continuous=["age"],
        categorical=["gender"],
        immutable=["age"],
        feasible_values=valid_feasible_values,
    )

def create_dummy_data(simple_dataframe, feasible_values) -> DummyData:
    """Helper to quickly create a DummyData object with custom feasible_values."""
    return DummyData(
        data=simple_dataframe,
        target="income",
        continuous=["age"],
        categorical=["gender"],
        immutable=[],
        feasible_values=feasible_values,
    )

class TestBaseData:

    def test_validate_data_passes_for_valid_data(self, valid_dummy_data):
        """Ensure validate_data passes without errors when configuration is valid."""
        try:
            valid_dummy_data._validate_data()
        except Exception as e:
            pytest.fail(f"validate_data raised an unexpected exception: {e}")

    def test_check_feature_overlap_raises_on_overlap(self, simple_dataframe, valid_feasible_values):
        """Ensure ConfigurationError is raised when a feature appears in both continuous and categorical lists."""

        overlapping_data = DummyData(
            data=simple_dataframe,
            target="income",
            continuous=["age"],
            categorical=["age", "gender"],
            immutable=[],
            feasible_values=valid_feasible_values,
        )

        with pytest.raises(ConfigurationError) as exc_info:
            overlapping_data._validate_data()

        err = exc_info.value
        assert "continuous" in str(err.config)
        assert "categorical" in str(err.config)
        assert "age" in str(err.config["overlap"])

    def test_check_feature_overlap_passes_when_disjoint(self, simple_dataframe, valid_feasible_values):
        """Ensure validate_data passes when continuous and categorical feature sets are disjoint."""

        disjoint_data = DummyData(
            data=simple_dataframe,
            target="income",
            continuous=["age"],
            categorical=["gender"],
            immutable=[],
            feasible_values=valid_feasible_values,
        )

        try:
            disjoint_data._validate_data()
        except Exception as e:
            pytest.fail(f"validate_data raised an unexpected exception for disjoint features: {e}")

    def test_invalid_range_tuple_length(self, simple_dataframe):
        """Raise ConfigurationError if a continuous feature has a tuple of incorrect length."""

        bad_feasible_values = {
            "age": (18,),  # wrong length
            "income": (20000, 100000),
            "gender": ["M", "F"],
        }

        bad_data = create_dummy_data(simple_dataframe, bad_feasible_values)

        with pytest.raises(ConfigurationError) as exc_info:
            bad_data._validate_data()

        err = exc_info.value
        assert err.param == "age"
        assert "tuple" in err.message

    def test_invalid_range_tuple_types(self, simple_dataframe):
        """Raise ConfigurationError if a continuous feature has a tuple with non-numeric values."""
        from celia.errors import ConfigurationError

        bad_feasible_values = {
            "age": (18, "sixty-five"),  # invalid type: string
            "income": (20000, 100000),
            "gender": ["M", "F"],
        }

        bad_data = create_dummy_data(simple_dataframe, bad_feasible_values)

        with pytest.raises(ConfigurationError) as exc_info:
            bad_data._validate_data()

        err = exc_info.value
        assert err.param == "age"
        assert "numeric" in err.hint

    def test_invalid_range_min_greater_than_max(self, simple_dataframe):
        """Raise ConfigurationError if a continuous feature has min >= max in its feasible range."""

        bad_feasible_values = {
            "age": (65, 18),  # invalid: min >= max
            "income": (20000, 100000),
            "gender": ["M", "F"],
        }

        bad_data = create_dummy_data(simple_dataframe, bad_feasible_values)

        with pytest.raises(ConfigurationError) as exc_info:
            bad_data._validate_data()

        err = exc_info.value
        assert err.param == "age"
        assert "min must be less than max" in err.message

    def test_categorical_empty_list(self, simple_dataframe):
        """Raise ConfigurationError if a categorical feature has an empty feasible_values list."""
        bad_feasible_values = {
            "age": (18, 65),
            "income": (20000, 100000),
            "gender": [],
        }

        bad_data = create_dummy_data(simple_dataframe, bad_feasible_values)

        with pytest.raises(ConfigurationError) as exc_info:
            bad_data._validate_data()

        err = exc_info.value
        assert err.param == "gender"
        assert "empty" in err.message

    def test_categorical_mixed_types(self, simple_dataframe):
        """Raise ConfigurationError if a categorical feature has feasible values of mixed types."""
        bad_feasible_values = {
            "age": (18, 65),
            "income": (20000, 100000),
            "gender": ["M", 1],
        }

        bad_data = create_dummy_data(simple_dataframe, bad_feasible_values)

        with pytest.raises(ConfigurationError) as exc_info:
            bad_data._validate_data()

        err = exc_info.value
        assert err.param == "gender"
        assert "must share the same type" in err.message

    def test_categorical_valid_list(self, simple_dataframe):
        """Ensure validate_data passes with a valid boolean categorical feasible_values list."""
        good_feasible_values = {
            "age": (18, 65),
            "income": (20000, 100000),
            "gender": [True, False],  # valid: homogeneous list of bool
        }

        good_data = create_dummy_data(simple_dataframe, good_feasible_values)

        try:
            good_data._validate_data()
        except Exception as e:
            pytest.fail(f"validate_data raised an unexpected exception for valid boolean categorical list: {e}")

    def test_unsupported_type_in_feasible_values(self, simple_dataframe):
        """Raise ConfigurationError if a feasible_values entry has an unsupported type."""
        bad_feasible_values = {
            "age": (18, 65),
            "income": (20000, 100000),
            "gender": "M/F",  # invalid: string instead of list or tuple
        }

        bad_data = create_dummy_data(simple_dataframe, bad_feasible_values)

        with pytest.raises(ConfigurationError) as exc_info:
            bad_data._validate_data()

        err = exc_info.value
        assert err.param == "gender"
        assert "Unsupported feasible value type" in err.message

    def test_feature_not_in_data_raises(self, simple_dataframe):
        """Raise ConfigurationError if feasible_values contains a feature not present in the dataset."""
        bad_feasible_values = {
            "age": (18, 65),
            "income": (20000, 100000),
            "gender": ["M", "F"],
            "nonexistent": (0, 1),
        }

        bad_data = create_dummy_data(simple_dataframe, bad_feasible_values)

        with pytest.raises(ConfigurationError) as exc_info:
            bad_data._validate_data()

        err = exc_info.value
        assert err.param == "nonexistent"
        assert "not in data" in err.message

    @pytest.mark.parametrize("param_name,feature_list", [
        ("continuous_column_names", ["age", "income"]),
        ("categorical_column_names", ["gender", "income"]),
    ])
    def test_target_not_allowed_in_feature_lists(self, simple_dataframe, param_name, feature_list):
        """Raise ConfigurationError if target column is incorrectly listed as a feature."""

        bad_feasible_values = {
            "age": (18, 65),
            "income": (20000, 100000),
            "gender": ["M", "F"],
        }

        bad_data = DummyData(
            data=simple_dataframe,
            target="income",  # target column
            continuous=feature_list if param_name == "continuous_column_names" else ["age"],
            categorical=feature_list if param_name == "categorical_column_names" else ["gender"],
            immutable=[],
            feasible_values=bad_feasible_values,
        )

        with pytest.raises(ConfigurationError) as exc_info:
            bad_data._validate_data()

        err = exc_info.value
        # Error should clearly indicate which parameter list is invalid
        assert err.param == param_name
        assert "Target column 'income'" in err.message

        # Config should contain the correct offending list
        assert param_name in err.config
        assert "income" in err.config[param_name]