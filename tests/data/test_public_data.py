import pytest
import pandas as pd
from celia.data.public_data import PublicData
from celia._errors import ConfigurationError

@pytest.fixture
def dummy_dataframe() -> pd.DataFrame:
    """A simple dataset with mixed feature types."""
    return pd.DataFrame({
        "age": [25, 32, 40, 29, 50],
        "income": [50000, 60000, 75000, 48000, 82000],
        "gender": ["M", "F", "F", "M", "F"],
        "employed": [True, False, True, True, False]
    })

@pytest.fixture
def valid_feasible_values() -> dict:
    """Valid feasible values for the dummy dataframe."""
    return {
        "age": (18, 65),
        "income": (20000, 100000),
        "gender": ["M", "F"],
        "employed": [True, False]
    }

@pytest.fixture
def valid_public_data(dummy_dataframe, valid_feasible_values) -> PublicData:
    """A valid PublicData object ready for testing."""
    return PublicData(
        data=dummy_dataframe,
        targets=pd.Series([0, 1, 0, 1, 0], name="target"),
        target_name="target",
        column_names=dummy_dataframe.columns,
        continuous_column_names=["age", "income"],
        categorical_column_names=["gender", "employed"],
        immutable_column_names=["age"],
        feasible_values=valid_feasible_values
    )


def create_public_data_with_overrides(
    dummy_dataframe,
    valid_feasible_values,
    **overrides
) -> PublicData:
    """
    Create a PublicData object with optional overrides.
    Useful for constructing invalid configurations.

    Example:
        create_public_data_with_overrides(
            dummy_dataframe,
            valid_feasible_values,
            continuous_column_names=["age", "target"]
        )
    """
    return PublicData(
        data=overrides.get("data", dummy_dataframe),
        targets=overrides.get("targets", pd.Series([0, 1, 0, 1, 0], name="target")),
        target_name=overrides.get("target_name", "target"),
        column_names=overrides.get("column_names", dummy_dataframe.columns),
        continuous_column_names=overrides.get("continuous_column_names", ["age", "income"]),
        categorical_column_names=overrides.get("categorical_column_names", ["gender", "employed"]),
        immutable_column_names=overrides.get("immutable_column_names", ["age"]),
        feasible_values=overrides.get("feasible_values", valid_feasible_values),
    )

class TestPublicData:

    def test_invalid_data_type(self, valid_feasible_values):
        """Raise ConfigurationError if `data` is not a DataFrame."""
        with pytest.raises(ConfigurationError) as exc_info:
            PublicData(
                data=[1, 2, 3],  # invalid
                targets=pd.Series([0, 1, 0], name="target"),
                target_name="target",
                column_names=["age"],
                continuous_column_names=["age"],
                categorical_column_names=[],
                immutable_column_names=[],
                feasible_values=valid_feasible_values,
            )

        err = exc_info.value
        assert err.param == "data"
        assert "must be a pandas DataFrame" in err.message

    def test_invalid_targets_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `targets` is not a Series."""
        with pytest.raises(ConfigurationError) as exc_info:
            PublicData(
                data=dummy_dataframe,
                targets= [0, 1, 0],  # invalid
                target_name="target",
                column_names=dummy_dataframe.columns,
                continuous_column_names=["age"],
                categorical_column_names=["gender"],
                immutable_column_names=["age"],
                feasible_values=valid_feasible_values,
            )

        err = exc_info.value
        assert err.param == "targets"
        assert "must be a pandas Series" in err.message

    def test_invalid_target_name_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `target_name` is not a string."""
        with pytest.raises(ConfigurationError) as exc_info:
            PublicData(
                data=dummy_dataframe,
                targets=pd.Series([0, 1, 0], name="target"),
                target_name=123,  # invalid
                column_names=dummy_dataframe.columns,
                continuous_column_names=["age"],
                categorical_column_names=["gender"],
                immutable_column_names=["age"],
                feasible_values=valid_feasible_values,
            )

        err = exc_info.value
        assert err.param == "target_name"
        assert "must be a string" in err.message