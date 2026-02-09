import pytest
import pandas as pd
import numpy as np
from celia.data.public_data import PublicData
from celia.errors import ConfigurationError

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

    def test_invalid_data_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `data` is not a DataFrame."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(dummy_dataframe,
                                              valid_feasible_values,
                                              data=[1, 2, 3])

        err = exc_info.value
        assert err.param == "data"
        assert "must be a pandas DataFrame" in err.message

    def test_invalid_targets_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `targets` is not a Series."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(dummy_dataframe,
                                              valid_feasible_values,
                                              targets= [0, 1, 0])

        err = exc_info.value
        assert err.param == "targets"
        assert "must be a pandas Series" in err.message

    def test_invalid_target_name_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `target_name` is not a string."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(dummy_dataframe,
                                              valid_feasible_values,
                                              target_name=123)

        err = exc_info.value
        assert err.param == "target_name"
        assert "must be a string" in err.message

    def test_invalid_continuous_column_names_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `continuous_column_names` is not a list or None."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(dummy_dataframe,
                                              valid_feasible_values,
                                              continuous_column_names="age")

        err = exc_info.value
        assert err.param == "continuous_column_names"
        assert "must be a list of strings" in err.message

    def test_invalid_categorical_column_names_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `categorical_column_names` is not a list or None."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(dummy_dataframe,
                                              valid_feasible_values,
                                              categorical_column_names="gender")

        err = exc_info.value
        assert err.param == "categorical_column_names"
        assert "must be a list of strings" in err.message

    def test_invalid_immutable_column_names_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `immutable_column_names` is not a list or None."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(dummy_dataframe,
                                              valid_feasible_values,
                                              immutable_column_names="age")

        err = exc_info.value
        assert err.param == "immutable_column_names"
        assert "must be a list of strings" in err.message

    def test_invalid_feasible_values_type(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if `feasible_values` is not a dict or None."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(dummy_dataframe,
                                              valid_feasible_values,
                                              feasible_values=["invalid", "list"])


        err = exc_info.value
        assert err.param == "feasible_values"
        assert "must be a dictionary" in err.message

    def test_data_label_mismatch_raises(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if data and targets have different lengths."""
        shorter_targets = pd.Series([0, 1, 0], name="target")  # only 3 instead of 5

        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(dummy_dataframe,
                                              valid_feasible_values,
                                              targets=shorter_targets)

        err = exc_info.value
        assert err.param == "targets"
        assert "must have the same number of instances" in err.message
        assert err.config["data_rows"] == 5
        assert err.config["target_rows"] == 3

    def test_data_label_alignment_passes(self, dummy_dataframe, valid_feasible_values):
        """Ensure validation passes when data and targets lengths match."""
        try:
            PublicData(
                data=dummy_dataframe,
                targets=pd.Series([0, 1, 0, 1, 0], name="target"),
                target_name="target",
                column_names=dummy_dataframe.columns,
                continuous_column_names=["age", "income"],
                categorical_column_names=["gender", "employed"],
                immutable_column_names=["age"],
                feasible_values=valid_feasible_values,
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception raised for matching data and targets: {e}")

    def test_missing_continuous_feature_raises(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if a continuous feature is not present in data."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(
                dummy_dataframe,
                valid_feasible_values,
                continuous_column_names=["age", "nonexistent_feature"]
            )

        err = exc_info.value
        assert err.param == "continuous"
        assert "not in the dataset" in err.message
        assert "nonexistent_feature" in err.config["missing_features"]

    def test_missing_categorical_feature_raises(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if a categorical feature is not present in data."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(
                dummy_dataframe,
                valid_feasible_values,
                categorical_column_names=["gender", "missing_col"]
            )

        err = exc_info.value
        assert err.param == "categorical"
        assert "not in the dataset" in err.message
        assert "missing_col" in err.config["missing_features"]

    def test_missing_immutable_feature_raises(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if an immutable feature is not present in data."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(
                dummy_dataframe,
                valid_feasible_values,
                immutable_column_names=["nonexistent_feature"]
            )

        err = exc_info.value
        assert err.param == "immutable"
        assert "not in the dataset" in err.message
        assert "nonexistent_feature" in err.config["missing_features"]

    def test_all_features_present_passes(self, dummy_dataframe, valid_feasible_values):
        """Ensure validate_data passes when all features exist in the dataset."""
        try:
            create_public_data_with_overrides(dummy_dataframe, valid_feasible_values)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised for valid feature names: {e}")

    def test_feature_overlap_raises(self, dummy_dataframe, valid_feasible_values):
        """Raise ConfigurationError if a feature appears in both continuous and categorical lists."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(
                dummy_dataframe,
                valid_feasible_values,
                continuous_column_names=["age", "income"],
                categorical_column_names=["gender", "income"]  # overlap: 'income'
            )

        err = exc_info.value
        assert err.param == "feature_overlap"
        assert "both continuous and categorical" in err.message
        assert "income" in err.config["overlap"]

    def test_no_feature_overlap_passes(self, dummy_dataframe, valid_feasible_values):
        """Ensure validate_data passes when no overlap exists between continuous and categorical lists."""
        try:
            create_public_data_with_overrides(
                dummy_dataframe,
                valid_feasible_values,
                continuous_column_names=["age", "income"],
                categorical_column_names=["gender", "employed"]
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception raised for non-overlapping features: {e}")

    def test_target_name_series_alignment(self, dummy_dataframe, valid_feasible_values):
        series = pd.Series([0, 1, 0, 1, 0], name="frailty")
        data = create_public_data_with_overrides(
            dummy_dataframe, valid_feasible_values, targets=series, target_name="health_index"
        )
        assert data.targets.name == "health_index"
        assert data.target_name == "health_index"

    def test_target_name_ndarray_converted(self, dummy_dataframe, valid_feasible_values):
        arr = np.array([0, 1, 0, 1, 0])
        data = create_public_data_with_overrides(
            dummy_dataframe, valid_feasible_values, targets=arr, target_name="frailty"
        )
        assert isinstance(data.targets, pd.Series)
        assert data.targets.name == "frailty"

    def test_target_name_ndarray_missing_name_raises(self, dummy_dataframe, valid_feasible_values):
        arr = np.array([0, 1, 0, 1, 0])
        with pytest.raises(ConfigurationError):
            create_public_data_with_overrides(
                dummy_dataframe, valid_feasible_values, targets=arr, target_name=None
            )

    @pytest.mark.parametrize(
        "bad_feasible_values, expected_param, expected_message_part",
        [
            (
                    {"age": (18,), "income": (20000, 100000), "gender": ["M", "F"]},
                    "age",
                    "Invalid range tuple",
            ),
            (
                    {"age": (65, 18), "income": (20000, 100000), "gender": ["M", "F"]},
                    "age",
                    "min must be less than max",
            ),
            (
                    {"age": (18, 65), "income": (20000, 100000), "gender": []},
                    "gender",
                    "empty feasible_values list",
            ),
            (
                    {"age": (18, 65), "income": (20000, 100000), "gender": ["M", 1]},
                    "gender",
                    "must share the same type",
            ),
            (
                    {"age": (18, 65), "income": {20000, 100000}, "gender": ["M", "F"]},
                    "income",
                    "Unsupported feasible value type",
            ),
        ],
    )
    def test_invalid_feasible_values_cases(
            self, dummy_dataframe, bad_feasible_values, expected_param, expected_message_part
    ):
        """Parametrized: check that invalid feasible_values raise the correct ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_public_data_with_overrides(
                dummy_dataframe,
                bad_feasible_values,
            )

        err = exc_info.value
        assert err.param == expected_param
        assert expected_message_part in err.message

    @pytest.mark.parametrize(
        "missing_param, should_raise_typeerror",
        [
            ("data", True),
            ("targets", True),
            ("target_name", False),
            ("column_names", False),
            ("continuous_column_names", False),
            ("categorical_column_names", False),
            ("immutable_column_names", False),
            ("feasible_values", False),
        ],
    )
    def test_missing_parameters_behavior(
            self, dummy_dataframe, valid_feasible_values, missing_param, should_raise_typeerror
    ):
        """Parametrized: check how PublicData behaves when one argument is omitted."""
        kwargs = {
            "data": dummy_dataframe,
            "targets": pd.Series([0, 1, 0, 1, 0], name="target"),
            "target_name": "target",
            "column_names": dummy_dataframe.columns,
            "continuous_column_names": ["age", "income"],
            "categorical_column_names": ["gender", "employed"],
            "immutable_column_names": ["age"],
            "feasible_values": valid_feasible_values,
        }

        kwargs.pop(missing_param)

        if should_raise_typeerror:
            with pytest.raises(TypeError):
                PublicData(**kwargs)
        else:
            try:
                PublicData(**kwargs)
            except Exception as e:
                pytest.fail(f"Unexpected exception for missing optional param '{missing_param}': {e}")