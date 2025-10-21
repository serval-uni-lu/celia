import pytest
from typing import Any, Union
import pandas as pd
from celia.data import PublicData, BaseData
from celia.model import SklearnModel, BaseModel
from celia.explainers import BaseExplainer
from sklearn.dummy import DummyRegressor
from celia._errors import ConfigurationError
import numpy as np

class ModelWithoutPredict:
    """A dummy model that does not implement the required interface."""
    def __init__(self):
        pass

    def incompatible_function(self, X):
        return [42] * len(X)

class IncompatibleModel:
    """A dummy model that implements predict, but it is not wrapped in a BaseModel subclass."""
    def __init__(self):
        pass

    def predict(self, X):
        return [42] * len(X)

class BasicExplainer(BaseExplainer):
    def __init__(self, model: BaseModel, data: BaseData):
        super().__init__(model, data)

    def _create_explainer(self, model: BaseModel, data:PublicData, *args, **kwargs) -> Any:
        pass

    def _validate_sample(self, sample: Union[pd.DataFrame, pd.Series], *args, **kwargs) -> None:
        pass

    def generate_counterfactuals(self, *args, **kwargs) -> Any:
        super().validate_sample(*args, **kwargs)

@pytest.fixture
def model_without_predict() -> ModelWithoutPredict:
    return ModelWithoutPredict()

@pytest.fixture
def incompatible_model() -> IncompatibleModel:
    return IncompatibleModel()

@pytest.fixture(scope="class")
def dummy_regression_data() -> pd.DataFrame:
    # Create a simple regression dataset with numerical and categorical features
    dataset = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'feature3': [100.0, 200.0, 300.0, 400.0, 500.0],
        'feature4': ["A", "B", "A", "B", "A"],
        'feature5': [True, False, True, False, True],
        'target': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    return pd.DataFrame(dataset)

@pytest.fixture
def compatible_model(dummy_regression_data) -> SklearnModel:
    # Train a simple dummy regressor
    X = dummy_regression_data.drop(columns=['target'])
    y = dummy_regression_data['target']

    model = DummyRegressor(strategy="median")
    model.fit(X, y)

    return SklearnModel(model)

@pytest.fixture
def valid_public_data(dummy_regression_data) -> PublicData:
    data : pd.DataFrame = dummy_regression_data.drop(columns=['target'])
    targets : pd.Series = dummy_regression_data['target']
    target_name : str = 'target'
    column_names: list[str] = dummy_regression_data.drop(columns=['target']).columns
    continuous_column_names: list[str] = (dummy_regression_data.drop(columns=['target'])
                                          .select_dtypes(include=['float', 'int']).columns.to_list())
    categorical_column_names: list[str] = (dummy_regression_data.drop(columns=['target'])
                                           .select_dtypes(include=['object', 'bool']).columns.to_list())
    immutable_column_names: list[str] = ['feature1']
    feasible_values : dict = build_feasible_values(dummy_regression_data)

    return PublicData(
        data=data,
        targets=targets,
        target_name=target_name,
        column_names=column_names,
        continuous_column_names=continuous_column_names,
        categorical_column_names=categorical_column_names,
        immutable_column_names=immutable_column_names,
        feasible_values=feasible_values
    )

def build_feasible_values(df: pd.DataFrame) -> dict:
    """
    Build feasible values for each feature in a dataframe, excluding the 'target' column.
    """
    feasible_values = {}

    for col in df.columns:
        if col == "target":
            continue  # skip target

        if pd.api.types.is_numeric_dtype(df[col]):
            feasible_values[col] = [df[col].min(), df[col].max()]
        else:
            feasible_values[col] = df[col].dropna().unique().tolist()

    return feasible_values

@pytest.fixture
def basic_explainer(compatible_model, valid_public_data) -> BasicExplainer:
    return BasicExplainer(
        model=compatible_model,
        data=valid_public_data
    )

class TestBaseExplainer:

    def test_valid_inputs(self, basic_explainer, compatible_model, valid_public_data):
        explainer = BasicExplainer(
            model=compatible_model,
            data=valid_public_data
        )
        assert isinstance(explainer, BasicExplainer)
        assert isinstance(explainer, BaseExplainer)
        assert isinstance(explainer.model, SklearnModel)
        assert isinstance(explainer.model, BaseModel)
        assert isinstance(explainer.data, BaseData)
        assert isinstance(explainer.data, PublicData)

    def test_invalid_data(self, compatible_model):
        # Create a dummy data object that is not an instance of BaseData
        class InvalidData:
            pass

        invalid_data = InvalidData()

        with pytest.raises(ConfigurationError) as exc_info:
            BasicExplainer(
                model=compatible_model,
                data=invalid_data,
            )

        err = exc_info.value
        assert err.message == "Explainer requires data to be an instance of BaseData."
        assert err.param == "data"
        assert err.hint.startswith("Please provide")
        assert err.config == {"data_type": "InvalidData"}

    def test_property_getters(self, basic_explainer, compatible_model, valid_public_data):
        """Ensure that model, data, and explainer properties return correct objects."""
        # Model and data should be exactly the same objects passed during init
        assert basic_explainer.model is compatible_model
        assert basic_explainer.data is valid_public_data

        # Explainer is created by subclass _create_explainer (in BasicExplainer it's None by default)
        assert basic_explainer.explainer is None

    def test_validate_sample_dtypes(self, basic_explainer):
        # Create a sample with incorrect dtypes
        sample = np.array([[0.20, 20.0, 300.0, "A", True]])

        with pytest.raises(ConfigurationError) as exc_info:
            basic_explainer.generate_counterfactuals(sample=sample)

        err = exc_info.value
        assert err.message == "Sample must be a pandas DataFrame or Series."
        assert err.param == "sample"
        assert err.hint.startswith("Please provide")
        assert err.config == {"sample_type": "ndarray"}

    def test_validate_sample_column_mismatches(self, basic_explainer, valid_public_data):
        """Ensure validate_sample raises errors for missing, extra, or duplicate columns."""

        # Missing required column
        sample_missing = valid_public_data.data.drop(columns=[valid_public_data.column_names[0]])
        with pytest.raises(ValueError) as exc_info_missing:
            basic_explainer.validate_sample(sample_missing)
        assert "Missing columns" in str(exc_info_missing.value)

        # Extra unexpected column
        sample_extra = valid_public_data.data.copy()
        sample_extra["extra_feature"] = 999
        with pytest.raises(ValueError) as exc_info_extra:
            basic_explainer.validate_sample(sample_extra)
        assert "Unexpected columns" in str(exc_info_extra.value)

        # Duplicate columns
        sample_dupes = valid_public_data.data.copy()
        sample_dupes = sample_dupes.rename(columns={valid_public_data.column_names[0]: "feature2"})
        with pytest.raises(ValueError) as exc_info_dupes:
            basic_explainer.validate_sample(sample_dupes)
        assert "Duplicate column names" in str(exc_info_dupes.value)

    def test_validate_sample_accepts_series_and_dataframe(self, basic_explainer, valid_public_data):
        """Ensure validate_sample works with both DataFrame and Series inputs."""

        # DataFrame should pass validation
        df_sample = valid_public_data.data.iloc[[0]]  # keep as DataFrame
        try:
            basic_explainer.validate_sample(df_sample)
        except Exception as e:
            pytest.fail(f"DataFrame validation raised an unexpected error: {e}")

        # Series should also pass validation
        series_sample = valid_public_data.data.iloc[0]  # single row as Series
        try:
            basic_explainer.validate_sample(series_sample)
        except Exception as e:
            pytest.fail(f"Series validation raised an unexpected error: {e}")

    def test_base_explainer_cannot_be_instantiated(self, compatible_model, valid_public_data):
        """Ensure BaseExplainer cannot be instantiated directly due to abstract methods."""
        with pytest.raises(TypeError) as exc_info:
            BaseExplainer(model=compatible_model, data=valid_public_data)

        assert "abstract class" in str(exc_info.value)
        assert "_create_explainer" in str(exc_info.value)
        assert "_validate_sample" in str(exc_info.value)
        assert "generate_counterfactuals" in str(exc_info.value)

    def test_validate_sample_case_insensitive_columns(self, basic_explainer, valid_public_data):
        """Ensure column name matching is case-insensitive."""

        # Create a sample where all column names are uppercased
        df_upper = valid_public_data.data.copy()
        df_upper.columns = [c.upper() for c in valid_public_data.column_names]

        # Should validate without error despite case differences
        try:
            basic_explainer.validate_sample(df_upper)
        except Exception as e:
            pytest.fail(f"Validation failed for case-insensitive columns: {e}")

