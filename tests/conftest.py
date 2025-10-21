import pytest
import pandas as pd
from sklearn.dummy import DummyRegressor
from celia.model import SklearnModel
from celia.data import PublicData

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

def build_feasible_values(df: pd.DataFrame) -> dict:
    """
    Build feasible values for each feature in a dataframe, excluding the 'target' column.
    """
    feasible_values = {}

    for col in df.columns:
        if col == "target":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feasible_values[col] = [df[col].min(), df[col].max()]
        else:
            feasible_values[col] = df[col].dropna().unique().tolist()

    return feasible_values

@pytest.fixture
def create_model_without_predict() -> ModelWithoutPredict:
    return ModelWithoutPredict()

@pytest.fixture
def create_incompatible_model() -> IncompatibleModel:
    return IncompatibleModel()

@pytest.fixture
def dummy_regression_dataframe() -> pd.DataFrame:
    """Small regression dataset with mixed types."""
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
def dummy_regression_dataframe_encoded(dummy_regression_dataframe) -> pd.DataFrame:
    # Simple encoding: Convert categorical columns to category dtype and then to codes
    df_encoded = dummy_regression_dataframe.copy()
    for col in df_encoded.select_dtypes(include=['object', 'bool']).columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    return df_encoded

@pytest.fixture
def dummy_test_regression_dataframe_encoded() -> pd.DataFrame:
    # A small test dataset with the same structure as dummy_regression_dataframe_encoded
    dataset = {
        'feature1': [6.0, 7.0],
        'feature2': [60.0, 70.0],
        'feature3': [600.0, 700.0],
        'feature4': [0, 1],  # Encoded values for "A" and "B"
        'feature5': [1, 0],  # Encoded values for True and False
        'target': [0.6, 0.7]
    }
    return pd.DataFrame(dataset)

@pytest.fixture
def celia_public_data_without_encoded_data(dummy_regression_dataframe) -> PublicData:
    data : pd.DataFrame = dummy_regression_dataframe.drop(columns=['target'])
    targets : pd.Series = dummy_regression_dataframe['target']
    target_name : str = 'target'
    column_names: list[str] = dummy_regression_dataframe.drop(columns=['target']).columns
    continuous_column_names: list[str] = (dummy_regression_dataframe.drop(columns=['target'])
                                          .select_dtypes(include=['float', 'int']).columns.to_list())
    categorical_column_names: list[str] = (dummy_regression_dataframe.drop(columns=['target'])
                                           .select_dtypes(include=['object', 'bool']).columns.to_list())
    immutable_column_names: list[str] = ['feature1']
    feasible_values : dict = build_feasible_values(dummy_regression_dataframe)

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

@pytest.fixture
def celia_public_data_with_encoded_data(dummy_regression_dataframe_encoded) -> PublicData:
    data : pd.DataFrame = dummy_regression_dataframe_encoded.drop(columns=['target'])
    targets : pd.Series = dummy_regression_dataframe_encoded['target']
    target_name : str = 'target'
    column_names: list[str] = dummy_regression_dataframe_encoded.drop(columns=['target']).columns
    continuous_column_names: list[str] = (dummy_regression_dataframe_encoded.drop(columns=['target'])
                                          .select_dtypes(include=['float', 'int']).columns.to_list())
    categorical_column_names: list[str] = (dummy_regression_dataframe_encoded.drop(columns=['target'])
                                           .select_dtypes(include=['object', 'bool']).columns.to_list())
    immutable_column_names: list[str] = ['feature1']
    feasible_values : dict = build_feasible_values(dummy_regression_dataframe_encoded)

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

@pytest.fixture
def model_trained_without_encoded_data(dummy_regression_dataframe) -> SklearnModel:
    # Train a simple dummy regressor
    X = dummy_regression_dataframe.drop(columns=['target'])
    y = dummy_regression_dataframe['target']

    model = DummyRegressor(strategy="median")
    model.fit(X, y)

    return SklearnModel(model)

@pytest.fixture
def model_trained_with_encoded_data(dummy_regression_dataframe_encoded) -> SklearnModel:
    X = dummy_regression_dataframe_encoded.drop(columns=['target'])
    y = dummy_regression_dataframe_encoded['target']

    model = DummyRegressor(strategy="median")
    model.fit(X, y)

    return SklearnModel(model)

