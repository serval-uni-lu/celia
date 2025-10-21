import pytest
import pandas as pd
from sklearn.dummy import DummyRegressor
from celia.model import SklearnModel
from celia.data import PublicData


@pytest.fixture(scope="session")
def dummy_regression_dataframe() -> pd.DataFrame:
    """Small regression dataset with mixed types."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "category": ["A", "B", "A", "B", "A"],
        "target": [0.1, 0.2, 0.3, 0.4, 0.5],
    })


@pytest.fixture(scope="session")
def trained_sklearn_model(dummy_regression_dataframe) -> SklearnModel:
    """Trained SklearnModel wrapper using DummyRegressor."""
    X = dummy_regression_dataframe.drop(columns=["target"])
    y = dummy_regression_dataframe["target"]

    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    return SklearnModel(model)


@pytest.fixture(scope="session")
def valid_public_data(dummy_regression_dataframe) -> PublicData:
    """Minimal valid PublicData object for testing explainers."""
    df = dummy_regression_dataframe
    feasible_values = {
        "feature1": (1.0, 5.0),
        "feature2": (10.0, 50.0),
        "category": ["A", "B"],
    }

    return PublicData(
        data=df.drop(columns=["target"]),
        targets=df["target"],
        target_name="target",
        column_names=["feature1", "feature2", "category"],
        continuous_column_names=["feature1", "feature2"],
        categorical_column_names=["category"],
        immutable_column_names=["feature1"],
        feasible_values=feasible_values,
    )
