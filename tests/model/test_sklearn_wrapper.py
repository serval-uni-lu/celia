import pytest
import numpy as np
from sklearn.dummy import DummyClassifier
from celia.model._base import SklearnModel


@pytest.fixture
def trained_sklearn_model():
    """Train a simple scikit-learn model for testing."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 1, 0, 1, 0])
    model = DummyClassifier(strategy="most_frequent")  # always predicts 0
    model.fit(X, y)
    return model

class ModelWithoutProba:
    """Simple dummy model with only a predict() method."""
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

class TestSklearnModel:

    def test_predict_calls_underlying_model(self, trained_sklearn_model):
        """Ensure SklearnModel.predict() delegates directly to the wrapped model."""
        wrapper = SklearnModel(trained_sklearn_model)
        X_test = np.array([[10], [20], [30]])

        expected = trained_sklearn_model.predict(X_test)
        result = wrapper.predict(X_test)

        # Both should be identical numpy arrays
        assert np.array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_predict_proba_calls_underlying_model(self, trained_sklearn_model):
        """Ensure SklearnModel.predict_proba() delegates directly to the wrapped model."""
        wrapper = SklearnModel(trained_sklearn_model)
        X_test = np.array([[10], [20], [30]])

        expected = trained_sklearn_model.predict_proba(X_test)
        result = wrapper.predict_proba(X_test)

        # Both arrays must match and represent valid probability distributions
        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_predict_proba_not_implemented_raises(self):
        """Ensure NotImplementedError is raised if underlying model lacks predict_proba()."""
        model = ModelWithoutProba()
        wrapper = SklearnModel(model)

        X_test = np.array([[1], [2], [3]])

        with pytest.raises(NotImplementedError) as exc_info:
            wrapper.predict_proba(X_test)

        msg = str(exc_info.value).lower()
        assert "does not support probability predictions" in msg

    def test_model_property_returns_original_model(self):
        """Ensure the .model property returns the same model instance passed at initialization."""
        original_model = DummyClassifier(strategy="most_frequent")
        wrapper = SklearnModel(original_model)

        assert wrapper.model is original_model
        assert id(wrapper.model) == id(original_model)