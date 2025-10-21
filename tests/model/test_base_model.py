import pytest
from celia.model import BaseModel
from sklearn.dummy import DummyClassifier


class DummySubClass(BaseModel):
    """Minimal subclass to test BaseModel abstract behavior."""

    def predict(self, X):
        return ["predicted"]

    def predict_proba(self, X):
        return [[0.2, 0.8]]

class MissingPredict(BaseModel):
    """Subclass missing `predict` implementation."""
    def predict_proba(self, X):
        return [[0.1, 0.9]]


class MissingPredictProba(BaseModel):
    """Subclass missing `predict_proba` implementation."""
    def predict(self, X):
        return ["predicted"]

class TestBaseModel():
    def test_cannot_instantiate_base_model(self):
        """Ensure BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseModel(model="dummy")
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_subclass_can_instantiate(self):
        """Ensure subclass implementing abstract methods can be instantiated."""
        dummy_model = DummyClassifier(strategy="most_frequent")
        subclass_instance = DummySubClass(model=dummy_model)
        assert isinstance(subclass_instance, BaseModel)
        assert subclass_instance.model == dummy_model

    def test_model_property_returns_instance(self):
        """Ensure model property returns the wrapped model."""
        dummy = DummyClassifier(strategy="most_frequent")
        subclass = DummySubClass(dummy)
        assert subclass.model is dummy

    def test_predict_and_predict_proba_are_overridden(self):
        """Ensure subclass implements predict and predict_proba methods."""
        dummy_model = DummyClassifier(strategy="most_frequent")
        subclass_instance = DummySubClass(model=dummy_model)

        preds = subclass_instance.predict([[1, 2, 3]])
        probs = subclass_instance.predict_proba([[1, 2, 3]])

        assert preds == ["predicted"]
        assert probs == [[0.2, 0.8]]

    def test_missing_predict_raises_typeerror(self):
        """Ensure subclass missing `predict` cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            MissingPredict(model=DummyClassifier(strategy="most_frequent"))
        msg = str(exc_info.value)
        assert "Can't instantiate abstract class" in msg
        assert "predict" in msg

    def test_missing_predict_proba_raises_typeerror(self):
        """Ensure subclass missing `predict_proba` cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            MissingPredictProba(model=DummyClassifier(strategy="most_frequent"))
        msg = str(exc_info.value)
        assert "Can't instantiate abstract class" in msg
        assert "predict_proba" in msg