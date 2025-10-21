import sys
import pytest
from celia.model._base import TorchModel
from celia._errors import ConfigurationError
import torch
import numpy as np
import pandas as pd

class DummyTorchClassifier(torch.nn.Module):
    """Minimal torch model for deterministic class predictions."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def torch_wrapper():
    """Fixture returning a TorchModel wrapping a simple classifier."""
    model = DummyTorchClassifier()
    return TorchModel(model)

class TestTorchModel:

    def test_torch_not_installed_raises_configuration_error(self, monkeypatch):
        """Ensure TorchModel raises ConfigurationError when torch is unavailable."""
        # Simulate torch being missing
        monkeypatch.setitem(sys.modules, "torch", None)

        class DummyTorchModule:
            pass

        with pytest.raises(ConfigurationError) as exc_info:
            TorchModel(DummyTorchModule())

        err = exc_info.value
        assert err.param == "torch"
        assert "TorchModel" in err.message
        assert "pip install celia[torch]" in err.hint
        assert err.config == {"class": "TorchModel", "required_package": "torch"}
        assert "TorchModel.__init__" in err.source

    def test_raw_model_returns_same_instance(self):
        """Ensure .raw_model returns the same nn.Module object passed at initialization."""
        model = torch.nn.Linear(10, 2)
        wrapper = TorchModel(model)

        assert isinstance(wrapper, TorchModel)
        assert hasattr(wrapper, "raw_model")

        assert wrapper.raw_model is model
        assert id(wrapper.raw_model) == id(model)

    def test_predict_accepts_tensor_input(self, torch_wrapper):
        """Ensure .predict works with torch.Tensor input."""
        X_tensor = torch.randn(3, 4)
        preds = torch_wrapper.predict(X_tensor)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (3,)

    def test_predict_accepts_numpy_input(self, torch_wrapper):
        """Ensure .predict works with numpy.ndarray input."""
        X_numpy = np.random.randn(3, 4)
        preds = torch_wrapper.predict(X_numpy)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (3,)

    def test_predict_invalid_input_raises_pytorch_error(self, torch_wrapper):
        """Ensure invalid input (e.g., pandas.DataFrame) raises a PyTorch-originated error."""
        X_invalid = pd.DataFrame(np.random.randn(3, 4))

        with pytest.raises((TypeError, RuntimeError)) as exc_info:
            torch_wrapper.predict(X_invalid)

        msg = str(exc_info.value).lower()
        assert "tensor" in msg or "expected" in msg or "float" in msg

    @pytest.mark.parametrize("input_type", ["tensor", "numpy"])
    def test_predict_proba_returns_valid_probabilities(self, torch_wrapper, input_type):
        """Ensure .predict_proba returns valid probability distributions."""
        # Generate input in the requested format
        X = torch.randn(5, 4) if input_type == "tensor" else np.random.randn(5, 4)

        probs = torch_wrapper.predict_proba(X)

        assert isinstance(probs, np.ndarray)
        assert probs.shape == (5, 2)

        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), rtol=1e-5, atol=1e-6)

