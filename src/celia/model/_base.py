from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, Union
from sklearn.base import BaseEstimator
import numpy as np

from celia._utils.dependecies import requires_torch_class
from celia._errors import ConfigurationError

if TYPE_CHECKING:
    import torch
    from torch import nn, Tensor

class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class to wrap models from different backends in Celia.
    All model classes should inherit from this class and implement the required methods.
    """
    def __init__(self, model: Any):
        """
        Initialize the BaseModel with a model instance.

        Parameters
        ----------
        model : Any
            The model instance to be wrapped.
        """
        self._model = model

    @property
    def model(self):
        """
        Get the wrapped model instance.

        Returns
        -------
        Any
            The wrapped model instance.
        """
        return self._model

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the model.

        Parameters
        ----------
        X : Any
            The data to make predictions on.

        Returns
        -------
        Any
            Predictions made by the model.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Make probability predictions using the model.

        Parameters
        ----------
        X : Any
            The data to make probability predictions on.

        Returns
        -------
        Any
            Probability predictions made by the model.
        """
        pass

class SklearnModel(BaseModel):
    """
    Wrapper for scikit-learn models.
    """
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support probability predictions.")

@requires_torch_class
class TorchModel(BaseModel):
    """
    Wrapper for PyTorch models to standardize prediction interface,
    while exposing the raw model for gradient-based methods (e.g., GRACE).
    """

    def __init__(self, model: "nn.Module"):
        try:
            import torch
        except ImportError as e:
            raise ConfigurationError(
                message="TorchModel requires 'torch', which is not currently installed.",
                config={"model_wrapper": "TorchModel"},
                param="torch",
                hint="Install it using: `pip install celia[torch]`.",
                source="TorchModel.__init__"
            ) from e
        super().__init__(model)

    @property
    def raw_model(self) -> "nn.Module":
        """
        Return the underlying PyTorch model.
        Used for methods that need access to gradients or model internals.

        Returns
        -------
        torch.nn.Module
        """
        return self._model

    def predict(self, X: Union[np.ndarray, "Tensor"]) -> np.ndarray:
        import torch

        self._model.eval()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        with torch.no_grad():
            outputs = self._model(X)
            predicted = torch.argmax(outputs, dim=1)
        return predicted.cpu().numpy()

    def predict_proba(self, X: Union[np.ndarray, "Tensor"]) -> np.ndarray:
        import torch

        self._model.eval()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        with torch.no_grad():
            outputs = self._model(X)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()
