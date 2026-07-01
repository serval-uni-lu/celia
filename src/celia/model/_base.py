from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from celia._utils.dependencies import requires_torch_class

if TYPE_CHECKING:
    from torch import Tensor, nn

# ruff: noqa: ANN001, ANN202


class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class to wrap models from different backends in CELIA.
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
    def model(self) -> Any:
        """The wrapped model instance."""
        return self._model

    @abstractmethod
    def predict(self, x):
        """
        Make predictions using the model.

        Parameters
        ----------
        x : Any
            The data to make predictions on.

        Returns
        -------
        Any
            Predictions made by the model.
        """
        pass

    @abstractmethod
    def predict_proba(self, x):
        """
        Make probability predictions using the model.

        Parameters
        ----------
        x : Any
            The data to make probability predictions on.

        Returns
        -------
        Any
            Probability predictions made by the model.
        """
        pass


class SklearnModel(BaseModel):
    """Wrapper for scikit-learn compatible estimators.

    Delegates ``predict`` and ``predict_proba`` directly to the underlying
    estimator.  Any object that implements the scikit-learn estimator
    interface (``fit`` / ``predict``) can be wrapped — this includes
    pipelines, gradient-boosting libraries (XGBoost, LightGBM), and custom
    estimators.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn estimator (or compatible object) that
        implements ``predict`` and, optionally, ``predict_proba``.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from celia import SklearnModel
    >>> clf = RandomForestClassifier().fit(X_train, y_train)
    >>> wrapped = SklearnModel(clf)
    >>> wrapped.predict(X_test)
    """

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)
        message = f"The model of type {type(self.model)} does not support probability predictions."
        raise NotImplementedError(message)


@requires_torch_class
class TorchModel(BaseModel):
    """Wrapper for PyTorch ``nn.Module`` models.

    Converts inputs to tensors, runs inference in ``eval`` mode with
    gradients disabled, and returns NumPy arrays.  The original module is
    accessible via :attr:`raw_model` for gradient-based explainers
    (e.g., GRACE).

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch module whose forward pass returns class logits.

    Examples
    --------
    >>> import torch.nn as nn
    >>> from celia import TorchModel
    >>> net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 2))
    >>> wrapped = TorchModel(net)
    >>> wrapped.predict(X_test)
    """

    def __init__(self, model: "nn.Module"):
        super().__init__(model)

    @property
    def raw_model(self) -> "nn.Module":
        """The original PyTorch model."""
        return self._model

    def predict(self, x: pd.DataFrame | pd.Series | np.ndarray | "Tensor") -> np.ndarray:
        import torch

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, pd.Series):
            x = x.to_numpy().reshape(1, -1)

        self._model.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x, copy=True)).float()

        with torch.no_grad():
            outputs = self._model(x)
            predicted = torch.argmax(outputs, dim=1)
        return predicted.cpu().numpy()

    def predict_proba(self, x: pd.DataFrame | pd.Series | np.ndarray | "Tensor") -> np.ndarray:
        import torch

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, pd.Series):
            x = x.to_numpy().reshape(1, -1)

        self._model.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x, copy=True)).float()

        with torch.no_grad():
            outputs = self._model(x)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()
