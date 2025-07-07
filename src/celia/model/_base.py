from abc import ABC, abstractmethod
from typing import Any
from sklearn.base import BaseEstimator

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
