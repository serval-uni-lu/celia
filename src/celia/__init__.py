from __future__ import annotations

import importlib.metadata
from typing import Any

from .counterfactuals import Counterfactual
from .data import BaseData, Data
from .errors import (
    CELIAError,
    ConfigurationError,
    InstancesAreWithinRangeError,
    MethodError,
    MethodValueError,
    NoCounterfactualsFoundError,
)
from .explainers import (
    BaseExplainer,
    BugDocClassifierExplainer,
    BugDocRegressorExplainer,
    ClassifierExplainer,
    CLEARClassifierExplainer,
    DiceClassifierExplainer,
    DiceRegressorExplainer,
    GRACEClassifierExplainer,
    GrowingSpheresClassifierExplainer,
    NNCEClassifierExplainer,
    NNCERegressorExplainer,
    RegressorExplainer,
)
from .model import BaseModel, SklearnModel, TorchModel

__version__ = importlib.metadata.version("celia")

_LAZY_IMPORTS: dict[str, str] = {
    "CounterGANClassifierExplainer": "CounterGANClassifierExplainer",
    "CCHVAEClassifierExplainer": "CCHVAEClassifierExplainer",
    "OCEANClassifierExplainer": "OCEANClassifierExplainer",
    "FastARClassifierExplainer": "FastARClassifierExplainer",
}

__all__ = [
    # version
    "__version__",
    # model
    "BaseModel",
    "SklearnModel",
    "TorchModel",
    # data
    "BaseData",
    "Data",
    # counterfactuals
    "Counterfactual",
    # errors
    "CELIAError",
    "ConfigurationError",
    "MethodError",
    "MethodValueError",
    "NoCounterfactualsFoundError",
    "InstancesAreWithinRangeError",
    # explainers (eager)
    "BaseExplainer",
    "RegressorExplainer",
    "ClassifierExplainer",
    "DiceClassifierExplainer",
    "DiceRegressorExplainer",
    "GRACEClassifierExplainer",
    "GrowingSpheresClassifierExplainer",
    "NNCEClassifierExplainer",
    "NNCERegressorExplainer",
    "CLEARClassifierExplainer",
    "BugDocClassifierExplainer",
    "BugDocRegressorExplainer",
    # explainers (lazy — require extras)
    "CounterGANClassifierExplainer",
    "CCHVAEClassifierExplainer",
    "OCEANClassifierExplainer",
    "FastARClassifierExplainer",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        from . import explainers

        return getattr(explainers, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
