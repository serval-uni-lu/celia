from typing import Any

from ._base import BaseExplainer, ClassifierExplainer, RegressorExplainer
from .bugdoc import BugDocClassifierExplainer, BugDocRegressorExplainer
from .clear import CLEARClassifierExplainer
from .dice import DiceClassifierExplainer, DiceRegressorExplainer
from .grace import GRACEClassifierExplainer
from .growing_spheres import GrowingSpheresClassifierExplainer
from .nnce import NNCEClassifierExplainer, NNCERegressorExplainer

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # torch extra
    "CounterGANClassifierExplainer": (".countergan", "CounterGANClassifierExplainer"),
    "CCHVAEClassifierExplainer": (".cchvae", "CCHVAEClassifierExplainer"),
    # ocean extra
    "OCEANClassifierExplainer": (".ocean", "OCEANClassifierExplainer"),
    # stochastic extra
    "FastARClassifierExplainer": (".fastar", "FastARClassifierExplainer"),
}

__all__ = [
    "BaseExplainer",
    "RegressorExplainer",
    "ClassifierExplainer",
    "CCHVAEClassifierExplainer",
    "CounterGANClassifierExplainer",
    "DiceClassifierExplainer",
    "DiceRegressorExplainer",
    "FastARClassifierExplainer",
    "GRACEClassifierExplainer",
    "GrowingSpheresClassifierExplainer",
    "NNCEClassifierExplainer",
    "NNCERegressorExplainer",
    "CLEARClassifierExplainer",
    "OCEANClassifierExplainer",
    "BugDocRegressorExplainer",
    "BugDocClassifierExplainer",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            from importlib import import_module

            module = import_module(module_path, package=__name__)
        except ImportError as e:
            msg = (
                f"{name} requires additional dependencies. "
                f"Install the appropriate extra: uv sync --extra <torch|ocean|stochastic>"
            )
            raise ImportError(msg) from e
        return getattr(module, attr_name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
