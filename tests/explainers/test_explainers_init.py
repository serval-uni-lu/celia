from __future__ import annotations

from unittest.mock import patch

import pytest

import celia.explainers as explainers_pkg


class TestExplainersLazyGetattr:
    def test_happy_path_lazy_import(self):
        """Accessing a lazy-imported name succeeds when the extra is installed."""
        pytest.importorskip("torch")
        cls = getattr(explainers_pkg, "CounterGANClassifierExplainer")
        assert cls.__name__ == "CounterGANClassifierExplainer"

    def test_missing_extra_raises_import_error(self):
        """Accessing a lazy-imported name raises ImportError with install hint when extra is missing."""
        import importlib
        import sys

        saved = sys.modules.pop("celia.explainers.countergan", None)
        try:
            sys.modules["celia.explainers.countergan"] = None  # type: ignore[assignment]
            mod = importlib.reload(explainers_pkg)
            with pytest.raises(ImportError, match="uv sync --extra"):
                getattr(mod, "CounterGANClassifierExplainer")
        finally:
            if saved is not None:
                sys.modules["celia.explainers.countergan"] = saved
            else:
                sys.modules.pop("celia.explainers.countergan", None)
            importlib.reload(explainers_pkg)

    def test_unknown_attribute_raises_attribute_error(self):
        """Accessing a truly non-existent name raises AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(explainers_pkg, "TotallyFakeExplainer")
