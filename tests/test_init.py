from __future__ import annotations

import pytest


class TestCeliaTopLevelInit:
    def test_lazy_getattr_happy_path(self):
        """Accessing an optional-extra explainer via celia.__getattr__ succeeds when the extra is installed."""
        pytest.importorskip("torch")
        import celia

        cls = celia.CounterGANClassifierExplainer
        assert cls.__name__ == "CounterGANClassifierExplainer"

    def test_lazy_getattr_unknown_attribute_raises(self):
        """Accessing a non-existent name on celia raises AttributeError."""
        import celia

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = celia.CompletelyBogusClassName
