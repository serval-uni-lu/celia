from __future__ import annotations

from unittest.mock import patch

import pytest

from celia._utils.dependencies import requires_ocean_class, requires_torch_class
from celia.errors import ConfigurationError


def _make_torch_class():
    @requires_torch_class
    class _NeedsTorch:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    return _NeedsTorch


def _make_ocean_class():
    @requires_ocean_class
    class _NeedsOcean:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    return _NeedsOcean


def _block_import(*blocked_names: str):
    """Return a side_effect for ``builtins.__import__`` that raises ``ImportError`` for *blocked_names*."""
    import builtins

    real_import = builtins.__import__

    def _guarded(name: str, *args: object, **kwargs: object):
        if name in blocked_names:
            raise ImportError(f"mocked: no module named {name!r}")
        return real_import(name, *args, **kwargs)

    return _guarded


class TestRequiresTorchClass:
    def test_instantiation_succeeds_when_torch_available(self):
        pytest.importorskip("torch")
        cls = _make_torch_class()
        obj = cls(value=42)
        assert obj.value == 42

    def test_raises_configuration_error_when_torch_missing(self):
        cls = _make_torch_class()
        with patch("builtins.__import__", side_effect=_block_import("torch")):
            with pytest.raises(ConfigurationError) as exc_info:
                cls()

        assert exc_info.value.param == "torch"
        assert "celia[torch]" in exc_info.value.hint

    def test_preserves_class_identity(self):
        cls = _make_torch_class()
        assert cls.__name__ == "_NeedsTorch"


class TestRequiresOceanClass:
    def test_instantiation_succeeds_when_ocean_available(self):
        pytest.importorskip("ocean")
        pytest.importorskip("gurobipy")
        cls = _make_ocean_class()
        obj = cls(value=7)
        assert obj.value == 7

    def test_raises_configuration_error_when_ocean_missing(self):
        cls = _make_ocean_class()
        with patch("builtins.__import__", side_effect=_block_import("ocean")):
            with pytest.raises(ConfigurationError) as exc_info:
                cls()

        assert exc_info.value.param == "ocean"
        assert "celia[ocean]" in exc_info.value.hint

    def test_raises_configuration_error_when_gurobipy_missing(self):
        cls = _make_ocean_class()
        with patch("builtins.__import__", side_effect=_block_import("gurobipy")):
            with pytest.raises(ConfigurationError) as exc_info:
                cls()

        assert exc_info.value.param == "ocean"
        assert "gurobipy" in exc_info.value.message

    def test_raises_configuration_error_when_both_missing(self):
        cls = _make_ocean_class()
        with patch("builtins.__import__", side_effect=_block_import("ocean", "gurobipy")):
            with pytest.raises(ConfigurationError) as exc_info:
                cls()

        assert "oceanpy" in exc_info.value.message
        assert "gurobipy" in exc_info.value.message
        assert "they are" in exc_info.value.message

    def test_preserves_class_identity(self):
        cls = _make_ocean_class()
        assert cls.__name__ == "_NeedsOcean"
