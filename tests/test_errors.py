from __future__ import annotations

from celia.errors import ConfigurationError


class TestCELIAErrorStr:
    def test_str_with_all_fields(self):
        err = ConfigurationError(
            message="Something went wrong",
            param="my_param",
            source="MyClass.__init__",
            config={"key": "value"},
            hint="Try doing X instead.",
        )
        text = str(err)
        assert "Something went wrong" in text
        assert "param: my_param" in text
        assert "source: MyClass.__init__" in text
        assert "config:" in text
        assert "hint: Try doing X instead." in text

    def test_str_with_only_message(self):
        err = ConfigurationError(message="Bare error")
        text = str(err)
        assert text == "Bare error"
        assert "param:" not in text
        assert "source:" not in text
        assert "config:" not in text
        assert "hint:" not in text

    def test_str_with_partial_fields(self):
        err = ConfigurationError(message="Oops", param="x", hint="Fix it")
        text = str(err)
        assert "param: x" in text
        assert "hint: Fix it" in text
        assert "source:" not in text
        assert "config:" not in text


class TestCELIAErrorRepr:
    def test_repr_contains_class_and_fields(self):
        err = ConfigurationError(
            message="bad config",
            param="p",
            source="s",
            config={"a": 1},
            hint="h",
        )
        r = repr(err)
        assert r.startswith("ConfigurationError(")
        assert "message='bad config'" in r
        assert "param='p'" in r
        assert "source='s'" in r
        assert "hint='h'" in r
