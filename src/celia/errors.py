from __future__ import annotations

from typing import Any, Mapping


class CELIAError(Exception):
    """Base exception for all CELIA-related errors.

    Parameters
    ----------
    message
        Description of the configuration problem.
    config
        The configuration mapping that triggered the error, or a relevant subset.
    param
        The specific configuration key that is invalid, if known.
    hint
        Optional remediation advice that helps the user fix the issue, if known.
    source
        Where the configuration came from.
    """

    message: str
    config: Mapping[str, Any] | None
    param: str | None
    hint: str | None
    source: str | None

    def __init__(
        self,
        message: str,
        config: Mapping[str, Any] | None = None,
        param: str | None = None,
        hint: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.config = config
        self.param = param
        self.hint = hint
        self.source = source

    def __str__(self) -> str:
        parts: list[str] = [self.message]

        if self.param:
            parts.append(f"param: {self.param}")

        if self.source:
            parts.append(f"source: {self.source}")

        if self.config:
            parts.append(f"config: {dict(self.config)}")

        if self.hint:
            parts.append(f"hint: {self.hint}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"param={self.param!r}, "
            f"source={self.source!r}, "
            f"config={self.config!r}, "
            f"hint={self.hint!r})"
        )


class ConfigurationError(CELIAError):
    """Raised when a user provides an invalid or incompatible configuration.

    Parameters
    ----------
    message
        Description of the configuration problem.
    config
        The configuration mapping that triggered the error, or a relevant subset.
    param
        The specific configuration key that is invalid, if known.
    hint
        Optional remediation advice that helps the user fix the issue, if known.
    source
        Where the configuration came from.
    """


class MethodError(CELIAError):
    """
    Error raised when an internal assumption in CELIA is violated.
     This error should only occur during development, not in normal use.

    Parameters
    ----------
    message
        Description of the Method problem.
    config
        The configuration mapping that triggered the error, or a relevant subset.
    param
        The specific configuration key that is invalid, if known.
    hint
        Optional remediation advice that helps the user fix the issue, if known.
    source
        Where the configuration came from.
    """


class MethodValueError(MethodError):
    """Raised when a method receives an argument with an inappropriate value.

    Parameters
    ----------
    message
        Description of the value problem.
    config
        The configuration mapping that triggered the error, or a relevant subset.
    param
        The specific configuration key that is invalid, if known.
    hint
        Optional remediation advice that helps the user fix the issue, if known.
    source
        Where the configuration came from.
    """

    def __init__(
        self,
        message: str,
        config: Mapping[str, Any] | None = None,
        param: str | None = None,
        hint: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message or "Data returned by Method is invalid", config, param, hint, source)


class NoCounterfactualsFoundError(CELIAError):
    """Raised when a method failed to find any counterfactuals for any instances."""

    def __init__(
        self,
        message: str = "No counterfactuals found for any instances.",
        config: Mapping[str, Any] | None = None,
        param: str | None = None,
        hint: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message, config=config, param=param, hint=hint, source=source)


class InstancesAreWithinRangeError(CELIAError):
    """Raised when all provided instances are already within the desired target range. Only applicable for regression tasks."""

    def __init__(
        self,
        message: str = "All provided instances are already within the desired target range.",
        config: Mapping[str, Any] | None = None,
        param: str | None = None,
        hint: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message, config=config, param=param, hint=hint, source=source)
