from typing import Any, TypeVar

from celia._errors import ConfigurationError

T = TypeVar("T", bound=type[Any])


def requires_torch_class(cls: T) -> T:
    """
    Class decorator to ensure that PyTorch is installed before instantiating the class.

    Raises
    ------
    ImportError
        If torch is not installed.
    """
    original_init = cls.__init__

    def new_init(self: Any, *args: object, **kwargs: object) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:
            message = f"{cls.__name__} requires PyTorch, but it is not installed."
            raise ConfigurationError(
                message=message,
                config={"class": cls.__name__, "required_package": "torch"},
                param="torch",
                hint="Install it using: `pip install celia[torch]`.",
                source=f"{cls.__name__}.__init__",
            ) from e

        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls
