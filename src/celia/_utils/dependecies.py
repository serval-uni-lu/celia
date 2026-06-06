from typing import Any, TypeVar

from celia.errors import ConfigurationError

T = TypeVar("T", bound=type[Any])


def requires_ocean_class(cls: T) -> T:
    """
    Class decorator to ensure that oceanpy and gurobipy are installed before instantiating the class.

    Raises
    ------
    ConfigurationError
        If oceanpy or gurobipy are not installed.
    """
    original_init = cls.__init__

    def new_init(self: Any, *args: object, **kwargs: object) -> None:
        missing = []
        try:
            import ocean  # noqa: F401  # ty: ignore[unresolved-import]
        except ImportError:
            missing.append("oceanpy")
        try:
            import gurobipy  # noqa: F401  # ty: ignore[unresolved-import]
        except ImportError:
            missing.append("gurobipy")

        if missing:
            missing_str = ", ".join(missing)
            message = f"{cls.__name__} requires {missing_str}, but {'it is' if len(missing) == 1 else 'they are'} not installed."
            raise ConfigurationError(
                message=message,
                config={"class": cls.__name__, "required_packages": missing},
                param="ocean",
                hint="Install them using: `pip install celia[ocean]`.",
                source=f"{cls.__name__}.__init__",
            )

        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


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
