from celia._errors import ConfigurationError

def requires_torch_class(cls):
    """
    Class decorator to enforce that torch is installed before instantiating the class.

    Raises
    ------
    ImportError
        If torch is not installed.
    """
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ConfigurationError(
                message=(
                    f"{cls.__name__} requires PyTorch, but it is not installed."
                ),
                config={"class": cls.__name__, "required_package": "torch"},
                param="torch",
                hint="Install it using: `pip install celia[torch]`.",
                source=f"{cls.__name__}.__init__"
            ) from e
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls
