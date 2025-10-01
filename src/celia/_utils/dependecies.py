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
            raise ImportError(
                f"{cls.__name__} requires PyTorch. Install it with:\n\n"
                "    pip install celia[torch]\n"
            ) from e
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls
