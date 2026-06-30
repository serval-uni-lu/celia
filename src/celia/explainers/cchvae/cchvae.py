from __future__ import annotations

from typing import Any

import pandas as pd
from cchvae import CCHVAE
from cchvae import Counterfactual as CCHVAECounterfactual
from cchvae.errors import CCHVAEError, CCHVAEValueError
from cchvae.types import VALID_FEATURE_TYPES, FeatureType

from celia._utils.dependencies import requires_torch_class
from celia.counterfactuals import Counterfactual
from celia.data import Data
from celia.data._base import BaseData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.model import BaseModel


@requires_torch_class
class CCHVAEClassifierExplainer(ClassifierExplainer):
    """C-CHVAE: Counterfactual Conditional Heterogeneous Variational Autoencoder.

    Trains a conditional heterogeneous VAE on the background dataset and
    searches latent-space hyperspheres to produce counterfactuals that flip
    the classifier's prediction to a user-specified ``target_class``.

    Because C-CHVAE is an amortized, class-conditioned method,
    ``target_class`` must be fixed at construction time.

    Parameters
    ----------
    model : BaseModel
        A CELIA model wrapper (``SklearnModel`` or ``TorchModel``).
    data : Data
        Training data with metadata (feature names, types, constraints).
    target_class : int or str
        The desired class for counterfactuals. Required.
    feature_types : dict[str, str] or None, default=None
        Per-column type override (``"real"``, ``"count"``, ``"cat"``).
        When ``None``, types are inferred from ``Data`` metadata.
    latent_dim : int, default=2
        Dimensionality of the VAE latent space.
    intermediate_dim : int, default=5
        Width of VAE hidden layers.
    categorical_latent_dim : int, default=3
        Latent dimension for categorical features.
    learning_rate : float, default=1e-3
        Optimiser learning rate.
    epochs : int, default=80
        Number of VAE training epochs.
    batch_size : int, default=100
        Training batch size.
    device : str, default="cpu"
        PyTorch device (``"cpu"`` or ``"cuda"``).
    random_state : int or None, default=619
        Random seed for reproducibility.
    verbose : bool, default=False
        If ``True``, print progress during training.

    Raises
    ------
    ConfigurationError
        If ``data`` is not a ``Data`` instance, ``target_class`` is missing,
        or the upstream C-CHVAE library rejects the configuration.

    References
    ----------
    Pawelczyk, M., Broelemann, K., & Kasneci, G. (2020). Learning
    Model-Agnostic Counterfactual Explanations for Tabular Data. WWW '20.

    Note
    ----
    **Supported constraints:** immutable features.

    Examples
    --------
    >>> from celia import Data, SklearnModel, CCHVAEClassifierExplainer
    >>> explainer = CCHVAEClassifierExplainer(
    ...     model=sklearn_model, data=data, target_class=1,
    ... )
    >>> cf = explainer.generate_counterfactuals(sample)
    """

    def __init__(
        self,
        model: BaseModel,
        data: BaseData,
        *args: object,
        **kwargs: object,
    ) -> None:
        if not isinstance(data, Data):
            message = "CCHVAEClassifierExplainer requires data to be an instance of Data."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "Data", "received": type(data).__name__},
                source="CCHVAEClassifierExplainer.__init__",
            )

        if "target_class" not in kwargs:
            message = "CCHVAEClassifierExplainer requires `target_class` to be provided."
            raise ConfigurationError(
                message=message,
                param="target_class",
                hint=(
                    "Pass `target_class=<int|str>` — C-CHVAE trains a class-conditioned VAE, "
                    "so the target must be fixed at construction time."
                ),
                source="CCHVAEClassifierExplainer.__init__",
            )

        # Pop all CCHVAE-specific kwargs BEFORE delegating to BaseExplainer so
        # they don't leak through to _create_explainer's *args/**kwargs.
        self._target_class = kwargs.pop("target_class")
        self._feature_types_override: dict[str, FeatureType] | None = kwargs.pop("feature_types", None)  # type: ignore[assignment]
        self._latent_dim: int = kwargs.pop("latent_dim", 2)  # type: ignore[assignment]
        self._intermediate_dim: int = kwargs.pop("intermediate_dim", 5)  # type: ignore[assignment]
        self._categorical_latent_dim: int = kwargs.pop("categorical_latent_dim", 3)  # type: ignore[assignment]
        self._learning_rate: float = kwargs.pop("learning_rate", 1e-3)  # type: ignore[assignment]
        self._epochs: int = kwargs.pop("epochs", 80)  # type: ignore[assignment]
        self._batch_size: int = kwargs.pop("batch_size", 100)  # type: ignore[assignment]
        self._device: str = kwargs.pop("device", "cpu")  # type: ignore[assignment]
        self._random_state: int | None = kwargs.pop("random_state", 619)  # type: ignore[assignment]
        self._verbose: bool = kwargs.pop("verbose", False)  # type: ignore[assignment]

        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(
        self,
        model: BaseModel,
        data: BaseData,
        *args: object,
        **kwargs: object,
    ) -> CCHVAE:
        """Train the underlying C-CHVAE model on the background data."""
        if not isinstance(data, Data):  # defensive — already checked in __init__
            message = "CCHVAEClassifierExplainer requires data to be an instance of Data."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "Data", "received": type(data).__name__},
                source="CCHVAEClassifierExplainer._create_explainer",
            )

        resolved_feature_types = self._resolve_feature_types(self._feature_types_override, data)
        immutable = data.immutable_column_names or []

        try:
            return CCHVAE(
                classifier=model,  # type: ignore[invalid-argument-type]
                background_data=data.data,
                target_class=self._target_class,  # type: ignore[invalid-argument-type]
                immutable_features=immutable,
                feature_types=resolved_feature_types,
                latent_dim=self._latent_dim,
                intermediate_dim=self._intermediate_dim,
                categorical_latent_dim=self._categorical_latent_dim,
                learning_rate=self._learning_rate,
                epochs=self._epochs,
                batch_size=self._batch_size,
                device=self._device,
                random_state=self._random_state,
                verbose=self._verbose,
            )
        except CCHVAEValueError as exc:
            message = f"C-CHVAE rejected the provided data/configuration: {exc}"
            raise ConfigurationError(
                message=message,
                param="data",
                source="CCHVAEClassifierExplainer._create_explainer",
            ) from exc
        except CCHVAEError as exc:
            message = f"C-CHVAE failed to initialize: {exc}"
            raise ConfigurationError(
                message=message,
                source="CCHVAEClassifierExplainer._create_explainer",
            ) from exc

    @staticmethod
    def _resolve_feature_types(
        override: dict[str, FeatureType] | None,
        data: Data,
    ) -> dict[str, FeatureType] | None:
        """
        Merge an explicit ``feature_types`` override with types inferred from ``Data``.

        The returned dict is intentionally partial: any column that cannot be
        classified from the override or from ``Data``'s
        ``continuous_column_names`` / ``categorical_column_names`` is **omitted**
        so that C-CHVAE's own ``infer_schema`` handles it at construction time.

        Rules
        -----
        1. If ``override`` is ``None`` and ``Data`` has no
           continuous/categorical classification, return ``None`` so C-CHVAE
           infers every column.
        2. Validate ``override``:
             - Every key must appear in ``data.column_names``, else
               ``ConfigurationError(param="feature_types")``.
             - Every value must be in :data:`cchvae.types.VALID_FEATURE_TYPES`,
               else ``ConfigurationError(param="feature_types")``.
        3. For every column in ``data.column_names`` not already supplied by
           the override:
             - categorical (per Data)         -> ``"cat"``
             - continuous with integer dtype        -> ``"count"``
             - continuous with any other dtype      -> ``"real"``
             - unclassified                         -> left out (C-CHVAE infers)
        """
        if override is not None:
            unknown = [col for col in override if col not in data.column_names]
            if unknown:
                message = (
                    f"`feature_types` contains column names that are not present in "
                    f"Data.column_names: {sorted(unknown)}"
                )
                raise ConfigurationError(
                    message=message,
                    param="feature_types",
                    config={"unknown_columns": sorted(unknown), "expected_columns": list(data.column_names)},
                    hint="Ensure every key in `feature_types` matches a column in the dataset.",
                    source="CCHVAEClassifierExplainer._resolve_feature_types",
                )

            invalid = {col: ftype for col, ftype in override.items() if ftype not in VALID_FEATURE_TYPES}
            if invalid:
                message = (
                    f"`feature_types` contains invalid feature type(s): {invalid}. "
                    f"Allowed values are {sorted(VALID_FEATURE_TYPES)}."
                )
                raise ConfigurationError(
                    message=message,
                    param="feature_types",
                    config={"invalid": invalid, "allowed": list(VALID_FEATURE_TYPES)},
                    hint=f"Use one of: {sorted(VALID_FEATURE_TYPES)}.",
                    source="CCHVAEClassifierExplainer._resolve_feature_types",
                )

        has_publicdata_classification = bool(data.continuous_column_names) or bool(data.categorical_column_names)
        if override is None and not has_publicdata_classification:
            return None

        resolved: dict[str, FeatureType] = dict(override) if override else {}

        continuous = set(data.continuous_column_names or [])
        categorical = set(data.categorical_column_names or [])

        for col in data.column_names:
            if col in resolved:
                continue
            if col in categorical:
                resolved[col] = "cat"
            elif col in continuous:
                if pd.api.types.is_integer_dtype(data.data[col]):
                    resolved[col] = "count"
                else:
                    resolved[col] = "real"
            # else: leave unclassified so C-CHVAE infers from dtype

        return resolved or None

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        """Delegate counterfactual generation to the trained C-CHVAE explainer."""
        if isinstance(sample, pd.Series):
            sample_df = sample.to_frame().T
            was_single = True
        else:
            sample_df = sample
            was_single = sample_df.shape[0] == 1

        num_counterfactuals = int(kwargs.get("num_counterfactuals", 5))  # type: ignore[arg-type]
        search_samples = int(kwargs.get("search_samples", 1000))  # type: ignore[arg-type]
        max_iterations = int(kwargs.get("max_iterations", 500))  # type: ignore[arg-type]
        step_size = float(kwargs.get("step_size", 0.5))  # type: ignore[arg-type]
        norm = int(kwargs.get("norm", 2))  # type: ignore[arg-type]

        try:
            upstream_result: Any = self.explainer.generate_counterfactuals(
                sample_df,
                num_counterfactuals=num_counterfactuals,
                search_samples=search_samples,
                max_iterations=max_iterations,
                step_size=step_size,
                norm=norm,
            )
        except CCHVAEError as exc:
            # Upstream raises this when no counterfactuals are found for any instance.
            raise NoCounterfactualsFoundError(
                message=str(exc),
                source="CCHVAEClassifierExplainer._generate_counterfactuals",
            ) from exc

        if isinstance(upstream_result, CCHVAECounterfactual):
            upstream_list: list[CCHVAECounterfactual] = [upstream_result]
        else:
            upstream_list = list(upstream_result)

        if not upstream_list:
            raise NoCounterfactualsFoundError(
                source="CCHVAEClassifierExplainer._generate_counterfactuals",
            )

        converted = [
            Counterfactual(
                original_instance=cf.original_instance,
                counterfactual_instance=cf.counterfactuals,
                original_prediction=cf.original_prediction,
                counterfactual_prediction=cf.counterfactual_prediction,
            )
            for cf in upstream_list
        ]

        if was_single and len(converted) == 1:
            return converted[0]
        return converted
