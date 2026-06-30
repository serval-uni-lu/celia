from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from clear_cf import CLEAR

from celia.counterfactuals import Counterfactual
from celia.data import Data
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.model import SklearnModel

if TYPE_CHECKING:
    from clear_cf import RegressionConfig

    from celia.data._base import BaseData
    from celia.model._base import BaseModel


class CLEARClassifierExplainer(ClassifierExplainer):
    """
    CLEAR (Counterfactual Local Explanations via Regression) builds local
    surrogate regression models around each instance and solves them
    analytically to find minimal feature perturbations that cross the
    decision boundary.

    CLEAR requires all features to be numeric. Categorical features must be
    one-hot encoded before use, and the corresponding prefix names passed
    via ``categorical_features``.

    Both ``SklearnModel`` and ``TorchModel`` backends are supported, as long
    as the underlying model exposes ``predict_proba``.

    Parameters
    ----------
    model : BaseModel
        A CELIA model wrapper (``SklearnModel`` or ``TorchModel``).
        The underlying model must support ``predict_proba``.
    data : Data
        Training data with metadata. All features must be numeric.
    num_classes : int, default=2
        Number of classes in the classification problem.
    class_labels : dict[int, str] or None, default=None
        Mapping of class indices to labels. Required when ``num_classes > 2``.
    categorical_features : list[str] or None, default=None
        Prefix names for one-hot encoded categorical feature groups
        (e.g., ``["gender"]`` for columns ``gender_Male``, ``gender_Female``).
    continuous_features : list[str] or None, default=None
        Names of continuous feature columns. If not provided, inferred from
        ``data.continuous_column_names`` or defaults to all columns.
    multi_class_focus : str, default="All"
        For multi-class problems, which class to focus on. ``"All"`` generates
        counterfactuals for all classes.
    number_of_synthetic_samples : int, default=5000
        Number of synthetic observations for neighbourhood construction.
    random_seed : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        If ``True``, print progress messages during computation.
    config : RegressionConfig or None, default=None
        Optional regression configuration for the CLEAR algorithm.

    References
    ----------
    White, A., & d'Avila Garcez, A. (2020).
    Measurable Counterfactual Local Explanations via Regression.
    In *ECAI 2020* (pp. 1529-1536). IOS Press.
    """

    def __init__(self, model: BaseModel, data: BaseData, *args: object, **kwargs: object) -> None:
        if not isinstance(data, Data):
            message = "CLEARClassifierExplainer requires data to be an instance of Data."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "Data", "received": type(data).__name__},
                source="CLEARClassifierExplainer.__init__",
            )

        non_numeric = data.data.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = (
                f"CLEAR requires all features to be numeric. "
                f"Found non-numeric columns: {non_numeric}. "
                f"Categorical features must be one-hot encoded."
            )
            raise ConfigurationError(
                message=message,
                param="data",
                hint="One-hot encode categorical features before passing data to CLEARClassifierExplainer.",
                source="CLEARClassifierExplainer.__init__",
            )

        if isinstance(model, SklearnModel) and not hasattr(model.model, "predict_proba"):
            message = (
                f"CLEARClassifierExplainer requires a model that supports predict_proba. "
                f"The provided {type(model.model).__name__} model does not."
            )
            raise ConfigurationError(
                message=message,
                param="model",
                hint="Use a classifier that supports predict_proba (e.g., set probability=True for SVM).",
                source="CLEARClassifierExplainer.__init__",
            )

        num_classes = int(kwargs.get("num_classes", 2))
        if num_classes < 2:
            message = f"num_classes must be at least 2, got {num_classes}."
            raise ConfigurationError(
                message=message,
                param="num_classes",
                source="CLEARClassifierExplainer.__init__",
            )

        class_labels = kwargs.get("class_labels")
        if num_classes > 2 and class_labels is None:
            message = "class_labels is required when num_classes > 2."
            raise ConfigurationError(
                message=message,
                param="class_labels",
                hint="Provide a dict mapping class indices to labels, e.g. {0: 'A', 1: 'B', 2: 'C'}.",
                source="CLEARClassifierExplainer.__init__",
            )

        self._num_classes: int = num_classes
        if class_labels is None:
            class_labels = {i: str(i) for i in range(num_classes)}
        self._class_labels: dict[int, str] = class_labels

        self._categorical_features: list[str] = kwargs.get("categorical_features") or self._infer_ohe_prefixes(data)
        self._continuous_features: list[str] | None = kwargs.get("continuous_features") or (
            list(data.continuous_column_names) if data.continuous_column_names else None
        )

        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model: BaseModel, data: BaseData, *args: object, **kwargs: object) -> Any:
        self._multi_class_focus: str = str(kwargs.get("multi_class_focus", "All"))
        self._number_of_synthetic_samples: int = int(kwargs.get("number_of_synthetic_samples", 5000))
        self._random_seed: int = int(kwargs.get("random_seed", 42))
        self._verbose: bool = bool(kwargs.get("verbose", False))
        self._config: RegressionConfig | None = kwargs.get("config")

        return CLEAR(model)

    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        """Reject samples containing non-numeric columns."""
        frame = sample.to_frame().T if isinstance(sample, pd.Series) else sample
        non_numeric = frame.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = (
                f"CLEARClassifierExplainer requires all features to be numeric. "
                f"Found non-numeric columns: {non_numeric}. "
                f"Categorical features must be one-hot encoded."
            )
            raise ConfigurationError(
                message=message,
                param="sample",
                hint="One-hot encode categorical features before calling generate_counterfactuals.",
                source="CLEARClassifierExplainer._validate_sample",
            )

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        if isinstance(sample, pd.Series):
            instances = sample.to_frame().T
            is_single = True
        else:
            instances = sample
            is_single = instances.shape[0] == 1

        continuous_features = kwargs.get("continuous_features", self._continuous_features)
        if continuous_features is None:
            continuous_features = (
                list(self.data.continuous_column_names)
                if self.data.continuous_column_names
                else list(self.data.column_names)
            )

        try:
            clear_results = self.explainer.generate_counterfactuals(
                X_train=self.data.data,
                instances_to_explain=instances,
                num_classes=kwargs.get("num_classes", self._num_classes),
                multi_class_focus=kwargs.get("multi_class_focus", self._multi_class_focus),
                class_labels=kwargs.get("class_labels", self._class_labels),
                number_of_synthetic_samples=kwargs.get(
                    "number_of_synthetic_samples", self._number_of_synthetic_samples
                ),
                categorical_features=kwargs.get("categorical_features", self._categorical_features),
                continuous_features=continuous_features,
                random_seed=kwargs.get("random_seed", self._random_seed),
                verbose=kwargs.get("verbose", self._verbose),
                config=kwargs.get("config", self._config),
            )
        except ValueError as exc:
            if "No counterfactuals found" in str(exc):
                raise NoCounterfactualsFoundError(
                    message=str(exc),
                    source="CLEARClassifierExplainer._generate_counterfactuals",
                ) from exc
            raise
        except AttributeError as exc:
            # clear_cf crashes with ``AttributeError: 'DataFrame' object has no
            # attribute 'observation'`` when its internal regression produces an
            # empty boundary DataFrame (e.g., a constant-prediction model). Treat
            # this as "no counterfactuals found" rather than leaking the upstream
            # bug to the user.
            if "observation" in str(exc):
                message = "No counterfactuals found."
                raise NoCounterfactualsFoundError(
                    message=message,
                    source="CLEARClassifierExplainer._generate_counterfactuals",
                ) from exc
            raise

        celia_results: list[Counterfactual] = []
        for cf in clear_results:
            original_pred: int | str = cf.original_class
            cf_pred: list[int] | list[str] = cf.counterfactual_class

            celia_results.append(
                Counterfactual(
                    original_instance=cf.original_instance,
                    counterfactual_instance=cf.counterfactuals,
                    original_prediction=original_pred,
                    counterfactual_prediction=cf_pred,
                )
            )

        return celia_results[0] if is_single else celia_results

    @staticmethod
    def _infer_ohe_prefixes(data: Data) -> list[str]:
        """Recover OHE group prefixes from expanded categorical column names.

        OHE columns follow the ``{prefix}_{value}`` convention.  Columns from
        the same original feature are lexicographically adjacent when sorted,
        so we sort and group by longest common prefix trimmed to the last ``_``.
        """
        if not data.categorical_column_names:
            return []

        sorted_cols = sorted(data.categorical_column_names)
        prefixes: list[str] = []
        i = 0

        while i < len(sorted_cols):
            col = sorted_cols[i]
            if i + 1 < len(sorted_cols):
                # Find character-level common prefix with the next column
                common_len = 0
                for a, b in zip(col, sorted_cols[i + 1]):
                    if a != b:
                        break
                    common_len += 1
                sep = col[:common_len].rfind("_")
                if sep > 0:
                    prefix = col[:sep]
                    pfx = prefix + "_"
                    j = i
                    while j < len(sorted_cols) and sorted_cols[j].startswith(pfx):
                        j += 1
                    if j - i >= 2:
                        prefixes.append(prefix)
                        i = j
                        continue

            # Single-column group fallback
            sep = col.rfind("_")
            prefixes.append(col[:sep] if sep > 0 else col)
            i += 1

        return prefixes
