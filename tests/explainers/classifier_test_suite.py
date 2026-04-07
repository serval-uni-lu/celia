"""
Reusable mixin test suite for ClassifierExplainer subclasses.

Provides core tests (always run) and conditional tests (auto-skipped based on
capability flags). A new classifier explainer can inherit from
``ClassifierExplainerTests`` and set a handful of class attributes to get a
comprehensive, consistent test suite with zero boilerplate.

Example
-------
::

    from tests.explainers.classifier_test_suite import ClassifierExplainerTests
    from celia.explainers.newmethod import NewMethodClassifierExplainer

    class TestNewMethodClassifier(ClassifierExplainerTests):
        explainer_class = NewMethodClassifierExplainer
        supports_sklearn = True
        supports_immutable_features = True
        supports_feasible_values = True
"""

from __future__ import annotations

import functools
from typing import Any

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.model import BaseModel, SklearnModel
from tests.conftest import categorical_select_dtypes

# ---------------------------------------------------------------------------
# Helper: build feasible-value dict from a dataframe (mirrors conftest helper)
# ---------------------------------------------------------------------------


def _build_feasible_values(df: pd.DataFrame, target: str = "target") -> dict[str, Any]:
    feasible_values: dict[str, Any] = {}
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feasible_values[col] = [df[col].min(), df[col].max()]
        else:
            feasible_values[col] = df[col].dropna().unique().tolist()
    return feasible_values


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------


def _skip_unless(flag_name: str):
    """Skip the decorated test when the class attribute *flag_name* is falsy."""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, flag_name, False):
                pytest.skip(f"Skipped: {flag_name} is not enabled")
            return method(self, *args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers for building models / data inside tests
# ---------------------------------------------------------------------------


def _make_public_data(
    df: pd.DataFrame,
    *,
    immutable: list[str] | None = None,
    feasible_values: dict[str, Any] | None = None,
) -> tuple[PublicData, pd.DataFrame]:
    """Build ``PublicData`` from a classification DataFrame."""
    X = df.drop(columns=["target"])
    y = df["target"]

    if feasible_values is None:
        feasible_values = _build_feasible_values(df)

    public_data = PublicData(
        data=X,
        targets=y,
        target_name="target",
        column_names=X.columns.tolist(),
        continuous_column_names=X.select_dtypes(include=["float", "int"]).columns.tolist(),
        categorical_column_names=X.select_dtypes(include=categorical_select_dtypes(with_bool=False)).columns.tolist(),
        immutable_column_names=immutable if immutable is not None else [],
        feasible_values=feasible_values,
    )
    return public_data, X


def _make_sklearn_model(df: pd.DataFrame) -> SklearnModel:
    """Train a ``DecisionTreeClassifier`` on *df* and wrap it in ``SklearnModel``."""
    X = df.drop(columns=["target"])
    y = df["target"]
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X, y)
    return SklearnModel(tree)


def _make_sklearn_dummy_model(df: pd.DataFrame) -> SklearnModel:
    """Train a ``DummyClassifier(strategy='most_frequent')``."""
    X = df.drop(columns=["target"])
    y = df["target"]
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X, y)
    return SklearnModel(dummy)


# ---------------------------------------------------------------------------
# Mixin test suite
# ---------------------------------------------------------------------------


class ClassifierExplainerTests:
    """Mixin providing standard tests for ``ClassifierExplainer`` subclasses.

    Subclasses **must** set at least ``explainer_class``.  All other class
    attributes have sensible defaults and should be overridden to match the
    method's actual capabilities.

    Class Attributes
    ----------------
    explainer_class : type[ClassifierExplainer]
        The concrete explainer class under test.
    explainer_kwargs : dict
        Extra keyword arguments forwarded to the explainer constructor.
    generate_kwargs : dict
        Extra keyword arguments forwarded to ``generate_counterfactuals()``.
    supports_sklearn : bool
        ``True`` if the method accepts ``SklearnModel``.
    supports_torch : bool
        ``True`` if the method accepts ``TorchModel``.
    rejects_sklearn : bool
        ``True`` if the method **actively validates** and rejects ``SklearnModel``
        (raises ``ConfigurationError``).  Only set this when the method enforces
        the model type at init — *not* merely because it's untested with sklearn.
    rejects_torch : bool
        Same as ``rejects_sklearn`` but for ``TorchModel``.
    supports_immutable_features : bool
        ``True`` if the method respects ``immutable_column_names``.
    supports_feasible_values : bool
        ``True`` if the method respects ``feasible_values`` bounds.
    supports_categorical_features : bool
        ``True`` if the method handles categorical (non-encoded) columns.
    requires_encoded_data : bool
        ``True`` if the method requires one-hot encoded input.
    """

    # --- Subclass MUST set these ----------------------------------------
    explainer_class: type[ClassifierExplainer] | None = None
    explainer_kwargs: dict[str, Any] = {}
    generate_kwargs: dict[str, Any] = {}

    # --- Capability flags -----------------------------------------------
    supports_sklearn: bool = True
    supports_torch: bool = False
    rejects_sklearn: bool = False
    rejects_torch: bool = False
    supports_immutable_features: bool = False
    supports_feasible_values: bool = False
    supports_categorical_features: bool = False
    requires_encoded_data: bool = False

    # =====================================================================
    # Internal helper — pick the right model for core tests
    # =====================================================================

    def _get_model(self, df: pd.DataFrame, request: pytest.FixtureRequest) -> BaseModel:
        """Return a compatible model based on the capability flags.

        Prefers sklearn when supported; falls back to the torch fixture
        when the method is torch-only.
        """
        if self.supports_sklearn:
            return _make_sklearn_model(df)
        # torch-only path
        return request.getfixturevalue("torch_classification_model")

    def _get_dummy_model(self, df: pd.DataFrame, request: pytest.FixtureRequest) -> BaseModel | None:
        """Return a dummy model that always predicts the same class.

        Only available for sklearn-based methods.  Returns ``None`` when
        no suitable dummy model can be constructed (torch-only methods).
        """
        if self.supports_sklearn:
            return _make_sklearn_dummy_model(df)
        return None

    # =====================================================================
    # Core tests — always run
    # =====================================================================

    def test_valid_initialization(self, dummy_classification_dataframe, request):
        """Explainer instantiates correctly with valid model and data."""
        model = self._get_model(dummy_classification_dataframe, request)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        assert isinstance(explainer, ClassifierExplainer)
        assert isinstance(explainer.data, PublicData)
        assert isinstance(explainer.model, BaseModel)

    def test_invalid_data_raises_configuration_error(self, dummy_classification_dataframe, request):
        """Non-PublicData object must raise ``ConfigurationError`` with ``param='data'``."""
        model = self._get_model(dummy_classification_dataframe, request)

        class InvalidData:
            pass

        with pytest.raises(ConfigurationError) as exc_info:
            self.explainer_class(
                model=model,
                data=InvalidData(),
                **self.explainer_kwargs,
            )

        assert exc_info.value.param == "data"

    def test_counterfactuals_exclude_target_column(self, dummy_classification_dataframe, request):
        """Output columns must be feature-only — no target column."""
        model = self._get_model(dummy_classification_dataframe, request)
        public_data, X = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        target_name = public_data.target_name
        feature_columns = set(public_data.column_names)

        results = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        if isinstance(results, Counterfactual):
            results = [results]

        for cf in results:
            assert target_name not in cf.original_instance.columns, (
                f"original_instance should not contain '{target_name}', "
                f"got: {list(cf.original_instance.columns)}"
            )
            assert target_name not in cf.counterfactuals.columns, (
                f"counterfactuals should not contain '{target_name}', "
                f"got: {list(cf.counterfactuals.columns)}"
            )
            assert set(cf.original_instance.columns) == feature_columns
            assert set(cf.counterfactuals.columns) == feature_columns

    def test_no_counterfactuals_found_raises_error(self, dummy_classification_dataframe, request):
        """``NoCounterfactualsFoundError`` when the algorithm cannot produce CFs.

        Uses ``DummyClassifier(strategy='most_frequent')`` so that every
        prediction is the same class, making counterfactuals impossible.
        Skipped for torch-only methods where a dummy model is not available.
        """
        dummy_model = self._get_dummy_model(dummy_classification_dataframe, request)
        if dummy_model is None:
            pytest.skip("No dummy model available for torch-only methods")

        public_data, X = _make_public_data(dummy_classification_dataframe)

        explainer = self.explainer_class(
            model=dummy_model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample, **self.generate_kwargs)

    def test_single_instance_returns_single_counterfactual(self, dummy_classification_dataframe, request):
        """A single-row input must return a ``Counterfactual``, not a list."""
        model = self._get_model(dummy_classification_dataframe, request)
        public_data, X = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        result = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        assert isinstance(result, Counterfactual)

    def test_multiple_instances_returns_list(self, dummy_classification_dataframe, request):
        """Multiple rows must return a ``list[Counterfactual]``."""
        model = self._get_model(dummy_classification_dataframe, request)
        public_data, X = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[:3]
        results = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        if isinstance(results, Counterfactual):
            results = [results]

        assert len(results) >= 1
        for cf in results:
            assert isinstance(cf, Counterfactual)

    # =====================================================================
    # Conditional tests — immutable features
    # =====================================================================

    @_skip_unless("supports_immutable_features")
    def test_immutable_features_unchanged(self, dummy_classification_dataframe, request):
        """Immutable columns in the counterfactual must equal the original values."""
        immutable_cols = ["feature1"]
        model = self._get_model(dummy_classification_dataframe, request)
        public_data, X = _make_public_data(
            dummy_classification_dataframe,
            immutable=immutable_cols,
        )

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        results = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        if isinstance(results, Counterfactual):
            results = [results]

        for cf in results:
            for col in immutable_cols:
                original_val = cf.original_instance[col].iloc[0]
                for _, row in cf.counterfactuals.iterrows():
                    assert row[col] == original_val, (
                        f"Immutable feature '{col}' was changed: "
                        f"original={original_val}, counterfactual={row[col]}"
                    )

    # =====================================================================
    # Conditional tests — feasible values
    # =====================================================================

    @_skip_unless("supports_feasible_values")
    def test_feasible_values_respected(self, dummy_classification_dataframe, request):
        """Continuous CF values must stay within the declared ``feasible_values`` bounds."""
        feasible = _build_feasible_values(dummy_classification_dataframe)
        model = self._get_model(dummy_classification_dataframe, request)
        public_data, X = _make_public_data(
            dummy_classification_dataframe,
            immutable=[],
            feasible_values=feasible,
        )

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        results = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        if isinstance(results, Counterfactual):
            results = [results]

        for cf in results:
            for col, bounds in feasible.items():
                if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(b, (int, float)) for b in bounds):
                    lo, hi = bounds
                    for val in cf.counterfactuals[col]:
                        assert lo <= val <= hi, (
                            f"Feature '{col}' value {val} is outside feasible range [{lo}, {hi}]"
                        )

    # =====================================================================
    # Conditional tests — categorical features
    # =====================================================================

    @_skip_unless("supports_categorical_features")
    def test_categorical_features_valid(self, dummy_classification_dataframe_with_categories, request):
        """Categorical CF values must belong to the allowed set from ``feasible_values``."""
        df = dummy_classification_dataframe_with_categories
        feasible = _build_feasible_values(df)
        model = self._get_model(df, request)
        public_data, X = _make_public_data(
            df,
            immutable=[],
            feasible_values=feasible,
        )

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        results = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        if isinstance(results, Counterfactual):
            results = [results]

        cat_cols = public_data.categorical_column_names or []
        for cf in results:
            for col in cat_cols:
                allowed = feasible.get(col, [])
                for val in cf.counterfactuals[col]:
                    assert val in allowed, (
                        f"Categorical feature '{col}' has value '{val}' "
                        f"not in allowed set: {allowed}"
                    )

    # =====================================================================
    # Conditional tests — encoding requirement
    # =====================================================================

    @_skip_unless("requires_encoded_data")
    def test_encoded_data_required(self, dummy_classification_dataframe_with_categories, request):
        """Must raise ``ConfigurationError`` when data contains unencoded categorical columns."""
        df = dummy_classification_dataframe_with_categories
        model = self._get_model(df, request)
        public_data, X = _make_public_data(df, immutable=[])

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        with pytest.raises(ConfigurationError):
            explainer.generate_counterfactuals(sample, **self.generate_kwargs)

    # =====================================================================
    # Conditional tests — model compatibility
    # =====================================================================

    @_skip_unless("supports_sklearn")
    def test_sklearn_model_compatibility(self, dummy_classification_dataframe):
        """Generation succeeds with a ``SklearnModel``."""
        model = _make_sklearn_model(dummy_classification_dataframe)
        public_data, X = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = self.explainer_class(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        result = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        if isinstance(result, Counterfactual):
            result = [result]

        assert len(result) >= 1
        for cf in result:
            assert isinstance(cf, Counterfactual)

    @_skip_unless("supports_torch")
    def test_torch_model_compatibility(self, dummy_classification_dataframe, torch_classification_model):
        """Generation runs against a ``TorchModel`` backend.

        The shared torch fixture is trained for only a few epochs on 8 rows,
        and stochastic init can produce a near-constant classifier on CI
        runners. Some methods (e.g. CLEAR) then have no decision boundary to
        work with and (correctly) raise ``NoCounterfactualsFoundError``. We
        accept that outcome here — the goal of this test is to verify that
        ``TorchModel`` is wired through end-to-end, not that CFs are always
        found.
        """
        df = dummy_classification_dataframe
        public_data, X = _make_public_data(df, immutable=[])

        explainer = self.explainer_class(
            model=torch_classification_model,
            data=public_data,
            **self.explainer_kwargs,
        )

        sample = X.iloc[[0]]
        try:
            result = explainer.generate_counterfactuals(sample, **self.generate_kwargs)
        except NoCounterfactualsFoundError:
            return

        if isinstance(result, Counterfactual):
            result = [result]

        assert len(result) >= 1
        for cf in result:
            assert isinstance(cf, Counterfactual)

    @_skip_unless("rejects_sklearn")
    def test_rejects_unsupported_sklearn_model(self, dummy_classification_dataframe):
        """``ConfigurationError`` when method actively rejects ``SklearnModel``."""
        model = _make_sklearn_model(dummy_classification_dataframe)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            self.explainer_class(
                model=model,
                data=public_data,
                **self.explainer_kwargs,
            )

        assert exc_info.value.param == "model"

    @_skip_unless("rejects_torch")
    def test_rejects_unsupported_torch_model(self, dummy_classification_dataframe, torch_classification_model):
        """``ConfigurationError`` when method actively rejects ``TorchModel``."""
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            self.explainer_class(
                model=torch_classification_model,
                data=public_data,
                **self.explainer_kwargs,
            )

        assert exc_info.value.param == "model"
