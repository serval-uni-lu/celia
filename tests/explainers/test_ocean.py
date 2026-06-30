from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from celia.counterfactuals import Counterfactual
from celia.data import Data
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers.ocean import OCEANClassifierExplainer
from celia.model import BaseModel, SklearnModel
from tests.explainers.classifier_test_suite import ClassifierExplainerTests, _make_data


class TestOCEANMissingDependencies:
    """Verify that missing oceanpy / gurobipy produce a clear ConfigurationError."""

    def _make_minimal_args(self, dummy_classification_dataframe):
        df = dummy_classification_dataframe
        X = df.drop(columns=["target"])
        y = df["target"]
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)
        model = SklearnModel(rf)
        data, _ = _make_data(df, immutable=[])
        return model, data

    def test_missing_both_raises_configuration_error(self, dummy_classification_dataframe, monkeypatch):
        """Both oceanpy and gurobipy absent → ConfigurationError listing both."""
        monkeypatch.setitem(sys.modules, "ocean", None)
        monkeypatch.setitem(sys.modules, "gurobipy", None)

        model, data = self._make_minimal_args(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            OCEANClassifierExplainer(model=model, data=data)

        err = exc_info.value
        assert err.param == "ocean"
        assert "OCEANClassifierExplainer" in err.message
        assert "pip install celia[ocean]" in err.hint
        assert set(err.config["required_packages"]) == {"oceanpy", "gurobipy"}
        assert "OCEANClassifierExplainer.__init__" in err.source

    def test_missing_only_ocean_raises_configuration_error(self, dummy_classification_dataframe, monkeypatch):
        """Only oceanpy absent → ConfigurationError mentioning oceanpy."""
        monkeypatch.setitem(sys.modules, "ocean", None)

        model, data = self._make_minimal_args(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            OCEANClassifierExplainer(model=model, data=data)

        err = exc_info.value
        assert err.param == "ocean"
        assert "oceanpy" in err.config["required_packages"]
        assert "gurobipy" not in err.config["required_packages"]

    def test_missing_only_gurobipy_raises_configuration_error(self, dummy_classification_dataframe, monkeypatch):
        """Only gurobipy absent → ConfigurationError mentioning gurobipy."""
        from unittest.mock import MagicMock

        monkeypatch.setitem(sys.modules, "ocean", MagicMock())
        monkeypatch.setitem(sys.modules, "gurobipy", None)

        model, data = self._make_minimal_args(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            OCEANClassifierExplainer(model=model, data=data)

        err = exc_info.value
        assert err.param == "ocean"
        assert "gurobipy" in err.config["required_packages"]
        assert "oceanpy" not in err.config["required_packages"]


class TestOCEANClassifier(ClassifierExplainerTests):
    """Test suite for ``OCEANClassifierExplainer``.

    OCEAN requires a tree-ensemble classifier (``RandomForestClassifier``,
    ``XGBClassifier``, or ``AdaBoostClassifier``), so several base helpers
    and tests are overridden to use ``RandomForestClassifier``.
    """

    explainer_class = OCEANClassifierExplainer
    explainer_kwargs: dict[str, Any] = {}
    generate_kwargs: dict[str, Any] = {"norm": 1, "max_time": 30, "verbose": False}

    # Capability flags
    supports_sklearn = True
    supports_torch = False
    rejects_torch = True
    supports_immutable_features = True
    supports_feasible_values = True
    supports_categorical_features = False
    requires_encoded_data = False

    # -----------------------------------------------------------------
    # Model helper overrides (OCEAN needs tree ensembles)
    # -----------------------------------------------------------------

    def _get_model(self, df: pd.DataFrame, request: pytest.FixtureRequest) -> BaseModel:
        """Use ``RandomForestClassifier`` instead of ``DecisionTreeClassifier``."""
        X = df.drop(columns=["target"])
        y = df["target"]
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        rf.fit(X, y)
        return SklearnModel(rf)

    def _get_dummy_model(self, df: pd.DataFrame, request: pytest.FixtureRequest) -> BaseModel | None:
        """Return ``None`` — ``DummyClassifier`` is not a tree ensemble."""
        return None

    # -----------------------------------------------------------------
    # Overridden base tests
    # -----------------------------------------------------------------

    def test_no_counterfactuals_found_raises_error(self, dummy_classification_dataframe, request):
        """Override: make all features immutable to force infeasibility.

        OCEAN requires a tree ensemble, so ``DummyClassifier`` cannot be used.
        Instead, we fix every feature as immutable so the MIP solver cannot
        find a counterfactual with a different prediction.
        """
        model = self._get_model(dummy_classification_dataframe, request)
        all_features = dummy_classification_dataframe.drop(columns=["target"]).columns.tolist()
        data, X = _make_data(
            dummy_classification_dataframe,
            immutable=all_features,
        )

        explainer = self.explainer_class(model=model, data=data, **self.explainer_kwargs)
        sample = X.iloc[[0]]

        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample, **self.generate_kwargs)

    def test_sklearn_model_compatibility(self, dummy_classification_dataframe):
        """Override: use ``RandomForestClassifier`` (tree ensemble required)."""
        X = dummy_classification_dataframe.drop(columns=["target"])
        y = dummy_classification_dataframe["target"]
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        rf.fit(X, y)
        model = SklearnModel(rf)

        data, X_features = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = OCEANClassifierExplainer(model=model, data=data)
        sample = X_features.iloc[[0]]
        result = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        if isinstance(result, Counterfactual):
            result = [result]

        assert len(result) >= 1
        for cf in result:
            assert isinstance(cf, Counterfactual)

    # -----------------------------------------------------------------
    # OCEAN-specific tests
    # -----------------------------------------------------------------

    def test_rejects_non_numeric_data_at_init(
        self,
        dummy_classification_dataframe_with_categories,
    ):
        """Non-numeric data must raise ``ConfigurationError`` at init."""
        df = dummy_classification_dataframe_with_categories
        # Train on numeric columns only so the model is valid
        numeric_cols = df.select_dtypes(include=["number"]).columns.difference(["target"])
        X_numeric = df[numeric_cols]
        y = df["target"]
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        rf.fit(X_numeric, y)
        model = SklearnModel(rf)

        # Data includes the non-numeric column
        data, _ = _make_data(df, immutable=[])

        with pytest.raises(ConfigurationError) as exc_info:
            OCEANClassifierExplainer(model=model, data=data)

        assert exc_info.value.param == "data"

    def test_target_class_auto_flip_binary(self, dummy_classification_dataframe, request):
        """For binary classification, target class is auto-determined as the opposite."""
        model = self._get_model(dummy_classification_dataframe, request)
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = OCEANClassifierExplainer(model=model, data=data)
        sample = X.iloc[[0]]
        result = explainer.generate_counterfactuals(sample, **self.generate_kwargs)

        assert isinstance(result, Counterfactual)
        assert result.original_prediction != result.counterfactual_prediction

    def test_explicit_target_class(self, dummy_classification_dataframe, request):
        """User can explicitly provide ``target_class``."""
        model = self._get_model(dummy_classification_dataframe, request)
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = OCEANClassifierExplainer(model=model, data=data)
        sample = X.iloc[[0]]
        result = explainer.generate_counterfactuals(
            sample,
            target_class=1,
            **self.generate_kwargs,
        )

        assert isinstance(result, Counterfactual)
        assert result.counterfactual_prediction == 1
