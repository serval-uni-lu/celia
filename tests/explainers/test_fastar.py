from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from celia.counterfactuals import Counterfactual
from celia.data import Data
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers.fastar import FastARClassifierExplainer
from tests.explainers.classifier_test_suite import ClassifierExplainerTests, _make_data, _make_sklearn_model


class TestFastARClassifier(ClassifierExplainerTests):
    explainer_class = FastARClassifierExplainer
    explainer_kwargs: dict[str, Any] = {"total_timesteps": 100}
    generate_kwargs: dict[str, Any] = {}

    supports_sklearn = True
    supports_torch = True
    supports_immutable_features = True
    supports_feasible_values = False
    supports_categorical_features = False
    requires_encoded_data = True

    # ------------------------------------------------------------------
    # Override core tests that expect CFs — with tiny timesteps the agent
    # is untrained, so CFs are not guaranteed.  We verify the wrapper runs
    # end-to-end without crashing.
    # ------------------------------------------------------------------

    def test_counterfactuals_exclude_target_column(self, dummy_classification_dataframe, request):
        model = self._get_model(dummy_classification_dataframe, request)
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = self.explainer_class(model=model, data=data, **self.explainer_kwargs)
        sample = X.iloc[[0]]

        try:
            results = explainer.generate_counterfactuals(sample, **self.generate_kwargs)
        except NoCounterfactualsFoundError:
            return

        if isinstance(results, Counterfactual):
            results = [results]

        feature_columns = set(data.column_names)
        for cf in results:
            assert data.target_name not in cf.original_instance.columns
            assert data.target_name not in cf.counterfactuals.columns
            assert set(cf.original_instance.columns) == feature_columns
            assert set(cf.counterfactuals.columns) == feature_columns

    def test_single_instance_returns_single_counterfactual(self, dummy_classification_dataframe, request):
        model = self._get_model(dummy_classification_dataframe, request)
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = self.explainer_class(model=model, data=data, **self.explainer_kwargs)
        sample = X.iloc[[0]]

        try:
            result = explainer.generate_counterfactuals(sample, **self.generate_kwargs)
        except NoCounterfactualsFoundError:
            return

        assert isinstance(result, Counterfactual)

    def test_multiple_instances_returns_list(self, dummy_classification_dataframe, request):
        model = self._get_model(dummy_classification_dataframe, request)
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = self.explainer_class(model=model, data=data, **self.explainer_kwargs)
        sample = X.iloc[:3]

        try:
            results = explainer.generate_counterfactuals(sample, **self.generate_kwargs)
        except NoCounterfactualsFoundError:
            return

        if isinstance(results, Counterfactual):
            results = [results]

        assert len(results) >= 1
        for cf in results:
            assert isinstance(cf, Counterfactual)

    def test_no_counterfactuals_found_raises_error(self, dummy_classification_dataframe, request):
        """All features immutable → agent cannot act → no CFs found."""
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

    def test_immutable_features_unchanged(self, dummy_classification_dataframe, request):
        """Override: tolerate NoCounterfactualsFoundError with untrained agent."""
        immutable_cols = ["feature1"]
        model = self._get_model(dummy_classification_dataframe, request)
        data, X = _make_data(
            dummy_classification_dataframe,
            immutable=immutable_cols,
        )

        explainer = self.explainer_class(model=model, data=data, **self.explainer_kwargs)
        sample = X.iloc[[0]]

        try:
            results = explainer.generate_counterfactuals(sample, **self.generate_kwargs)
        except NoCounterfactualsFoundError:
            return

        if isinstance(results, Counterfactual):
            results = [results]

        for cf in results:
            for col in immutable_cols:
                original_val = cf.original_instance[col].iloc[0]
                for _, row in cf.counterfactuals.iterrows():
                    assert row[col] == original_val

    # ------------------------------------------------------------------
    # Override mixin tests that don't apply to FastAR
    # ------------------------------------------------------------------

    def test_sklearn_model_compatibility(self, dummy_classification_dataframe):
        """Override: tolerate NoCounterfactualsFoundError with untrained agent."""
        model = _make_sklearn_model(dummy_classification_dataframe)
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = self.explainer_class(model=model, data=data, **self.explainer_kwargs)
        sample = X.iloc[[0]]

        try:
            result = explainer.generate_counterfactuals(sample, **self.generate_kwargs)
        except NoCounterfactualsFoundError:
            return

        if isinstance(result, Counterfactual):
            result = [result]

        assert len(result) >= 1

    def test_encoded_data_required(self, dummy_classification_dataframe_with_categories, request):
        """Override: FastAR validates numeric data at generate time, not init.

        The categorical fixture can't even train a DecisionTreeClassifier, so
        we build the explainer on numeric data and inject a non-numeric sample.
        """
        model = self._get_model(
            request.getfixturevalue("dummy_classification_dataframe"), request
        )
        data, X = _make_data(
            request.getfixturevalue("dummy_classification_dataframe"), immutable=[]
        )

        explainer = self.explainer_class(model=model, data=data, **self.explainer_kwargs)

        sample = X.iloc[[0]].copy()
        sample["feature1"] = "A"

        with pytest.raises(ConfigurationError):
            explainer.generate_counterfactuals(sample, **self.generate_kwargs)

    # ------------------------------------------------------------------
    # Method-specific tests
    # ------------------------------------------------------------------

    def test_invalid_target_class_type(self, dummy_classification_dataframe, request):
        model = self._get_model(dummy_classification_dataframe, request)
        data, _ = _make_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            FastARClassifierExplainer(
                model=model,
                data=data,
                target_class="not_an_int",
                total_timesteps=100,
            )

        assert exc_info.value.param == "target_class"

    def test_invalid_total_timesteps(self, dummy_classification_dataframe, request):
        model = self._get_model(dummy_classification_dataframe, request)
        data, _ = _make_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            FastARClassifierExplainer(
                model=model,
                data=data,
                total_timesteps=0,
            )

        assert exc_info.value.param == "total_timesteps"

    def test_invalid_policy_path(self, dummy_classification_dataframe, request):
        model = self._get_model(dummy_classification_dataframe, request)
        data, _ = _make_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            FastARClassifierExplainer(
                model=model,
                data=data,
                policy_path="/nonexistent/path/policy.zip",
                total_timesteps=100,
            )

        assert exc_info.value.param == "policy_path"

    def test_save_policy(self, dummy_classification_dataframe, request, tmp_path):
        model = self._get_model(dummy_classification_dataframe, request)
        data, _ = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = FastARClassifierExplainer(
            model=model,
            data=data,
            total_timesteps=100,
        )

        policy_file = tmp_path / "fastar_policy"
        explainer.save_policy(policy_file)

        assert Path(f"{policy_file}.zip").exists()

    def test_load_policy_from_path(self, dummy_classification_dataframe, request, tmp_path):
        model = self._get_model(dummy_classification_dataframe, request)
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = FastARClassifierExplainer(
            model=model,
            data=data,
            total_timesteps=100,
        )

        policy_file = tmp_path / "fastar_policy"
        explainer.save_policy(policy_file)

        loaded_explainer = FastARClassifierExplainer(
            model=model,
            data=data,
            policy_path=f"{policy_file}.zip",
            total_timesteps=100,
        )

        sample = X.iloc[[0]]
        try:
            result = loaded_explainer.generate_counterfactuals(sample)
        except NoCounterfactualsFoundError:
            pass

    def test_non_numeric_data_raises_at_generate(self, dummy_classification_dataframe, request):
        """Non-numeric sample must raise ConfigurationError at generate time."""
        model = self._get_model(dummy_classification_dataframe, request)
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = FastARClassifierExplainer(
            model=model,
            data=data,
            total_timesteps=100,
        )

        sample = X.iloc[[0]].copy()
        sample["feature1"] = "not_a_number"

        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample)

        assert exc_info.value.param == "sample"

    def test_feature_spec_built_from_data(self, dummy_classification_dataframe, request):
        """Verify the internal FeatureSpec matches Data attributes."""
        model = self._get_model(dummy_classification_dataframe, request)
        data, _ = _make_data(dummy_classification_dataframe, immutable=["feature1"])

        explainer = FastARClassifierExplainer(
            model=model,
            data=data,
            total_timesteps=100,
        )

        spec = explainer.explainer.feature_spec
        assert list(spec.columns) == data.column_names
        assert list(spec.immutable) == (data.immutable_column_names or [])
        assert list(spec.continuous) == (data.continuous_column_names or [])
