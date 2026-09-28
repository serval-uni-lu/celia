from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers.countergan import CounterGANClassifierExplainer
from tests.explainers.classifier_test_suite import (
    ClassifierExplainerTests,
    _make_data,
    make_countergan_classification_dataframe,
)


class TestCounterGANClassifier(ClassifierExplainerTests):
    explainer_class = CounterGANClassifierExplainer
    explainer_kwargs = {"desired_class": 1}
    generate_kwargs = {}

    @pytest.fixture
    def dummy_classification_dataframe(self):
        """Override: CounterGAN needs same-scale features near the boundary."""
        return make_countergan_classification_dataframe()

    supports_sklearn = False
    supports_torch = True
    rejects_sklearn = True
    rejects_torch = False
    supports_immutable_features = True
    supports_feasible_values = False
    supports_categorical_features = False
    requires_encoded_data = True

    _xfail_gan = pytest.mark.xfail(
        reason="CounterGAN training is stochastic; the GAN may not produce class-flipping CFs in test settings",
        raises=NoCounterfactualsFoundError,
        strict=False,
    )

    @_xfail_gan
    def test_counterfactuals_exclude_target_column(self, dummy_classification_dataframe, request):
        super().test_counterfactuals_exclude_target_column(dummy_classification_dataframe, request)

    @_xfail_gan
    def test_single_instance_returns_single_counterfactual(self, dummy_classification_dataframe, request):
        super().test_single_instance_returns_single_counterfactual(dummy_classification_dataframe, request)

    @_xfail_gan
    def test_multiple_instances_returns_list(self, dummy_classification_dataframe, request):
        super().test_multiple_instances_returns_list(dummy_classification_dataframe, request)

    @_xfail_gan
    def test_immutable_features_unchanged(self, dummy_classification_dataframe, request):
        super().test_immutable_features_unchanged(dummy_classification_dataframe, request)

    def test_encoded_data_required(self, dummy_classification_dataframe, torch_classification_model):
        """ConfigurationError when sample contains non-numeric columns.

        Overrides the mixin test because CounterGAN validates data at init time
        (the GAN needs numeric data to train), so we init with numeric data and
        pass a non-numeric sample instead.
        """
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = CounterGANClassifierExplainer(
            model=torch_classification_model,
            data=data,
            desired_class=1,
        )

        sample = X.iloc[[0]].copy()
        sample["feature3"] = "A"

        with pytest.raises(ConfigurationError):
            explainer.generate_counterfactuals(sample)

    def test_non_numeric_data_raises_error_at_init(
        self, dummy_classification_dataframe_with_categories, torch_classification_model
    ):
        """ConfigurationError(param='data') when data contains non-numeric columns at init."""
        data, _ = _make_data(dummy_classification_dataframe_with_categories, immutable=[])

        with pytest.raises(ConfigurationError) as exc_info:
            CounterGANClassifierExplainer(
                model=torch_classification_model,
                data=data,
                desired_class=1,
            )

        assert exc_info.value.param == "data"

    def test_desired_class_missing_raises_error(self, dummy_classification_dataframe, torch_classification_model):
        """ConfigurationError(param='desired_class') when desired_class is omitted."""
        data, _ = _make_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CounterGANClassifierExplainer(model=torch_classification_model, data=data)

        assert exc_info.value.param == "desired_class"

    def test_desired_class_invalid_value_raises_error(self, dummy_classification_dataframe, torch_classification_model):
        """ConfigurationError(param='desired_class') when desired_class is not in data targets."""
        data, _ = _make_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CounterGANClassifierExplainer(
                model=torch_classification_model,
                data=data,
                desired_class=999,
            )

        assert exc_info.value.param == "desired_class"

    def test_desired_class_valid_for_each_class(self, dummy_classification_dataframe, torch_classification_model):
        """Explainer initializes successfully for every valid class in targets."""
        data, _ = _make_data(dummy_classification_dataframe)
        unique_classes = np.unique(data.targets).tolist()

        for cls in unique_classes:
            explainer = CounterGANClassifierExplainer(
                model=torch_classification_model,
                data=data,
                desired_class=cls,
            )
            assert explainer.desired_class == cls
