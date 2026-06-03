from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celia.errors import ConfigurationError
from celia.explainers.counternet import CounterNetClassifierExplainer 
from tests.explainers.classifier_test_suite import ClassifierExplainerTests, _make_public_data


class TestCounterNetClassifier(ClassifierExplainerTests):
    explainer_class = CounterNetClassifierExplainer
    
    # We use 1 epoch and a small batch size so the test suite runs instantly
    explainer_kwargs = {"epochs": 1, "batch_size": 16} 
    generate_kwargs = {}

    # CounterNet Capabilities
    supports_sklearn = True
    supports_torch = True
    rejects_sklearn = False
    rejects_torch = False
    supports_immutable_features = False # CounterNet does not natively freeze features without loss modifications
    supports_feasible_values = False
    supports_categorical_features = False
    requires_encoded_data = True

    def test_encoded_data_required(self, model_trained_classifier_stratified, dummy_classification_dataframe):
        """ConfigurationError when sample contains non-numeric columns during generation.

        CounterNet requires the inputs to be purely numerical (including Label/CatBoost encoded categories) 
        before being converted to PyTorch tensors.
        """
        public_data, X = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = CounterNetClassifierExplainer(
            model=model_trained_classifier_stratified,
            data=public_data,
            epochs=1
        )

        sample = X.iloc[[0]].copy()
        sample["feature3"] = "A"  # Inject a raw string to trigger the validation

        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample)

        assert exc_info.value.source == "CounterNetExplainer._validate_sample"

    def test_non_numeric_data_raises_error_at_init(
        self, dummy_classification_dataframe_with_categories, torch_classification_model
    ):
        """ConfigurationError(param='data') when data contains non-numeric columns at init."""
        public_data, _ = _make_public_data(dummy_classification_dataframe_with_categories, immutable=[])

        with pytest.raises(ConfigurationError) as exc_info:
            CounterGANClassifierExplainer(
                model=torch_classification_model,
                data=public_data,
                desired_class=1,
            )

        assert exc_info.value.param == "data"

    def test_public_data_required_raises_error(self, torch_classification_model):
        """ConfigurationError(param='data') when initialized without PublicData."""
        from celia.data._base import BaseData
        
        # Create a dummy data object that is NOT an instance of PublicData
        class DummyData(BaseData):
            pass
        dummy_data = DummyData()

        with pytest.raises(ConfigurationError) as exc_info:
            CounterNetClassifierExplainer(
                model=torch_classification_model,
                data=dummy_data,
            )

        assert exc_info.value.param == "data"
        assert exc_info.value.source == "CounterNetExplainer.__init__"
