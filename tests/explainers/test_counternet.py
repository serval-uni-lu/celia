from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celia.errors import ConfigurationError
# Update this import path to match where you saved the explainer
from celia.explainers.counternet import CounterNetClassifierExplainer 
from tests.explainers.classifier_test_suite import ClassifierExplainerTests, _make_public_data


class TestCounterNetClassifier(ClassifierExplainerTests):
    explainer_class = CounterNetClassifierExplainer
    
    # We use 1 epoch and a small batch size so the test suite runs instantly
    explainer_kwargs = {"epochs": 1, "batch_size": 16} 
    generate_kwargs = {}

    # CounterNet Capabilities
    supports_sklearn = False
    supports_torch = True
    rejects_sklearn = True
    rejects_torch = False
    supports_immutable_features = False # CounterNet does not natively freeze features without loss modifications
    supports_feasible_values = False
    supports_categorical_features = True # Thanks to our CategoricalNormalizer!
    requires_encoded_data = True

    def test_encoded_data_required(self, dummy_classification_dataframe, torch_classification_model):
        """ConfigurationError when sample contains non-numeric columns during generation.

        CounterNet requires the inputs to be purely numerical (including Label/CatBoost encoded categories) 
        before being converted to PyTorch tensors.
        """
        public_data, X = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = CounterNetClassifierExplainer(
            model=torch_classification_model,
            data=public_data,
            epochs=1
        )

        sample = X.iloc[[0]].copy()
        sample["feature3"] = "A"  # Inject a raw string to trigger the validation

        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample)

        assert exc_info.value.source == "CounterNetExplainer._validate_sample"

    def test_pytorch_model_required_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='model') when initialized with a non-TorchModel."""
        from celia.model import BaseModel # Or whatever dummy mock you use for non-torch models
        
        class DummyNonTorchModel(BaseModel):
            def predict(self, X): return np.zeros(len(X))
            def predict_proba(self, X): return np.zeros((len(X), 2))
            
        dummy_model = DummyNonTorchModel()
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CounterNetClassifierExplainer(
                model=dummy_model,
                data=public_data,
            )

        assert exc_info.value.param == "model"
        assert "expected" in exc_info.value.config

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
