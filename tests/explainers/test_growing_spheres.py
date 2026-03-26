from __future__ import annotations

from typing import Any

import pytest

from celia.errors import ConfigurationError
from celia.explainers.growing_spheres import GSGClassifierExplainer
from tests.explainers.classifier_test_suite import ClassifierExplainerTests, _make_public_data, _make_sklearn_model


class TestGSGClassifier(ClassifierExplainerTests):
    explainer_class = GSGClassifierExplainer
    explainer_kwargs: dict[str, Any] = {}
    generate_kwargs: dict[str, Any] = {}

    supports_sklearn = True
    supports_torch = True
    supports_immutable_features = True
    supports_feasible_values = False
    supports_categorical_features = False
    requires_encoded_data = False

    def test_rejects_non_numeric_data_at_init(self, dummy_classification_dataframe, dummy_classification_dataframe_with_categories):
        """Non-numeric columns in data must raise ``ConfigurationError`` at init."""
        # Use a model trained on numeric data (the categorical dataset can't train a tree)
        model = _make_sklearn_model(dummy_classification_dataframe)
        public_data, _ = _make_public_data(dummy_classification_dataframe_with_categories)

        with pytest.raises(ConfigurationError) as exc_info:
            GSGClassifierExplainer(model=model, data=public_data)

        assert exc_info.value.param == "data"

    def test_rejects_non_numeric_sample(self, dummy_classification_dataframe, dummy_classification_dataframe_with_categories):
        """Non-numeric columns in sample must raise ``ConfigurationError``."""
        model = _make_sklearn_model(dummy_classification_dataframe)
        public_data, X = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = GSGClassifierExplainer(model=model, data=public_data)

        # Build a sample with a string column matching the expected feature names
        import pandas as pd

        bad_sample = pd.DataFrame(
            {"feature1": [1.0], "feature2": [10.0], "feature3": ["A"]},
        )

        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(bad_sample)

        assert exc_info.value.param == "sample"

    def test_gsg_kwargs_forwarded(self, dummy_classification_dataframe):
        """Custom GSG algorithm kwargs reach the underlying GSG instance."""
        model = _make_sklearn_model(dummy_classification_dataframe)
        public_data, _ = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = GSGClassifierExplainer(
            model=model,
            data=public_data,
            n_samples=500,
            step_size=0.1,
            max_iterations=5,
        )

        assert explainer.explainer.n_samples == 500
        assert explainer.explainer.step_size == 0.1
        assert explainer.explainer.max_iterations == 5

    def test_binary_features_detected(self, dummy_classification_dataframe):
        """Binary columns (2 unique values) are detected and passed to GSG."""
        df = dummy_classification_dataframe.copy()
        df["binary_feature"] = [0, 1, 0, 1, 0, 1, 0, 1]

        model = _make_sklearn_model(df)
        public_data, _ = _make_public_data(df, immutable=[])

        explainer = GSGClassifierExplainer(
            model=model,
            data=public_data,
            max_iterations=5,
        )

        assert "binary_feature" in explainer.explainer.binary_features
