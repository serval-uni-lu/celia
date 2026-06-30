from __future__ import annotations

from celia.explainers.grace import GRACEClassifierExplainer
from tests.explainers.classifier_test_suite import ClassifierExplainerTests


class TestGRACEClassifier(ClassifierExplainerTests):
    explainer_class = GRACEClassifierExplainer
    explainer_kwargs = {}
    generate_kwargs = {}

    supports_sklearn = False
    supports_torch = True
    rejects_sklearn = True
    supports_immutable_features = False
    supports_feasible_values = True
    supports_categorical_features = False
    requires_encoded_data = False
