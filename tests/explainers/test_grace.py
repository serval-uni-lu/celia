from __future__ import annotations

import numpy as np
import pytest

from celia.errors import ConfigurationError
from celia.explainers.grace import GRACEClassifierExplainer
from tests.explainers.classifier_test_suite import (
    ClassifierExplainerTests,
    make_learnable_classification_dataframe,
)


class TestGRACEClassifier(ClassifierExplainerTests):
    explainer_class = GRACEClassifierExplainer
    explainer_kwargs = {}
    generate_kwargs = {}

    @pytest.fixture
    def dummy_classification_dataframe(self):
        """Override: GRACE needs a dataset with a real decision boundary to cross."""
        return make_learnable_classification_dataframe()

    supports_sklearn = False
    supports_torch = True
    rejects_sklearn = True
    supports_immutable_features = False
    supports_feasible_values = True
    supports_categorical_features = False
    requires_encoded_data = False


class TestGRACEFeatureSelector:
    """Tests for the custom feature_selector code path in GRACE._naive_gradient_attack."""

    @pytest.fixture
    def grace_instance(self, dummy_classification_dataframe, torch_classification_model):
        from celia.explainers.grace.grace import GRACE

        return GRACE(torch_classification_model)

    def test_feature_selector_with_select_method(
        self, grace_instance, dummy_classification_dataframe
    ):
        """A feature_selector with a .select() method is used for feature selection."""
        torch = pytest.importorskip("torch")

        X = dummy_classification_dataframe.drop(columns=["target"])
        sample = torch.tensor(X.iloc[[0]].to_numpy(), dtype=torch.float32)

        class MySelector:
            def select(self, indices, num_features):
                return list(range(num_features))

        _, _, cf_array, selected, _ = grace_instance._naive_gradient_attack(
            original_instance=sample,
            num_features=2,
            feature_selector=MySelector(),
        )

        assert len(selected) == 2

    def test_feature_selector_without_select_raises(
        self, grace_instance, dummy_classification_dataframe
    ):
        """A feature_selector lacking .select() raises ConfigurationError."""
        torch = pytest.importorskip("torch")

        X = dummy_classification_dataframe.drop(columns=["target"])
        sample = torch.tensor(X.iloc[[0]].to_numpy(), dtype=torch.float32)

        class BadSelector:
            pass

        with pytest.raises(ConfigurationError) as exc_info:
            grace_instance._naive_gradient_attack(
                original_instance=sample,
                num_features=2,
                feature_selector=BadSelector(),
            )

        assert exc_info.value.param == "feature_selector"


class TestGRACEMaintainDomain:
    """Tests for integer rounding in GRACE._maintain_domain."""

    def test_integer_features_are_rounded(self):
        from celia.explainers.grace.grace import GRACE

        cf_array = np.array([[1.7, 2.3, 3.9]], dtype=np.float32)
        selected = [0, 2]
        alphas = [1.0, 1.0, 1.0]
        is_integer = [True, False, True]

        result = GRACE._maintain_domain(
            cf_array,
            selected_features=selected,
            feature_alphas=alphas,
            feature_is_integer=is_integer,
        )

        assert result[0, 0] == pytest.approx(2.0)
        assert result[0, 1] == pytest.approx(2.3)
        assert result[0, 2] == pytest.approx(4.0)

    def test_non_finite_alpha_skips_rounding(self):
        from celia.explainers.grace.grace import GRACE

        cf_array = np.array([[1.7]], dtype=np.float32)
        result = GRACE._maintain_domain(
            cf_array,
            selected_features=[0],
            feature_alphas=[float("inf")],
            feature_is_integer=[True],
        )

        assert result[0, 0] == pytest.approx(1.7)
