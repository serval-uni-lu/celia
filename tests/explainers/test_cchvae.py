from __future__ import annotations

import pytest

from celia.errors import ConfigurationError
from celia.explainers import ClassifierExplainer
from celia.explainers.cchvae import CCHVAEClassifierExplainer
from tests.explainers.classifier_test_suite import ClassifierExplainerTests, _make_public_data


class TestCCHVAEClassifier(ClassifierExplainerTests):
    explainer_class = CCHVAEClassifierExplainer
    explainer_kwargs = {"target_class": 1, "epochs": 10, "verbose": False}
    generate_kwargs = {}

    supports_sklearn = True
    supports_torch = True
    supports_immutable_features = True

    # ---- Method-specific tests ----

    def test_target_class_missing_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='target_class') when target_class is omitted."""
        model = self._get_model(dummy_classification_dataframe, None)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CCHVAEClassifierExplainer(model=model, data=public_data)

        assert exc_info.value.param == "target_class"

    def test_feature_types_unknown_column_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='feature_types') when override contains unknown columns."""
        model = self._get_model(dummy_classification_dataframe, None)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CCHVAEClassifierExplainer(
                model=model,
                data=public_data,
                target_class=1,
                feature_types={"nonexistent_column": "real"},
            )

        assert exc_info.value.param == "feature_types"

    def test_feature_types_invalid_type_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='feature_types') when override contains invalid type values."""
        model = self._get_model(dummy_classification_dataframe, None)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CCHVAEClassifierExplainer(
                model=model,
                data=public_data,
                target_class=1,
                feature_types={"feature1": "invalid_type"},
            )

        assert exc_info.value.param == "feature_types"

    def test_feature_types_valid_override_accepted(self, dummy_classification_dataframe):
        """Explainer initializes successfully when feature_types override is valid."""
        model = self._get_model(dummy_classification_dataframe, None)
        public_data, _ = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = CCHVAEClassifierExplainer(
            model=model,
            data=public_data,
            target_class=1,
            feature_types={"feature1": "real", "feature2": "real", "feature3": "real"},
            epochs=10,
        )

        assert isinstance(explainer, ClassifierExplainer)

    def test_cchvae_kwargs_stored(self, dummy_classification_dataframe):
        """C-CHVAE-specific kwargs are stored on the explainer instance."""
        model = self._get_model(dummy_classification_dataframe, None)
        public_data, _ = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = CCHVAEClassifierExplainer(
            model=model,
            data=public_data,
            target_class=1,
            latent_dim=4,
            intermediate_dim=10,
            categorical_latent_dim=5,
            learning_rate=0.01,
            epochs=10,
            batch_size=64,
            device="cpu",
            random_state=42,
            verbose=True,
        )

        assert explainer._target_class == 1
        assert explainer._latent_dim == 4
        assert explainer._intermediate_dim == 10
        assert explainer._categorical_latent_dim == 5
        assert explainer._learning_rate == 0.01
        assert explainer._epochs == 10
        assert explainer._batch_size == 64
        assert explainer._device == "cpu"
        assert explainer._random_state == 42
        assert explainer._verbose is True
