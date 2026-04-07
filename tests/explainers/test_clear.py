from __future__ import annotations

import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier

from celia.errors import ConfigurationError
from celia.explainers.clear import CLEARClassifierExplainer
from celia.model import SklearnModel
from tests.explainers.classifier_test_suite import ClassifierExplainerTests, _make_public_data


def _fit_tree_model(df) -> SklearnModel:
    tree = DecisionTreeClassifier(random_state=0).fit(
        df.drop(columns=["target"]),
        df["target"],
    )
    return SklearnModel(tree)


class TestCLEARClassifier(ClassifierExplainerTests):
    explainer_class = CLEARClassifierExplainer
    explainer_kwargs = {"number_of_synthetic_samples": 200, "random_seed": 0}
    generate_kwargs = {}

    supports_sklearn = True
    supports_torch = True
    supports_immutable_features = False
    supports_feasible_values = False
    supports_categorical_features = False
    requires_encoded_data = True

    def test_encoded_data_required(self, dummy_classification_dataframe):
        """ConfigurationError when sample contains non-numeric columns.

        Overrides the mixin test because CLEAR validates non-numeric features at
        init time, so we init with numeric data and pass a non-numeric sample
        instead — exercising ``_validate_sample``.
        """
        model = _fit_tree_model(dummy_classification_dataframe)
        public_data, X = _make_public_data(dummy_classification_dataframe, immutable=[])

        explainer = CLEARClassifierExplainer(
            model=model,
            data=public_data,
            **self.explainer_kwargs,
        )

        bad_sample = X.iloc[[0]].copy()
        bad_sample["feature3"] = "A"

        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(bad_sample, **self.generate_kwargs)

        assert exc_info.value.param == "sample"

    def test_non_numeric_data_raises_error_at_init(
        self, dummy_classification_dataframe, dummy_classification_dataframe_with_categories
    ):
        """ConfigurationError(param='data') when data contains non-numeric columns at init."""
        model = _fit_tree_model(dummy_classification_dataframe)

        public_data, _ = _make_public_data(
            dummy_classification_dataframe_with_categories, immutable=[]
        )

        with pytest.raises(ConfigurationError) as exc_info:
            CLEARClassifierExplainer(model=model, data=public_data, **self.explainer_kwargs)

        assert exc_info.value.param == "data"

    def test_model_without_predict_proba_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='model') when sklearn model lacks predict_proba."""
        ridge = RidgeClassifier().fit(
            dummy_classification_dataframe.drop(columns=["target"]),
            dummy_classification_dataframe["target"],
        )
        model = SklearnModel(ridge)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CLEARClassifierExplainer(model=model, data=public_data, **self.explainer_kwargs)

        assert exc_info.value.param == "model"

    def test_invalid_num_classes_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='num_classes') when num_classes < 2."""
        model = _fit_tree_model(dummy_classification_dataframe)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CLEARClassifierExplainer(
                model=model,
                data=public_data,
                num_classes=1,
            )

        assert exc_info.value.param == "num_classes"

    def test_multiclass_without_class_labels_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='class_labels') when num_classes > 2 and class_labels missing."""
        model = _fit_tree_model(dummy_classification_dataframe)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CLEARClassifierExplainer(
                model=model,
                data=public_data,
                num_classes=3,
            )

        assert exc_info.value.param == "class_labels"

    def test_clear_kwargs_forwarded(self, dummy_classification_dataframe):
        """CLEAR-specific kwargs are stored on the explainer instance."""
        model = _fit_tree_model(dummy_classification_dataframe)
        public_data, _ = _make_public_data(dummy_classification_dataframe)

        explainer = CLEARClassifierExplainer(
            model=model,
            data=public_data,
            num_classes=2,
            number_of_synthetic_samples=123,
            random_seed=7,
            verbose=True,
            multi_class_focus="All",
        )

        assert explainer._number_of_synthetic_samples == 123
        assert explainer._random_seed == 7
        assert explainer._verbose is True
        assert explainer._multi_class_focus == "All"
        assert explainer._num_classes == 2
