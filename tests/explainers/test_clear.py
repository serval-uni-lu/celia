from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier

from celia.data import Data
from celia.errors import ConfigurationError
from celia.explainers.clear import CLEARClassifierExplainer
from celia.model import SklearnModel
from tests.explainers.classifier_test_suite import ClassifierExplainerTests, _make_data


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
        data, X = _make_data(dummy_classification_dataframe, immutable=[])

        explainer = CLEARClassifierExplainer(
            model=model,
            data=data,
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

        data, _ = _make_data(
            dummy_classification_dataframe_with_categories, immutable=[]
        )

        with pytest.raises(ConfigurationError) as exc_info:
            CLEARClassifierExplainer(model=model, data=data, **self.explainer_kwargs)

        assert exc_info.value.param == "data"

    def test_model_without_predict_proba_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='model') when sklearn model lacks predict_proba."""
        ridge = RidgeClassifier().fit(
            dummy_classification_dataframe.drop(columns=["target"]),
            dummy_classification_dataframe["target"],
        )
        model = SklearnModel(ridge)
        data, _ = _make_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CLEARClassifierExplainer(model=model, data=data, **self.explainer_kwargs)

        assert exc_info.value.param == "model"

    def test_invalid_num_classes_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='num_classes') when num_classes < 2."""
        model = _fit_tree_model(dummy_classification_dataframe)
        data, _ = _make_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CLEARClassifierExplainer(
                model=model,
                data=data,
                num_classes=1,
            )

        assert exc_info.value.param == "num_classes"

    def test_multiclass_without_class_labels_raises_error(self, dummy_classification_dataframe):
        """ConfigurationError(param='class_labels') when num_classes > 2 and class_labels missing."""
        model = _fit_tree_model(dummy_classification_dataframe)
        data, _ = _make_data(dummy_classification_dataframe)

        with pytest.raises(ConfigurationError) as exc_info:
            CLEARClassifierExplainer(
                model=model,
                data=data,
                num_classes=3,
            )

        assert exc_info.value.param == "class_labels"

    def test_clear_kwargs_forwarded(self, dummy_classification_dataframe):
        """CLEAR-specific kwargs are stored on the explainer instance."""
        model = _fit_tree_model(dummy_classification_dataframe)
        data, _ = _make_data(dummy_classification_dataframe)

        explainer = CLEARClassifierExplainer(
            model=model,
            data=data,
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


class TestCLEARInferOhePrefixes:
    """Tests for CLEARClassifierExplainer._infer_ohe_prefixes."""

    @staticmethod
    def _data_with_categoricals(cat_cols: list[str]) -> Data:
        n = 3
        all_cols = cat_cols if cat_cols else ["_dummy"]
        df = pd.DataFrame({col: [0.0] * n for col in all_cols})
        return Data(
            data=df,
            targets=pd.Series([0, 1, 0], name="target"),
            target_name="target",
            column_names=df.columns.tolist(),
            continuous_column_names=[c for c in df.columns if c not in cat_cols],
            categorical_column_names=cat_cols,
            immutable_column_names=[],
            feasible_values=None,
        )

    def test_no_categorical_columns(self):
        data = self._data_with_categoricals([])
        assert CLEARClassifierExplainer._infer_ohe_prefixes(data) == []

    def test_multi_column_ohe_group(self):
        data = self._data_with_categoricals(["color_red", "color_blue", "color_green"])
        prefixes = CLEARClassifierExplainer._infer_ohe_prefixes(data)
        assert prefixes == ["color"]

    def test_single_column_group_with_underscore(self):
        data = self._data_with_categoricals(["size_large"])
        prefixes = CLEARClassifierExplainer._infer_ohe_prefixes(data)
        assert prefixes == ["size"]

    def test_single_column_group_without_underscore(self):
        data = self._data_with_categoricals(["standalone"])
        prefixes = CLEARClassifierExplainer._infer_ohe_prefixes(data)
        assert prefixes == ["standalone"]

    def test_mixed_groups(self):
        data = self._data_with_categoricals([
            "color_red", "color_blue",
            "shape_circle", "shape_square", "shape_triangle",
            "solo",
        ])
        prefixes = CLEARClassifierExplainer._infer_ohe_prefixes(data)
        assert "color" in prefixes
        assert "shape" in prefixes
        assert "solo" in prefixes
