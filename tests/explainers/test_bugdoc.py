import pandas as pd
import pytest

from celia.counterfactuals import Counterfactual
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers.bugdoc import BugDocClassifierExplainer, BugDocRegressorExplainer
from celia.explainers.bugdoc import bugdoc as bugdoc_module


class DummyStackedShortcut:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, entry_point, input_dict, historical_runs=None):
        return [["feature2 >= 20"]]


class DummyDebuggingDecisionTrees:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, entry_point, input_dict, historical_runs=None):
        return None, None, None


class DummyStackedShortcutEmpty(DummyStackedShortcut):
    def run(self, entry_point, input_dict, historical_runs=None):
        return []


class DummyStackedShortcutClauseOperators(DummyStackedShortcut):
    def run(self, entry_point, input_dict, historical_runs=None):
        return [[
            "feature4 == B",
            "feature4 != A",
            "feature3 < 200",
            "feature5 >= 1",
        ]]


class DummyDebuggingDecisionTreesWithTree(DummyDebuggingDecisionTrees):
    def run(self, entry_point, input_dict, historical_runs=None):
        return None, object(), None


@pytest.fixture
def regression_sample(celia_data_without_encoded_data) -> pd.DataFrame:
    return celia_data_without_encoded_data.data.iloc[[0]]


@pytest.fixture
def classification_sample(celia_data_classification) -> pd.DataFrame:
    return celia_data_classification.data.iloc[[0]]


class TestBugDocExplainer:
    def test_regressor_explainer_requires_data(self, model_trained_without_encoded_data):
        with pytest.raises(ConfigurationError, match="data must be an instance of Data"):
            BugDocRegressorExplainer(model=model_trained_without_encoded_data, data=object())

    def test_classifier_explainer_requires_data(self, model_trained_classifier):
        with pytest.raises(ConfigurationError, match="data must be an instance of Data"):
            BugDocClassifierExplainer(model=model_trained_classifier, data=object())

    def test_regressor_generate_counterfactuals_uses_stackedshortcut(
        self,
        monkeypatch,
        model_trained_without_encoded_data,
        celia_data_without_encoded_data,
        regression_sample,
    ):
        explainer = BugDocRegressorExplainer(
            model=model_trained_without_encoded_data,
            data=celia_data_without_encoded_data,
        )

        monkeypatch.setattr(
            bugdoc_module,
            "StackedShortcut",
            DummyStackedShortcut,
        )

        result = explainer.generate_counterfactuals(sample=regression_sample, target_range=(0.0, 0.2))

        assert isinstance(result, Counterfactual)
        assert result.original_prediction == pytest.approx(0.3)
        assert result.counterfactuals.iloc[0]["feature2"] == 20.0

    def test_regressor_generate_counterfactuals_parses_all_clause_operators(
        self,
        monkeypatch,
        model_trained_without_encoded_data,
        celia_data_without_encoded_data,
        regression_sample,
    ):
        explainer = BugDocRegressorExplainer(
            model=model_trained_without_encoded_data,
            data=celia_data_without_encoded_data,
        )

        monkeypatch.setattr(
            bugdoc_module,
            "StackedShortcut",
            DummyStackedShortcutClauseOperators,
        )

        result = explainer.generate_counterfactuals(sample=regression_sample, target_range=(0.0, 0.2))

        assert isinstance(result, Counterfactual)
        assert result.counterfactuals.iloc[0]["feature4"] == "B"
        assert result.counterfactuals.iloc[0]["feature3"] == 100.0
        assert result.counterfactuals.iloc[0]["feature5"] == True

    def test_regressor_generate_counterfactuals_uses_debug_tree_fallback(
        self,
        monkeypatch,
        model_trained_without_encoded_data,
        celia_data_without_encoded_data,
        regression_sample,
    ):
        explainer = BugDocRegressorExplainer(
            model=model_trained_without_encoded_data,
            data=celia_data_without_encoded_data,
        )

        monkeypatch.setattr(
            bugdoc_module,
            "StackedShortcut",
            DummyStackedShortcutEmpty,
        )
        monkeypatch.setattr(
            bugdoc_module,
            "DebuggingDecisionTrees",
            DummyDebuggingDecisionTreesWithTree,
        )
        monkeypatch.setattr(
            bugdoc_module._tree,
            "get_depth",
            lambda *_: 1,
        )
        monkeypatch.setattr(
            bugdoc_module,
            "prune_tree",
            lambda t, keys: [["feature4 != A"]],
        )

        result = explainer.generate_counterfactuals(sample=regression_sample, target_range=(0.0, 0.2))

        assert isinstance(result, Counterfactual)
        assert result.counterfactuals.iloc[0]["feature4"] == "B"

    def test_regressor_generate_counterfactuals_raises_when_no_counterfactuals(
        self,
        monkeypatch,
        model_trained_without_encoded_data,
        celia_data_without_encoded_data,
        regression_sample,
    ):
        explainer = BugDocRegressorExplainer(
            model=model_trained_without_encoded_data,
            data=celia_data_without_encoded_data,
        )

        monkeypatch.setattr(
            bugdoc_module,
            "StackedShortcut",
            DummyStackedShortcutEmpty,
        )
        monkeypatch.setattr(
            bugdoc_module,
            "DebuggingDecisionTrees",
            DummyDebuggingDecisionTrees,
        )
        monkeypatch.setattr(
            bugdoc_module._tree,
            "get_depth",
            lambda *_: 0,
        )

        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample=regression_sample, target_range=(0.0, 0.2))

    def test_classifier_generate_counterfactuals_uses_stackedshortcut(
        self,
        monkeypatch,
        model_trained_classifier,
        celia_data_classification,
        classification_sample,
    ):
        explainer = BugDocClassifierExplainer(
            model=model_trained_classifier,
            data=celia_data_classification,
        )

        monkeypatch.setattr(
            bugdoc_module,
            "StackedShortcut",
            DummyStackedShortcut,
        )

        result = explainer.generate_counterfactuals(sample=classification_sample)

        assert isinstance(result, Counterfactual)
        assert result.original_prediction in {0, 1}
        assert "feature2" in result.counterfactuals.columns

    def test_classifier_generate_counterfactuals_uses_debug_tree_fallback(
        self,
        monkeypatch,
        model_trained_classifier,
        celia_data_classification,
        classification_sample,
    ):
        explainer = BugDocClassifierExplainer(
            model=model_trained_classifier,
            data=celia_data_classification,
        )

        monkeypatch.setattr(
            bugdoc_module,
            "StackedShortcut",
            DummyStackedShortcutEmpty,
        )
        monkeypatch.setattr(
            bugdoc_module,
            "DebuggingDecisionTrees",
            DummyDebuggingDecisionTreesWithTree,
        )
        monkeypatch.setattr(
            bugdoc_module._tree,
            "get_depth",
            lambda *_: 1,
        )
        monkeypatch.setattr(
            bugdoc_module,
            "prune_tree",
            lambda t, keys: [["feature3 >= 200"]],
        )

        result = explainer.generate_counterfactuals(sample=classification_sample)

        assert isinstance(result, Counterfactual)
        assert result.original_prediction in {0, 1}
        assert result.counterfactuals.iloc[0]["feature3"] == 200.0

    def test_classifier_generate_counterfactuals_raises_when_no_counterfactuals(
        self,
        monkeypatch,
        model_trained_classifier,
        celia_data_classification,
        classification_sample,
    ):
        explainer = BugDocClassifierExplainer(
            model=model_trained_classifier,
            data=celia_data_classification,
        )

        monkeypatch.setattr(
            bugdoc_module,
            "StackedShortcut",
            DummyStackedShortcutEmpty,
        )
        monkeypatch.setattr(
            bugdoc_module,
            "DebuggingDecisionTrees",
            DummyDebuggingDecisionTrees,
        )
        monkeypatch.setattr(
            bugdoc_module._tree,
            "get_depth",
            lambda *_: 0,
        )

        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample=classification_sample)
