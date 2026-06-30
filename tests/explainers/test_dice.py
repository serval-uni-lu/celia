import re

import pandas as pd
import pytest
from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic
from dice_ml.explainer_interfaces.dice_random import DiceRandom

from celia.counterfactuals import Counterfactual
from celia.data import Data
from celia.errors import ConfigurationError
from celia.explainers import DiceClassifierExplainer, DiceRegressorExplainer
from celia.model import SklearnModel

"""Unit tests for the DiceRegressorExplainer class."""




class TestDICERegressorExplainer:

    @pytest.mark.parametrize("method, expected_cls", [("random", DiceRandom), ("genetic", DiceGenetic), ], )
    def test_dice_regressor_with_valid_init(
            self,
            method,
            expected_cls,
            model_trained_with_encoded_data,
            celia_data_with_encoded_data,
    ):
        explainer = DiceRegressorExplainer(
            model=model_trained_with_encoded_data,
            data=celia_data_with_encoded_data,
            method=method,
        )

        assert isinstance(explainer, DiceRegressorExplainer)
        assert isinstance(explainer.explainer, expected_cls)
        assert isinstance(explainer.data, Data)
        assert isinstance(explainer.model, SklearnModel)

    def test_dice_with_invalid_method(self, model_trained_with_encoded_data, celia_data_with_encoded_data):
        with pytest.raises(Exception, match=r"Unsupported sample strategy .* provided\. Please choose one of .*"):
            DiceRegressorExplainer(
                model=model_trained_with_encoded_data,
                data=celia_data_with_encoded_data,
                method="invalid_method",
            )

    def test_invalid_data_object(self, model_trained_with_encoded_data):
        # Create a dummy data object that is not an instance of Data
        class InvalidData:
            pass

        invalid_data = InvalidData()

        with pytest.raises(ConfigurationError) as exc_info:
            DiceRegressorExplainer(
                model=model_trained_with_encoded_data,
                data=invalid_data,
            )

        err = exc_info.value
        assert re.search(r"requires data to be an instance of Data\.", str(err.message))
        assert err.param == "data"
        assert err.hint.startswith("Please provide")
        assert err.config == {"data_type": "InvalidData"}

#def test_model_without_predict(create_model_without_predict, dummy_celia_data_encoded):

    def test_dice_regressor_counterfactuals_exclude_target_column(
        self,
        dummy_regression_dataframe_encoded,
    ):
        """Counterfactual.original_instance and .counterfactuals must not contain the target column."""
        from sklearn.tree import DecisionTreeRegressor

        X = dummy_regression_dataframe_encoded.drop(columns=["target"])
        y = dummy_regression_dataframe_encoded["target"]

        tree = DecisionTreeRegressor(random_state=0)
        tree.fit(X, y)
        model = SklearnModel(tree)

        data = Data(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = DiceRegressorExplainer(model=model, data=data)

        sample = X.iloc[[0]]
        current_pred = tree.predict(sample)[0]
        target_range = [current_pred + 0.05, current_pred + 0.5]
        target_name = data.target_name
        feature_columns = set(data.column_names)

        results = explainer.generate_counterfactuals(sample, target_range=target_range)

        if isinstance(results, Counterfactual):
            results = [results]

        for cf in results:
            assert target_name not in cf.original_instance.columns, (
                f"original_instance should not contain target column '{target_name}', "
                f"got columns: {list(cf.original_instance.columns)}"
            )
            assert target_name not in cf.counterfactuals.columns, (
                f"counterfactuals should not contain target column '{target_name}', "
                f"got columns: {list(cf.counterfactuals.columns)}"
            )
            assert set(cf.original_instance.columns) == feature_columns
            assert set(cf.counterfactuals.columns) == feature_columns


class TestDICEClassifierExplainer:

    @pytest.mark.parametrize("method, expected_cls", [("random", DiceRandom), ("genetic", DiceGenetic)])
    def test_dice_classifier_with_valid_init(
            self,
            method,
            expected_cls,
            model_trained_classifier_stratified,
            celia_data_classification,
    ):
        explainer = DiceClassifierExplainer(
            model=model_trained_classifier_stratified,
            data=celia_data_classification,
            method=method,
        )

        assert isinstance(explainer, DiceClassifierExplainer)
        assert isinstance(explainer.explainer, expected_cls)
        assert isinstance(explainer.data, Data)
        assert isinstance(explainer.model, SklearnModel)

    def test_dice_classifier_with_invalid_method(
            self,
            model_trained_classifier_stratified,
            celia_data_classification,
    ):
        with pytest.raises(Exception, match=r"Unsupported sample strategy .* provided\. Please choose one of .*"):
            DiceClassifierExplainer(
                model=model_trained_classifier_stratified,
                data=celia_data_classification,
                method="invalid_method",
            )

    def test_dice_classifier_invalid_data_object(self, model_trained_classifier):
        class InvalidData:
            pass

        invalid_data = InvalidData()

        with pytest.raises(ConfigurationError) as exc_info:
            DiceClassifierExplainer(
                model=model_trained_classifier,
                data=invalid_data,
            )

        err = exc_info.value
        assert re.search(r"requires data to be an instance of Data\.", str(err.message))
        assert err.param == "data"
        assert err.hint.startswith("Please provide")
        assert err.config == {"data_type": "InvalidData"}

    def test_dice_classifier_counterfactuals_exclude_target_column(
        self,
        dummy_classification_dataframe,
    ):
        """Counterfactual.original_instance and .counterfactuals must not contain the target column."""
        from sklearn.tree import DecisionTreeClassifier

        X = dummy_classification_dataframe.drop(columns=["target"])
        y = dummy_classification_dataframe["target"]

        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        model = SklearnModel(tree)

        data = Data(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = DiceClassifierExplainer(model=model, data=data)

        sample = X.iloc[[0]]
        target_name = data.target_name
        feature_columns = set(data.column_names)

        results = explainer.generate_counterfactuals(sample)

        if isinstance(results, Counterfactual):
            results = [results]

        for cf in results:
            assert target_name not in cf.original_instance.columns, (
                f"original_instance should not contain target column '{target_name}', "
                f"got columns: {list(cf.original_instance.columns)}"
            )
            assert target_name not in cf.counterfactuals.columns, (
                f"counterfactuals should not contain target column '{target_name}', "
                f"got columns: {list(cf.counterfactuals.columns)}"
            )
            assert set(cf.original_instance.columns) == feature_columns
            assert set(cf.counterfactuals.columns) == feature_columns
