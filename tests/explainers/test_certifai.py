import pandas as pd
import pytest

from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import CertifaiClassifierExplainer, CertifaiRegressorExplainer
from celia.model import SklearnModel
from tests.conftest import categorical_select_dtypes

# TODO: Debug failing tests

class TestCertifaiRegressorExplainer:

    def test_certifai_creation_with_valid_init(
            self,
            model_trained_without_encoded_data,
            dummy_regression_dataframe
    ):
        data: pd.DataFrame = dummy_regression_dataframe
        X = data.drop(columns=['target'])
        y = data['target']
        public_data = PublicData(
            data=X,
            targets=y,
            target_name='target',
            column_names=X.columns.tolist(),
            continuous_column_names=X.select_dtypes(include=['float', 'int']).columns.tolist(),
            categorical_column_names=X.select_dtypes(include=categorical_select_dtypes()).columns.tolist(),
            immutable_column_names=['feature1'],
            feasible_values={}
        )

        explainer = CertifaiRegressorExplainer(
            model=model_trained_without_encoded_data,
            data=public_data,
        )

        assert isinstance(explainer, CertifaiRegressorExplainer)
        assert isinstance(explainer.data, PublicData)
        assert isinstance(explainer.model, SklearnModel)
        assert explainer.data.data.equals(X)

    def test_certifai_creation_without_public_data(self, model_trained_with_encoded_data):
        class InvalidData:
            pass

        invalid_data = InvalidData()
        with pytest.raises(ConfigurationError) as exc_info:
            CertifaiRegressorExplainer(model=model_trained_with_encoded_data,
                                       data=invalid_data)
            err = exc_info.value
            assert "CertifaiRegressorExplainer requires data to be an instance of PublicData." in err.message
            assert err.param == "data"

    def test_certifai_run_without_target_range(self,  model_trained_without_encoded_data,
                                                  celia_public_data_without_encoded_data,
                                               dummy_test_regression_dataframe_encoded):
        """Test that CertifaiRegressorExplainer raises ConfigurationError when target_range is not provided"""

        explainer = CertifaiRegressorExplainer(model=model_trained_without_encoded_data,
                                               data=celia_public_data_without_encoded_data)

        sample = dummy_test_regression_dataframe_encoded
        sample = sample.drop(columns=['target'])
        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample)
            msg = str(exc_info.value)
            assert "target_range must be provided for regression counterfactual generation." in msg

    def test_certifai_run_without_encoding(self, model_trained_without_encoded_data,
                                           celia_public_data_without_encoded_data,
                                           dummy_regression_dataframe):

        """Test that CertifaiRegressorExplainer raises ConfigurationError when data is not encoded"""
        explainer = CertifaiRegressorExplainer(model=model_trained_without_encoded_data,
                                                 data=celia_public_data_without_encoded_data,
                                               )

        sample = dummy_regression_dataframe
        sample = sample.drop(columns=['target'])
        target_range = [10.0, 20.0]
        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample, target_range=target_range)
            err = exc_info.value
            assert "CERTIFAI requires that categorical features are one-hot encoded." in err.message
            assert err.param == "sample"

    def test_certifai_no_counterfactuals_found(self, model_trained_with_encoded_data,
                                               celia_public_data_with_encoded_data,
                                               dummy_test_regression_dataframe_encoded):
        """Test that CertifaiRegressorExplainer raises NoCounterfactualsFound when no counterfactuals are found"""

        explainer = CertifaiRegressorExplainer(model=model_trained_with_encoded_data,
                                               data=celia_public_data_with_encoded_data)

        sample = dummy_test_regression_dataframe_encoded
        sample = sample.drop(columns=['target'])
        target_range = [1000.0, 2000.0] # Reminder, fixture model is a DummyRegressor predicting median = 0.3
        with pytest.raises(NoCounterfactualsFoundError) as exc_info:
            explainer.generate_counterfactuals(sample, target_range=target_range)
            err = exc_info.value
            assert "No counterfactuals generated. Check the input parameters and data." in err.message

    @pytest.mark.xfail(reason="Method broken, under investigation", strict=False)
    def test_certifai_regressor_counterfactuals_exclude_target_column(
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

        public_data = PublicData(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = CertifaiRegressorExplainer(model=model, data=public_data)

        sample = X.iloc[[0]]
        current_pred = tree.predict(sample)[0]
        # Shift target range away from current prediction
        target_range = [current_pred + 0.05, current_pred + 0.5]

        results = explainer.generate_counterfactuals(sample, target_range=target_range, generations=2)
        target_name = public_data.target_name
        feature_columns = set(public_data.column_names)

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


class TestCertifaiClassifierExplainer:

    def test_certifai_classifier_creation_with_valid_init(
            self,
            model_trained_classifier_stratified,
            celia_public_data_classification,
    ):
        explainer = CertifaiClassifierExplainer(
            model=model_trained_classifier_stratified,
            data=celia_public_data_classification,
        )

        assert isinstance(explainer, CertifaiClassifierExplainer)
        assert isinstance(explainer.data, PublicData)
        assert isinstance(explainer.model, SklearnModel)

    def test_certifai_classifier_creation_without_public_data(self, model_trained_classifier):
        class InvalidData:
            pass

        invalid_data = InvalidData()
        with pytest.raises(ConfigurationError) as exc_info:
            CertifaiClassifierExplainer(model=model_trained_classifier, data=invalid_data)

        err = exc_info.value
        assert "CertifaiClassifierExplainer requires data to be an instance of PublicData." in err.message
        assert err.param == "data"

    def test_certifai_classifier_run_without_encoding(
            self,
            model_trained_classifier_stratified,
            celia_public_data_classification_with_categories,
            dummy_classification_dataframe_with_categories,
    ):
        """Test that CertifaiClassifierExplainer raises ConfigurationError when data is not encoded."""
        explainer = CertifaiClassifierExplainer(
            model=model_trained_classifier_stratified,
            data=celia_public_data_classification_with_categories,
        )

        sample = dummy_classification_dataframe_with_categories.drop(columns=['target'])
        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample)

        err = exc_info.value
        assert "CERTIFAI requires that categorical features are one-hot encoded." in err.message
        assert err.param == "sample"

    def test_certifai_classifier_no_counterfactuals_found(
            self,
            model_trained_classifier,
            celia_public_data_classification,
            dummy_classification_dataframe,
    ):
        """Test that CertifaiClassifierExplainer raises NoCounterfactualsFoundError.

        DummyClassifier(most_frequent) always predicts the same class, so CERTIFAI
        cannot find counterfactuals with a different prediction.
        """
        explainer = CertifaiClassifierExplainer(
            model=model_trained_classifier,
            data=celia_public_data_classification,
        )

        sample = dummy_classification_dataframe.drop(columns=['target'])
        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample)

    @pytest.mark.xfail(reason="Method broken, under investigation", strict=False)
    def test_certifai_classifier_counterfactuals_exclude_target_column(
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

        public_data = PublicData(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = CertifaiClassifierExplainer(model=model, data=public_data)

        sample = X.iloc[[0]]
        target_name = public_data.target_name
        feature_columns = set(public_data.column_names)

        results = explainer.generate_counterfactuals(sample, generations=2)

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