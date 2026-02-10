import pytest
from celia.explainers import NNCEClassifierExplainer, NNCERegressorExplainer
from celia.model import SklearnModel
from celia.data import PublicData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError

# TODO : Validate that NNCE receives exactly one instance at a time.

class TestNNCERegressorExplainer:

    def test_nnce_creation_with_valid_init(
            self,
            model_trained_without_encoded_data,
            dummy_regression_dataframe
    ):
        data = dummy_regression_dataframe
        X = data.drop(columns=['target'])
        y = data['target']
        public_data = PublicData(
            data=X,
            targets=y,
            target_name='target',
            column_names=X.columns.tolist(),
            continuous_column_names=X.select_dtypes(include=['float', 'int']).columns.tolist(),
            categorical_column_names=X.select_dtypes(include=['object', 'bool']).columns.tolist(),
            immutable_column_names=['feature1'],
            feasible_values={}
        )

        explainer = NNCERegressorExplainer(
            model=model_trained_without_encoded_data,
            data=public_data,
        )

        assert isinstance(explainer, NNCERegressorExplainer)
        assert isinstance(explainer.data, PublicData)
        assert isinstance(explainer.model, SklearnModel)
        assert explainer.data.data.equals(X)

    def test_nnce_creation_without_public_data(self, model_trained_with_encoded_data):
        """Test that NNCERegressorExplainer raises ConfigurationError when data is not PublicData"""
        class InvalidData:
            pass

        invalid_data = InvalidData()
        with pytest.raises(ConfigurationError) as exc_info:
            NNCERegressorExplainer(model=model_trained_with_encoded_data,
                                       data=invalid_data)
            err = exc_info.value
            assert "NNCERegressorExplainer requires data to be an instance of PublicData." in err.message
            assert err.param == "data"

    def test_nnce_run_without_target_range(self,  model_trained_without_encoded_data,
                                                  celia_public_data_without_encoded_data,
                                               dummy_test_regression_dataframe_encoded):
        """Test that NNCERegressorExplainer raises ConfigurationError when target_range is not provided"""

        explainer = NNCERegressorExplainer(model=model_trained_without_encoded_data,
                                               data=celia_public_data_without_encoded_data)

        sample = dummy_test_regression_dataframe_encoded
        sample = sample.drop(columns=['target'])
        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample)
            msg = str(exc_info.value)
            assert "target_range must be provided for regression counterfactual generation." in msg

    def test_nnce_run_with_multiple_instances(self, model_trained_without_encoded_data,
                                           celia_public_data_without_encoded_data,
                                           dummy_regression_dataframe):


        """Test that NNCERegressorExplainer raises ConfigurationError when multiple instances are provided. It can only do
        one at a time."""

        explainer = NNCERegressorExplainer(model=model_trained_without_encoded_data,
                                                 data=celia_public_data_without_encoded_data,
                                               )

        sample = dummy_regression_dataframe
        sample = sample.drop(columns=['target'])
        target_range = [10.0, 20.0]
        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample, target_range=target_range)
            err = exc_info.value
            assert "NNCERegressorExplainer can only generate counterfactuals for one instance at a time." in err.message
            assert err.param == "sample"

    def test_nnce_no_counterfactuals_found(self, model_trained_without_encoded_data,
                                              celia_public_data_without_encoded_data,
                                              dummy_regression_dataframe):
        """Test that CertifaiRegressorExplainer raises ConfigurationError when data is not encoded"""
        explainer = NNCERegressorExplainer(model=model_trained_without_encoded_data,
                                           data=celia_public_data_without_encoded_data,
                                           )

        sample = dummy_regression_dataframe.iloc[[0]]
        sample = sample.drop(columns=['target'])
        target_range = [10.0, 20.0]

        with pytest.raises(NoCounterfactualsFoundError) as exc_info:
            explainer.generate_counterfactuals(sample, target_range=target_range)
            err = exc_info.value
            assert "No counterfactuals found for the given instance and target range." in err.message


class TestNNCEClassifierExplainer:

    def test_nnce_classifier_creation_with_valid_init(
            self,
            model_trained_classifier_stratified,
            dummy_classification_dataframe,
    ):
        data = dummy_classification_dataframe
        X = data.drop(columns=['target'])
        y = data['target']
        public_data = PublicData(
            data=X,
            targets=y,
            target_name='target',
            column_names=X.columns.tolist(),
            continuous_column_names=X.select_dtypes(include=['float', 'int']).columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=['feature1'],
            feasible_values={},
        )

        explainer = NNCEClassifierExplainer(
            model=model_trained_classifier_stratified,
            data=public_data,
        )

        assert isinstance(explainer, NNCEClassifierExplainer)
        assert isinstance(explainer.data, PublicData)
        assert isinstance(explainer.model, SklearnModel)
        assert explainer.data.data.equals(X)

    def test_nnce_classifier_creation_without_public_data(self, model_trained_classifier):
        """Test that NNCEClassifierExplainer raises ConfigurationError when data is not PublicData."""
        class InvalidData:
            pass

        invalid_data = InvalidData()
        with pytest.raises(ConfigurationError) as exc_info:
            NNCEClassifierExplainer(model=model_trained_classifier, data=invalid_data)

        err = exc_info.value
        assert "`data` must be an instance of PublicData." in err.message
        assert err.param == "data"

    def test_nnce_classifier_run_with_multiple_instances(
            self,
            model_trained_classifier_stratified,
            celia_public_data_classification,
            dummy_classification_dataframe,
    ):
        """Test that NNCEClassifierExplainer raises ConfigurationError when multiple instances are provided."""
        explainer = NNCEClassifierExplainer(
            model=model_trained_classifier_stratified,
            data=celia_public_data_classification,
        )

        sample = dummy_classification_dataframe.drop(columns=['target'])
        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample)

        err = exc_info.value
        assert "`sample` must contain exactly one instance." in err.message
        assert err.param == "sample"

    def test_nnce_classifier_no_counterfactuals_found(
            self,
            model_trained_classifier,
            celia_public_data_classification,
            dummy_classification_dataframe,
    ):
        """Test that NNCEClassifierExplainer raises NoCounterfactualsFoundError when no CFs are found.

        DummyClassifier(most_frequent) always predicts the same class, so no neighbors
        will have a different class prediction.
        """
        explainer = NNCEClassifierExplainer(
            model=model_trained_classifier,
            data=celia_public_data_classification,
        )

        sample = dummy_classification_dataframe.drop(columns=['target']).iloc[[0]]
        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample)

