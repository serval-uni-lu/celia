import pytest
from celia.data import PublicData
from celia.explainers import CertifaiRegressorExplainer
from celia.model import SklearnModel
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
import pandas as pd

#TODO: Add tests where counterfactuals are successfully generated.


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
            categorical_column_names=X.select_dtypes(include=['object', 'bool']).columns.tolist(),
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