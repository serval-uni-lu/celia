import pytest
from celia.explainers._base import RegressorExplainer
import pandas as pd
from celia._errors import InstancesAreWithinRange, ConfigurationError


class DummyRegressorExplainer(RegressorExplainer):
    """Minimal subclass to allow instantiation for testing static methods."""
    def _generate_counterfactuals(self, *args, **kwargs):
        return []

    def _create_explainer(self, model, data, *args, **kwargs):
        return None

    def _validate_sample(self, sample, *args, **kwargs):
        return None


class TestRegressorExplainer:
    @pytest.mark.parametrize(
        "target_range",
        [
            [0.1, 0.9],
            (0, 1),
            [-5, 5],
        ],
    )
    def test_validate_target_range_valid_inputs(self, target_range,
                                                model_trained_without_encoded_data,
                                                celia_public_data_without_encoded_data):
        """Ensure _validate_target_range accepts valid range definitions."""
        explainer = DummyRegressorExplainer(model_trained_without_encoded_data, celia_public_data_without_encoded_data)
        explainer._validate_target_range(target_range)

    @pytest.mark.parametrize(
        "invalid_range,expected_msg",
        [
            (None, "must be provided"),
            ([1, 2, 3], "must be a list or tuple of two elements"),
            ("invalid", "must be a list or tuple"),
            ([5, 2], "Min must be less than max"),
            ((5, 5), "Min must be less than max"),
        ],
    )
    def test_validate_target_range_invalid_inputs(self, invalid_range, expected_msg,
                                                  model_trained_without_encoded_data,
                                                  celia_public_data_without_encoded_data):
        """Ensure _validate_target_range raises ConfigurationError for invalid target ranges."""

        explainer = DummyRegressorExplainer(model_trained_without_encoded_data, celia_public_data_without_encoded_data)
        with pytest.raises(ConfigurationError) as exc_info:
            explainer._validate_target_range(invalid_range)
        msg = str(exc_info.value)
        assert expected_msg.split()[0] in msg

    def test_filter_samples_within_target_range_dataframe(self, model_trained_without_encoded_data,
                                                          celia_public_data_without_encoded_data,
                                                          capsys):
        """Ensure samples within the target range are excluded from DataFrame input."""
        explainer = DummyRegressorExplainer(model_trained_without_encoded_data, celia_public_data_without_encoded_data)

        # Input sample (All predictions of sample are 0.3)
        sample = celia_public_data_without_encoded_data.data.iloc[:3]
        target_range = [0.2, 0.4]

        #Should be empty
        instances_outside_range = explainer._filter_samples_within_target_range(sample, target_range)

        assert isinstance(instances_outside_range, pd.DataFrame)
        assert all(col in instances_outside_range.columns for col in sample.columns)
        assert instances_outside_range.empty


        captured = capsys.readouterr()
        assert "Excluded" in captured.out
        assert "within target range" in captured.out

    def test_filter_samples_within_target_range_series(self, model_trained_without_encoded_data,
                                                       celia_public_data_without_encoded_data,
                                                       capsys):
        """Ensure Series input is converted to DataFrame internally and works correctly."""
        explainer = DummyRegressorExplainer(model_trained_without_encoded_data, celia_public_data_without_encoded_data)

        sample = celia_public_data_without_encoded_data.data.iloc[:3]  # Single instance (Series)
        target_range = [0.2, 0.4]

        instance_outside_range = explainer._filter_samples_within_target_range(sample, target_range)

        assert isinstance(instance_outside_range, pd.DataFrame)
        assert list(instance_outside_range.columns) == list(celia_public_data_without_encoded_data.data.columns)
        assert instance_outside_range.empty
        captured = capsys.readouterr()
        assert "Excluded" in captured.out
        assert "within target range" in captured.out

    def test_generate_counterfactuals_raises_when_all_within_range(self, model_trained_without_encoded_data,
                                                                   celia_public_data_without_encoded_data):
        """Ensure InstancesAreWithinRange is raised when all samples fall inside the target range."""
        explainer = DummyRegressorExplainer(model_trained_without_encoded_data, celia_public_data_without_encoded_data)

        sample = celia_public_data_without_encoded_data.data.iloc[0]  # Single instance (Series)
        target_range = [0.2, 0.4]

        with pytest.raises(InstancesAreWithinRange) as exc_info:
            explainer.generate_counterfactuals(sample, target_range)

        # Validate error message content
        assert "within the desired target range" in str(exc_info.value)