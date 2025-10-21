import pytest
from celia.explainers._base import RegressorExplainer
import pandas as pd
from celia._errors import InstancesAreWithinRange


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
    def test_validate_target_range_valid_inputs(self, target_range, trained_sklearn_model, valid_public_data):
        """Ensure _validate_target_range accepts valid range definitions."""
        explainer = DummyRegressorExplainer(trained_sklearn_model, valid_public_data)
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
    def test_validate_target_range_invalid_inputs(self, invalid_range, expected_msg, trained_sklearn_model, valid_public_data):
        """Ensure _validate_target_range raises ValueError for invalid target ranges."""

        explainer = DummyRegressorExplainer(trained_sklearn_model, valid_public_data)
        with pytest.raises(ValueError) as exc_info:
            explainer._validate_target_range(invalid_range)
        msg = str(exc_info.value)
        assert expected_msg.split()[0] in msg

    def test_filter_samples_within_target_range_dataframe(self, trained_sklearn_model, valid_public_data, capsys):
        """Ensure samples within the target range are excluded for DataFrame input."""
        explainer = DummyRegressorExplainer(trained_sklearn_model, valid_public_data)

        # Input sample (predictions of sample are all 0.3)
        sample = valid_public_data.data.iloc[:3]
        target_range = [0.2, 0.4]

        filtered = explainer._filter_samples_within_target_range(sample, target_range)

        assert isinstance(filtered, pd.DataFrame)
        assert all(col in filtered.columns for col in sample.columns)
        assert len(filtered) <= len(sample)


        captured = capsys.readouterr()
        assert "Excluded" in captured.out
        assert "within target range" in captured.out

    def test_filter_samples_within_target_range_series(self, trained_sklearn_model, valid_public_data, capsys):
        """Ensure Series input is converted to DataFrame internally and works correctly."""
        explainer = DummyRegressorExplainer(trained_sklearn_model, valid_public_data)

        sample = valid_public_data.data.iloc[:3]  # Single instance (Series)
        target_range = [0.2, 0.4]

        filtered = explainer._filter_samples_within_target_range(sample, target_range)

        assert isinstance(filtered, pd.DataFrame)
        assert list(filtered.columns) == list(valid_public_data.data.columns)
        captured = capsys.readouterr()
        assert "Excluded" in captured.out
        assert "within target range" in captured.out

    def test_generate_counterfactuals_raises_when_all_within_range(self, valid_public_data, trained_sklearn_model):
        """Ensure InstancesAreWithinRange is raised when all samples fall inside the target range."""
        explainer = DummyRegressorExplainer(trained_sklearn_model, valid_public_data)

        sample = valid_public_data.data.iloc[0]  # Single instance (Series)
        target_range = [0.2, 0.4]

        filtered = explainer._filter_samples_within_target_range(sample, target_range)

        with pytest.raises(InstancesAreWithinRange) as exc_info:
            explainer.generate_counterfactuals(sample, target_range)

        # Validate error message content
        assert "within the desired target range" in str(exc_info.value)