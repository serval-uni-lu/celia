import pytest
from typing import Any, Union
import pandas as pd
from celia.data import PublicData, BaseData
from celia.model import SklearnModel, BaseModel
from celia.explainers import BaseExplainer
from celia.errors import ConfigurationError
import numpy as np


class BasicExplainer(BaseExplainer):
    def __init__(self, model: BaseModel, data: BaseData):
        super().__init__(model, data)

    def _create_explainer(self, model: BaseModel, data:PublicData, *args, **kwargs) -> Any:
        pass

    def _validate_sample(self, sample: Union[pd.DataFrame, pd.Series], *args, **kwargs) -> None:
        pass

    def generate_counterfactuals(self, *args, **kwargs) -> Any:
        super().validate_sample(*args, **kwargs)

@pytest.fixture
def basic_explainer(model_trained_without_encoded_data, celia_public_data_without_encoded_data) -> BasicExplainer:
    return BasicExplainer(
        model=model_trained_without_encoded_data,
        data=celia_public_data_without_encoded_data
    )

class TestBaseExplainer:

    def test_valid_inputs(self, basic_explainer, model_trained_without_encoded_data, celia_public_data_without_encoded_data):
        explainer = BasicExplainer(
            model=model_trained_without_encoded_data,
            data=celia_public_data_without_encoded_data
        )
        assert isinstance(explainer, BasicExplainer)
        assert isinstance(explainer, BaseExplainer)
        assert isinstance(explainer.model, SklearnModel)
        assert isinstance(explainer.model, BaseModel)
        assert isinstance(explainer.data, BaseData)
        assert isinstance(explainer.data, PublicData)

    def test_invalid_data(self, model_trained_without_encoded_data):
        # Create a dummy data object that is not an instance of BaseData
        class InvalidData:
            pass

        invalid_data = InvalidData()

        with pytest.raises(ConfigurationError) as exc_info:
            BasicExplainer(
                model=model_trained_without_encoded_data,
                data=invalid_data,
            )

        err = exc_info.value
        assert err.message == "Explainer requires data to be an instance of BaseData."
        assert err.param == "data"
        assert err.hint.startswith("Please provide")
        assert err.config == {"data_type": "InvalidData"}

    def test_property_getters(self, basic_explainer, model_trained_without_encoded_data, celia_public_data_without_encoded_data):
        """Ensure that model, data, and explainer properties return correct objects."""
        # Model and data should be exactly the same objects passed during init
        assert basic_explainer.model is model_trained_without_encoded_data
        assert basic_explainer.data is celia_public_data_without_encoded_data

        # Explainer is created by subclass _create_explainer (in BasicExplainer it's None by default)
        assert basic_explainer.explainer is None

    def test_validate_sample_dtypes(self, basic_explainer):
        # Create a sample with incorrect dtypes
        sample = np.array([[0.20, 20.0, 300.0, "A", True]])

        with pytest.raises(ConfigurationError) as exc_info:
            basic_explainer.generate_counterfactuals(sample=sample)

        err = exc_info.value
        assert err.message == "Sample must be a pandas DataFrame or Series."
        assert err.param == "sample"
        assert err.hint.startswith("Please provide")
        assert err.config == {"sample_type": "ndarray"}

    def test_validate_sample_column_mismatches(self, basic_explainer, celia_public_data_without_encoded_data):
        """Ensure validate_sample raises errors for missing, extra, or duplicate columns."""

        # Missing required column
        sample_missing = celia_public_data_without_encoded_data.data.drop(columns=[celia_public_data_without_encoded_data.column_names[0]])
        with pytest.raises(ConfigurationError) as exc_info_missing:
            basic_explainer.validate_sample(sample_missing)
        assert "Missing columns" in str(exc_info_missing.value)

        # Extra unexpected column
        sample_extra = celia_public_data_without_encoded_data.data.copy()
        sample_extra["extra_feature"] = 999
        with pytest.raises(ConfigurationError) as exc_info_extra:
            basic_explainer.validate_sample(sample_extra)
        assert "Unexpected columns" in str(exc_info_extra.value)

        # Duplicate columns
        sample_dupes = celia_public_data_without_encoded_data.data.copy()
        sample_dupes = sample_dupes.rename(columns={celia_public_data_without_encoded_data.column_names[0]: "feature2"})
        with pytest.raises(ConfigurationError) as exc_info_dupes:
            basic_explainer.validate_sample(sample_dupes)
        assert "Duplicate column names" in str(exc_info_dupes.value)

    def test_validate_sample_accepts_series_and_dataframe(self, basic_explainer, celia_public_data_without_encoded_data):
        """Ensure validate_sample works with both DataFrame and Series inputs."""

        # DataFrame should pass validation
        df_sample = celia_public_data_without_encoded_data.data.iloc[[0]]  # keep as DataFrame
        try:
            basic_explainer.validate_sample(df_sample)
        except Exception as e:
            pytest.fail(f"DataFrame validation raised an unexpected error: {e}")

        # Series should also pass validation
        series_sample = celia_public_data_without_encoded_data.data.iloc[0]  # single row as Series
        try:
            basic_explainer.validate_sample(series_sample)
        except Exception as e:
            pytest.fail(f"Series validation raised an unexpected error: {e}")

    def test_base_explainer_cannot_be_instantiated(self, model_trained_without_encoded_data, celia_public_data_without_encoded_data):
        """Ensure BaseExplainer cannot be instantiated directly due to abstract methods."""
        with pytest.raises(TypeError) as exc_info:
            BaseExplainer(model=model_trained_without_encoded_data, data=celia_public_data_without_encoded_data)

        assert "abstract class" in str(exc_info.value)
        assert "_create_explainer" in str(exc_info.value)
        assert "_validate_sample" in str(exc_info.value)
        assert "generate_counterfactuals" in str(exc_info.value)

    def test_validate_sample_case_insensitive_columns(self, basic_explainer, celia_public_data_without_encoded_data):
        """Ensure column name matching is case-insensitive."""

        # Create a sample where all column names are uppercased
        df_upper = celia_public_data_without_encoded_data.data.copy()
        df_upper.columns = [c.upper() for c in celia_public_data_without_encoded_data.column_names]

        # Should validate without error despite case differences
        try:
            basic_explainer.validate_sample(df_upper)
        except Exception as e:
            pytest.fail(f"Validation failed for case-insensitive columns: {e}")

