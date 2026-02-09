import pytest
from celia.data import PublicData
from celia.explainers import DiceRegressorExplainer
from celia.model import SklearnModel
from dice_ml.explainer_interfaces.dice_random import DiceRandom
from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic
from celia.errors import ConfigurationError
import re

"""Unit tests for the DiceRegressorExplainer class."""




class TestDICERegressorExplainer:

    @pytest.mark.parametrize("method, expected_cls", [("random", DiceRandom), ("genetic", DiceGenetic), ], )
    def test_dice_regressor_with_valid_init(
            self,
            method,
            expected_cls,
            model_trained_with_encoded_data,
            celia_public_data_with_encoded_data,
    ):
        explainer = DiceRegressorExplainer(
            model=model_trained_with_encoded_data,
            data=celia_public_data_with_encoded_data,
            method=method,
        )

        assert isinstance(explainer, DiceRegressorExplainer)
        assert isinstance(explainer.explainer, expected_cls)
        assert isinstance(explainer.data, PublicData)
        assert isinstance(explainer.model, SklearnModel)

    def test_dice_with_invalid_method(self, model_trained_with_encoded_data, celia_public_data_with_encoded_data):
        with pytest.raises(Exception, match=r"Unsupported sample strategy .* provided\. Please choose one of .*"):
            DiceRegressorExplainer(
                model=model_trained_with_encoded_data,
                data=celia_public_data_with_encoded_data,
                method="invalid_method",
            )

    def test_invalid_data_object(self, model_trained_with_encoded_data):
        # Create a dummy data object that is not an instance of PublicData
        class InvalidData:
            pass

        invalid_data = InvalidData()

        with pytest.raises(ConfigurationError) as exc_info:
            DiceRegressorExplainer(
                model=model_trained_with_encoded_data,
                data=invalid_data,
            )

        err = exc_info.value
        assert re.search(r"requires data to be an instance of PublicData\.", str(err.message))
        assert err.param == "data"
        assert err.hint.startswith("Please provide")
        assert err.config == {"data_type": "InvalidData"}

#def test_model_without_predict(create_model_without_predict, dummy_celia_public_data_encoded):
