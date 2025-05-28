import pytest
import pandas as pd
from celia.data import PublicData


@pytest.fixture
def valid_inputs():
    df = pd.DataFrame({
        "age": [25, 30, 45],
        "income": [50000, 60000, 80000],
        "gender": ["M", "F", "F"]
    })
    labels = pd.Series([1, 0, 1])
    return {
        "data": df,
        "labels": labels,
        "target_names": ["target"],
        "continuous": ["age", "income"],
        "categorical": ["gender"],
        "immutable": ["age"],
        "feasible_values": {
            "age": (18, 65),
            "income": (0, 100000),
            "gender": ["M", "F"]
        }
    }


def test_valid_creation(valid_inputs):
    d = PublicData(**valid_inputs)
    assert isinstance(d.data, pd.DataFrame)
    assert isinstance(d.labels, pd.Series)
    assert set(d.continuous) == {"age", "income"}
    assert set(d.categorical) == {"gender"}


def test_missing_feature_in_continuous(valid_inputs):
    valid_inputs["continuous"] = ["age", "height"]  # 'height' not in data
    with pytest.raises(ValueError, match="not in the dataset"):
        PublicData(**valid_inputs)


def test_overlap_between_continuous_and_categorical(valid_inputs):
    valid_inputs["categorical"].append("age")
    with pytest.raises(ValueError, match="both continuous and categorical"):
        PublicData(**valid_inputs)


def test_misaligned_data_labels(valid_inputs):
    valid_inputs["labels"] = pd.Series([0, 1])  # length mismatch
    with pytest.raises(ValueError, match="must have the same number of instances"):
        PublicData(**valid_inputs)


def test_invalid_range_in_feasible_values(valid_inputs):
    valid_inputs["feasible_values"]["income"] = (100000, 50000)  # invalid range
    with pytest.raises(ValueError, match="min must be less than max"):
        PublicData(**valid_inputs)


def test_feature_not_in_data_in_feasible_values(valid_inputs):
    valid_inputs["feasible_values"]["occupation"] = ["engineer", "doctor"]
    with pytest.raises(ValueError, match="not in data"):
        PublicData(**valid_inputs)
