"""
Pytest that verifies **CELIA's** `DiceRegressorExplainer` keeps integer
columns as integer dtype in the generated counterfactuals.

This version uses a **fully synthetic dataset and model** so it does not rely
on any external files.

Run with:
    pytest -q -s test_dice_dtypes.py
(The **-s** flag lets you see the debug prints below.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


from celia.model import SklearnModel
from celia.data import PublicData
from celia.explainers import DiceRegressorExplainer

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_dummy_dataset(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Construct a toy dataset with two integer and one float feature."""
    rng = np.random.default_rng(seed)

    age = rng.integers(20, 80, size=n)
    systolic_bp = rng.integers(90, 180, size=n)
    cholesterol = rng.uniform(150, 300, size=n)

    # A simple continuous outcome between 0 and 1
    risk = (
        0.4 * (age - 20) / 60
        + 0.4 * (systolic_bp - 90) / 90
        + 0.2 * (cholesterol - 150) / 150
    )
    # Add small noise and clip
    risk = np.clip(risk + rng.normal(0, 0.02, size=n), 0, 1)

    df = pd.DataFrame(
        {
            "age": age,
            "systolic_bp": systolic_bp,
            "cholesterol": cholesterol,
            "risk": risk,
        }
    )
    return df


def _build_celia_objects(seed: int = 42):
    """Create PublicData, SklearnModel, and DiceRegressorExplainer on dummy data."""
    df = _make_dummy_dataset(seed=seed)

    # Split train / test
    train_frac = 0.8
    train_df = df.sample(frac=train_frac, random_state=seed)
    test_df = df.drop(train_df.index)

    X_train, y_train = train_df.drop(columns=["risk"]), train_df["risk"]
    X_test, y_test = test_df.drop(columns=["risk"]), test_df["risk"]

    # Basic scikit‑learn regressor
    model = LinearRegression().fit(X_train, y_train)

    # Wrap for CELIA
    m = SklearnModel(model)

    d = PublicData(
        data=X_train,
        targets=y_train,
        target_name="risk",
        column_names=X_train.columns,
        continuous_column_names=list(X_train.columns),  # treat every feature as continuous
        categorical_column_names=None,
        immutable_column_names=None,  # age is immutable in CFs
        feasible_values={'age': (20, 80), 'systolic_bp': (90, 180), 'cholesterol': (150, 300)},
    )

    dice = DiceRegressorExplainer(data=d, model=m, method="random")
    return dice, X_train, X_test


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
def test_celia_dice_respects_int_dtype():
    """Integer‑typed features from the original data must remain ints in CFs."""

    dice, X_train, X_test = _build_celia_objects()

    print("Original training dtypes:\n", X_train.dtypes)

    # Pick an instance to explain
    query_instance = X_test.iloc[[0]]  # keep as DataFrame slice
    print("\nQuery instance used for CF generation:\n", query_instance)
    print(f"\nCurrent prediction for query instance: {dice.model.predict(query_instance)}")

    # Generate counterfactuals
    cf_result = dice.generate_counterfactuals(
        query_instance,
        target_range=[0.85, 1],
        total_CFs=3,
        verbose=False,
    )

    # Extract the DataFrame that holds counterfactuals
    if isinstance(cf_result, pd.DataFrame):
        cf_df = cf_result
    elif hasattr(cf_result, "cf_examples_list"):
        cf_df = cf_result.cf_examples_list[0].final_cfs_df
    elif hasattr(cf_result, "final_cfs_df"):
        cf_df = cf_result.final_cfs_df
    else:
        raise AttributeError("Unable to extract CF DataFrame from result object")

    print("\nCounterfactuals returned:\n", cf_df)
    print("\nDtypes of counterfactuals:\n", cf_df.dtypes)

    # Identify integer columns in the *training* data
    int_columns = [
        col for col in X_train.columns if pd.api.types.is_integer_dtype(X_train[col])
    ]
    print("\nInteger columns expected to remain ints:", int_columns)

    # ------------------------------------------------------------------
    # 1️⃣  Assert dtype stays integer
    # ------------------------------------------------------------------
    for col in int_columns:
        print(f"Checking dtype of column '{col}': {cf_df[col].dtype}")
        assert pd.api.types.is_integer_dtype(
            cf_df[col]
        ), f"Column '{col}' lost integer dtype (got {cf_df[col].dtype})."

    # ------------------------------------------------------------------
    # 2️⃣  Extra sanity: values themselves are integral (no .5 etc.)
    # ------------------------------------------------------------------
    for col in int_columns:
        series = cf_df[col].dropna()
        residue = series - series.astype(int)
        max_residue = residue.abs().max()
        print(f"Max non-integer residue in column '{col}':", max_residue)
        assert np.allclose(
            series,
            series.astype(int),
        ), f"Column '{col}' contains non-integer values."
