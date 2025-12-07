from typing import List, Optional, Tuple

import pandas as pd
from pandas import DataFrame, Series
from sklearn.datasets import fetch_openml
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Optional dependencies
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from category_encoders import CatBoostEncoder
except ImportError:
    CatBoostEncoder = None


def create_dataset(
    regression: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    use_catboost: bool = False,
) -> Tuple[DataFrame, DataFrame, str]:
    """Download, preprocess, and split an OpenML dataset.

    Parameters
    ----------
    regression : bool, optional
        If True, use the Kidney dataset (ID=553, regression).
        If False, use the German Credit dataset ("credit-g", classification).
    test_size : float, optional
        Fraction of the dataset to include in the test split.
    random_state : int, optional
        Random seed for reproducibility.
    use_catboost : bool, optional
        If True, apply CatBoostEncoder on categorical features.

    Returns
    -------
    train_df : pd.DataFrame
        Training set including target column.
    test_df : pd.DataFrame
        Test set including target column.
    target_col : str
        Name of the target column.
    """
    df: DataFrame
    target_col: str
    categorical_cols: List[str]

    if regression:
        # Kidney dataset
        data = fetch_openml(data_id=553, as_frame=True, parser="liac-arff")
        df = data.frame.copy()
        target_col = "frailty"
        categorical_cols = ["status", "sex", "disease_type"]

    else:
        # German Credit dataset (classification)
        data = fetch_openml(name="credit-g", version=2, as_frame=True, parser="liac-arff")
        df = data.frame.copy()
        target_col = "class"
        categorical_cols = (
            df.select_dtypes(include="category").columns.drop(target_col).tolist()
        )

    y: Series = df[target_col]
    if not regression:
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=target_col)
    else:
        y = df[target_col]

    # Features
    X: DataFrame = df.drop(columns=[target_col])

    if use_catboost:
        if CatBoostEncoder is None:
            raise ImportError("category_encoders must be installed to use CatBoostEncoder.")
        encoder = CatBoostEncoder(cols=categorical_cols, return_df=True, random_state=random_state)
        X = encoder.fit_transform(X, y)
    else:
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col])

    stratify = y if not regression else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = stratify, random_state = random_state)

    # Rebuild DataFrames with target
    train_df: DataFrame = X_train.copy()
    train_df[target_col] = y_train
    test_df: DataFrame = X_test.copy()
    test_df[target_col] = y_test

    return train_df, test_df, target_col


def create_model(
    X_train: DataFrame,
    y_train: Series,
    X_test: Optional[DataFrame] = None,
    y_test: Optional[Series] = None,
    regression: bool = True,
    verbose: bool = True,
):
    """Train a regression (BayesianRidge) or classification (XGBClassifier) model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame, optional
        Test features for evaluation (required if verbose=True).
    y_test : pd.Series, optional
        Test target for evaluation (required if verbose=True).
    regression : bool, optional
        If True, train a BayesianRidge regression model.
        If False, train an XGBClassifier classification model.
    verbose : bool, optional
        If True, print evaluation metrics on the test set.

    Returns
    -------
    model : object
        Trained model.
    """
    if regression:
        model = BayesianRidge()
    else:
        if XGBClassifier is None:
            raise ImportError("xgboost must be installed to use XGBClassifier.")
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
        )

    # Train model
    model.fit(X_train, y_train)

    # Evaluation
    if verbose:
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided when verbose=True")

        y_pred = model.predict(X_test)

        if regression:
            mse = mean_squared_error(y_test, y_pred)
            print(f"BayesianRidge MSE (test set): {mse:.4f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            print(f"XGBClassifier Accuracy (test set): {acc:.4f}\n")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    regression = True
    train_df, test_df, target_col = create_dataset(regression=regression, use_catboost=False)
    X_train, X_test = train_df.drop(columns=target_col), test_df.drop(columns=target_col)
    y_train, y_test = train_df[target_col], test_df[target_col]
    create_model(X_train, y_train, X_test, y_test, regression=regression, verbose=True)
