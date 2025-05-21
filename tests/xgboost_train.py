import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from category_encoders import CatBoostEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Load dataset
data = fetch_openml("credit-g", version=2, as_frame=True, parser="liac-arff")
df = data.frame

# Separate features and target
X = df.drop(columns="class")
y = df["class"]


le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), name='class')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)


# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Fit and apply CatBoostEncoder
cat_encoder = CatBoostEncoder(cols=categorical_cols, return_df=True, random_state=42)
cat_encoder.fit_transform(X_train[categorical_cols], y_train)
X_train[categorical_cols] = cat_encoder.transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_encoder.transform(X_test[categorical_cols])

print(X_train.head())

# Save encoded datasets
X_train["target"] = y_train
X_test["target"] = y_test
X_train.to_csv("data/german_train.csv", index=False)
X_test.to_csv("data/german_test.csv", index=False)

# Train XGBoost model
clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    eval_metric='logloss',
    random_state=42
)
clf.fit(X_train.drop(columns="target"), y_train)

# Evaluate
y_pred = clf.predict(X_test.drop(columns="target"))
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# Save model and encoder

joblib.dump(clf, "models/xgboost_model_credit.pkl")
joblib.dump(cat_encoder, "encoders/catboost_encoder_credit.pkl")
joblib.dump(le, "encoders/label_encoder_credit.pkl")
print("Model and encoders saved.")
