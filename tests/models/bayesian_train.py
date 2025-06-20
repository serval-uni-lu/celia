from sklearn.datasets import fetch_openml
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import joblib

# Load the dataset from OpenML with ID 553
data = fetch_openml(data_id=553, as_frame=True)
df = data.frame.copy()

# Label encode specific categorical columns
categorical_cols = ['status', 'sex', 'disease_type']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Separate features and target
X = df.drop(columns=['frailty'])
y = df['frailty']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save training and test sets (with target included)
train_df = X_train.copy()
train_df['frailty'] = y_train
train_df.to_csv('kidney_train.csv', index=False)

test_df = X_test.copy()
test_df['frailty'] = y_test
test_df.to_csv('kidney_test.csv', index=False)

# Scale full dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train and evaluate model
model = BayesianRidge()
model.fit(X_scaled, y)
preds = model.predict(X_scaled)
mse = mean_squared_error(y, preds)
print(f"Bayesian Ridge MSE (in-sample): {mse:.4f}")

# Save model
joblib.dump(model, 'bayesian_model_kidney.pkl')
