import joblib
import pandas as pd
from certifai import CERTIFAI
from tests.test_constants import *


#Load data
df_train = pd.read_csv('../data/german_test.csv')
df_test = pd.read_csv('../data/german_test.csv')
X_train, y_train = df_train.drop(columns=['target']), df_train['target']
X_test, y_test = df_test.drop(columns=['target']), df_test['target']

#Load model and encoders
model = joblib.load('../models/xgboost_model_credit.pkl')
encoder = joblib.load('../encoders/catboost_encoder_credit.pkl')
label = joblib.load('../encoders/label_encoder_credit.pkl')

#Metadata
categorical_cols = GERMAN_CREDIT_CATEGORIC_COLUMNS
immutable_features = GERMAN_CREDIT_IMMUTABLE_FEATURES
mutable_features = X_train.columns.difference(immutable_features).tolist()
continuous_features = X_train.columns.difference(categorical_cols).tolist()
feasible_ranges = GERMAN_CREDIT_FEASIBLE_VALUES

certifai = CERTIFAI(Pm=0.1 , Pc=0.1)
certifai.set_constraints(x=X_train, fixed=immutable_features)

itbe = X_test.iloc[0].to_frame().T

results = certifai.fit(
            model,
            x=itbe,
            trained_with_columns=True,
            classification=True,
            generations=3,
            model_type="sklearn",
            distance='L1',
            select_retain=1000,
            gen_retain=500,
            final_k=1,
            verbose=False)

counterfactual_list = certifai.results[0][1]

columns = list(itbe.columns) + ['Target']
counterfactuals_df = pd.DataFrame(counterfactual_list, columns=columns)

print(counterfactuals_df)