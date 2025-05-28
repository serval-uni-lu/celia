import joblib
import pandas as pd
from dice_ml import Data, Dice, Model
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

d = Data(dataframe=df_train, continuous_features=continuous_features, permitted_range=feasible_ranges,outcome_name='target')
m = Model(model=model, backend='sklearn', model_type='classifier')
dice_random = Dice(d, m, method="random")

itbe = X_test.iloc[0].to_frame().T

results_random = dice_random.generate_counterfactuals(
            itbe,
            total_CFs=3,
            features_to_vary=mutable_features,
            permitted_range=feasible_ranges)

counterfactual_list = results_random.cf_examples_list[0].final_cfs_df