import pandas as pd
import numpy as np
import joblib
import pickle
from celia.explainers.certifai import CERTIFAI
from tests.test_constants import *

# -----------------------------------------------------------------------------
# Load dataset and model
# -----------------------------------------------------------------------------
import joblib
import pandas as pd

#Load data
df_train = pd.read_csv('../german_test.csv')
df_test = pd.read_csv('../german_test.csv')
X_train, y_train = df_train.drop(columns=['target']), df_train['target']
X_test, y_test = df_test.drop(columns=['target']), df_test['target']

#Load model and encoders
model = joblib.load('../gradient_boosting.pkl')
onehot = joblib.load('../onehot_encoder.pkl')
label = joblib.load('../label_encoder.pkl')

# -----------------------------------------------------------------------------
# Setup CERTIFAI
# -----------------------------------------------------------------------------
immutable_features = GERMAN_CREDIT_IMMUTABLE_FEATURES
mutable_features = [col for col in df.columns if col not in immutable_features]

# -----------------------------------------------------------------------------
# Load sampled indices and build sample
# -----------------------------------------------------------------------------
with open("../sample.csv", "r") as f:
    sampled_indices = [int(line.strip()) for line in f]

# -----------------------------------------------------------------------------
# Run CERTIFAI one by one
# -----------------------------------------------------------------------------
results = {}
explainer = CERTIFAI(Pm=0.1, Pc=0.1)
explainer.set_constraints(x=df, fixed=immutable_features)
# Create mappings from index to position for tolerance and target
index_to_position = {idx: pos for pos, idx in enumerate(df.index)}

for idx in tqdm(sampled_indices, desc="Generating CEs one by one"):
    pos = index_to_position[idx]
    single_instance = df.loc[[idx]]
    single_tolerance = tolerances[pos]
    single_target = rayon_cibles[pos]

    explainer.fit(
        model,
        x=single_instance,
        model_input=df,
        generations=100,
        model_type="sklearn",
        classification=False,
        trained_with_columns=True,
        distance="L1",
        target_lower=np.array([single_target - single_tolerance]),
        target_upper=np.array([single_target + single_tolerance]),
        verbose=False
    )

    if explainer.results:
        original_sample = explainer.results[0][0]
        counterfactuals_list = explainer.results[0][1]

        if isinstance(counterfactuals_list, list) and counterfactuals_list:
            counterfactuals_df = pd.DataFrame(counterfactuals_list, columns=original_sample.columns)
            results[idx] = {"query": original_sample, "ces": counterfactuals_df}
        else:
            results[idx] = {"query": original_sample, "ces": None}

# -----------------------------------------------------------------------------
# Save to pickle
# -----------------------------------------------------------------------------
output_path = "../results/ces_caus_certifai.pkl"
with open(output_path, "wb") as f:
    pickle.dump(results, f)

print(f"✅ CERTIFAI results saved to: {output_path}")
