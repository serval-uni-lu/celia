# CELIA
## Counterfactual Explanations for Tabular Data
CELIA is a Python library designed to generate counterfactual explanations for Machine Learning models trained on 
tabular data. It provides a user-friendly interface to create counterfactuals with multiple methods from the 
state-of-the-art that help understand model predictions and improve interpretability.

Counterfactual explanations are a powerful tool in the field of explainable AI, allowing users to understand how
small changes to input features can lead to different predictions from a model.

## Methods Implemented
|  Method Name  |       Reference       | Achieving Method |
|:-------------:|:---------------------:|:----------------:|
| DiCE | Mothilal et al., 2020 |   Optimization   |
| CERTIFAI|  Sharma et al., 2019  | Heuristic |

## Installation
This package is currently only available via GitHub. To install it, make sure you have **Python 3.12** or later installed.
You can install it with either ``uv`` or ``pip``.
### Option A. Using `uv`
#### Quick Install from Github
```bash
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

# Install the package straight from GitHub
uv pip install git+https://github.com/<username>/celia.git
```
#### From source with uv.lock
```bash
git clone https://github.com/<username>/celia.git
cd celia

# Create venv and install exactly the locked deps
uv venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

uv sync
```

### Option B. Using `pip`
#### Quick Install from Github
```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

pip install git+https://github.com/<username>/celia.git
```

## Basic Usage
Suppose you want to generate counterfactual explanations for a regression model trained on a tabular dataset.

```python
import joblib
import pandas as pd
from celia.model import SklearnModel
from tests.test_constants import *
```

First load your data and model:
```python
#Load data
df_train = pd.read_csv('training_data.csv')
df_test = pd.read_csv('test_data.csv')
X_train, y_train = df_train.drop(columns=['Target']), df_train['Target']
X_test, y_test = df_test.drop(columns=['Target']), df_test['Target']

#Load model
model = joblib.load('model.joblib')

#targets
targets = model.predict(X_train)
targets = pd.Series(targets, name='target', index=X_train.index)
```
### Defining the "Counterfactual Problem"
The counterfactual explanation search is a problem dependent on the model, data, and a set of specific domain constraints.
As each problem is different, CELIA provides a flexible interface to define the problem.

#### Create a Data Object
CELIA's Data Object is a wrapper around your dataset that provides the necessary information for generating counterfactuals.
For example, you can constrain the search space to specific columns (mutable features), specify the allowed values for 
each feature, specify which columns should be treated as categories (in case you labelled encoded them), and more.

```python
#Metadata
categorical_cols = ["Category_Column_1", "Category_Column_2"]
immutable_features = ["Immutable_Column_1", "Immutable_Column_2"]
continuous_features = X_train.columns.difference(categorical_cols).tolist()
feasible_ranges = {"Category_Column_1": ["A", "B", "C"],
                   "Category_Column_2": ["X", "Y", "Z"],
                   "Continuous_Column_1": (0, 100),
                   "Continuous_Column_2": (10, 50)}
```
```python
from celia.data import PublicData
d = PublicData(data=X_train,
               targets=targets,
               target_name="Target",
               column_names=X_train.columns,
               continuous_column_names=continuous_features,
               categorical_column_names=categorical_cols,
               immutable_column_names=immutable_features,
               feasible_values=feasible_ranges,)
```

#### Create a Model Object
Methods in CELIA handle models in their own unique way, that is why we created a Model Object that wraps your model
to provide a unified interface for all methods. At the moment, CELIA provides a wrapper for Scikit-learn models, but
you can easily create your own wrapper for any model you want to use.

```python
from celia.model import SklearnModel

m = SklearnModel(model)
```

#### Create an Explainer Object
Now it is time to create an Explainer Object that will handle the counterfactual generation process. CELIA
provides several methods for generating counterfactuals, each with its own strengths and weaknesses. You can choose
the method that best suits your needs.

```python
from celia.explainers import DiceRegressorExplainer
dice = DiceRegressorExplainer(data=d, model=m, method='random')
```

#### Generate Counterfactuals
Now you can generate counterfactuals for a specific instance in your dataset. Depending on the method you choose, you 
can use extra arguments to control the generation process, such as the number of counterfactuals to generate, the maximum
number of iterations, and more.

```python
# Generate counterfactuals for a specific instance
itbe = X_test.iloc[1].to_frame().T
counterfactuals = dice.generate_counterfactuals(itbe, target_range=[0.5,1], total_CFs=1)
```
