# CELIA
## Counterfactual Explanations for Tabular Data
CELIA is a Python library designed to generate counterfactual explanations for Machine Learning models trained on 
tabular data. It provides a user-friendly interface to create counterfactuals with multiple methods from the 
state-of-the-art that help understand model predictions and improve interpretability.

Counterfactual explanations are a powerful tool in the field of explainable AI, allowing users to understand how
small changes to input features can lead to different predictions from a model.

## Methods Implemented
|   Method Name   |            Reference            | Achieving Method |
|:---------------:|:-------------------------------:|:----------------:|
|      DiCE       |      Mothilal et al., 2020      |   Optimization   |
|    CERTIFAI     |       Sharma et al., 2019       |    Heuristic     |
| Growing Spheres |       Laugel et al., 2018       |   Optimization   |
|      NNCE       |        Nearest Neighbor         |     Instance     |
|      GRACE      |         Le at al., 2020         |    Heuristic     |
|      OCEAN      |     Parmentier et al., 2021     |   Optimization   |
|   CounterGAN    |     Nemirovsky et al., 2022     |    Generative    | 
|      CLEAR      | White and d'Avila Garcez., 2020 |    Heuristic     |

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
uv pip install git+https://github.com/serval-uni-lu/celia.git
```
#### From source with uv.lock
```bash
git clone https://github.com/serval-uni-lu/celia.git
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

pip install git+https://github.com/serval-uni-lu/celia.git
```

## Basic Usage

The following example shows how to generate counterfactual explanations for a classification model.
A full, runnable tutorial is available in [`notebooks/classifier_tutorial.ipynb`](notebooks/classifier_tutorial.ipynb).

### 1. Prepare your data and model

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your data (features and target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)
```

### 2. Define constraints and wrap with CELIA

```python
from celia.data import PublicData
from celia.model import SklearnModel

public_data = PublicData(
    data=X_train,
    targets=y_train,
    target_name="approved",
    continuous_column_names=["age", "annual_income", "credit_score"],
    immutable_column_names=["age"],           # age cannot be changed
    feasible_values={"credit_score": (300, 850)},  # realistic bounds
)

sklearn_model = SklearnModel(model)
```

### 3. Generate counterfactuals

```python
from celia.explainers import DiceClassifierExplainer, GSGClassifierExplainer, NNCEClassifierExplainer

# Pick an instance to explain
sample = X_test.iloc[[0]]

# DiCE — multiple diverse counterfactuals
dice = DiceClassifierExplainer(model=sklearn_model, data=public_data, method="random")
dice_cfs = dice.generate_counterfactuals(sample, total_CFs=3)

# Growing Spheres — closest decision boundary crossing
gsg = GSGClassifierExplainer(model=sklearn_model, data=public_data)
gsg_cf = gsg.generate_counterfactuals(sample)

# NNCE — nearest real training examples with a different prediction
nnce = NNCEClassifierExplainer(model=sklearn_model, data=public_data)
nnce_cf = nnce.generate_counterfactuals(sample, n_counterfactuals=3)
```

### 4. Inspect results

Each result is a `Counterfactual` object. Use `highlighted_counterfactuals` to see only the features that changed
(unchanged values appear as `"-"`):

```python
dice_cfs[0].highlighted_counterfactuals
```
