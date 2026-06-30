# CELIA

**Counterfactual Explanations Library for Tabular Data**

CELIA is a Python library that generates counterfactual explanations for machine-learning models trained on tabular data. It wraps **10 state-of-the-art methods** behind a single, consistent API — so you can swap algorithms in one line, declare real-world constraints once, and compare results side by side.

> *"What would need to change about this input for the model to predict a different outcome?"*

Counterfactual explanations answer this question. Imagine Alice applies for a loan and gets denied — a counterfactual might say: *"If your credit score were 720 instead of 650, you would have been approved."*

---

## Key Features

- **Unified interface** — every method follows the same `Explainer → generate_counterfactuals → Counterfactual` workflow.
- **10 explanation methods** spanning optimization, instance-based, generative, heuristic, and stochastic approaches.
- **Constraint-aware** — declare immutable features, feasible value ranges, monotonicity, and feature correlations once in the `Data` object;
- **Backend-agnostic** — wrap any scikit-learn estimator or PyTorch module with a single adapter class.

---

## Implemented Methods

| Method | Reference | Approach | Constraints |
|:---:|:---:|:---:|:---|
| DiCE | Mothilal et al., 2020 | Optimization | immutable, feasible |
| Growing Spheres | Laugel et al., 2018 | Optimization | immutable |
| NNCE | Nearest Neighbor | Instance | immutable |
| GRACE | Le et al., 2020 | Heuristic | feasible |
| OCEAN | Parmentier et al., 2021 | Optimization | immutable, feasible |
| CounterGAN | Nemirovsky et al., 2022 | Generative | immutable |
| CLEAR | White & d'Avila Garcez, 2020 | Heuristic | — |
| C-CHVAE | Pawelczyk et al., 2020 | Generative | immutable |
| BugDoc | Lourenco et al., 2020 | Heuristic | immutable |
| FastAR | Verma et al., 2020 | Stochastic | immutable, monotonic, correlated |

---

## Installation

CELIA is installed from GitHub (requires **Python 3.12+**).

### Using `uv` (recommended)

```bash
git clone https://github.com/serval-uni-lu/celia.git
cd celia

uv venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows

uv sync                       # install core dependencies
```

### Using `pip`

```bash
pip install git+https://github.com/serval-uni-lu/celia.git
```

### Optional extras

Some methods require additional dependencies:

| Extra | Methods enabled | Install |
|:---:|:---|:---|
| `torch` | CounterGAN, C-CHVAE | `uv sync --extra torch` or `pip install "celia[torch] @ git+https://github.com/serval-uni-lu/celia.git"` |
| `ocean` | OCEAN | `uv sync --extra ocean` or `pip install "celia[ocean] @ git+https://github.com/serval-uni-lu/celia.git"` |
| `stochastic` | FastAR | `uv sync --extra stochastic` or `pip install "celia[stochastic] @ git+https://github.com/serval-uni-lu/celia.git"` |
| `all` | All of the above | `uv sync --extra all` or `pip install "celia[all] @ git+https://github.com/serval-uni-lu/celia.git"` |

---

## Quick Start

A full, runnable walkthrough is available in [`notebooks/classifier_tutorial.ipynb`](notebooks/classifier_tutorial.ipynb).

### 1. Prepare your data and model

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)
```

### 2. Define constraints and wrap with CELIA

```python
from celia import Data, SklearnModel

data = Data(
    data=X_train,
    targets=y_train,
    target_name="approved",
    continuous_column_names=["age", "annual_income", "credit_score"],
    immutable_column_names=["age"],                    # age cannot be changed
    feasible_values={"credit_score": (300, 850)},      # realistic bounds
)

sklearn_model = SklearnModel(model)
```

### 3. Generate counterfactuals

```python
from celia import DiceClassifierExplainer, GrowingSpheresClassifierExplainer, NNCEClassifierExplainer

sample = X_test.iloc[[0]]

# DiCE — multiple diverse counterfactuals
dice = DiceClassifierExplainer(model=sklearn_model, data=data, method="random")
dice_cfs = dice.generate_counterfactuals(sample, total_CFs=3)

# Growing Spheres — closest decision boundary crossing
gs = GrowingSpheresClassifierExplainer(model=sklearn_model, data=data)
gs_cf = gs.generate_counterfactuals(sample)

# NNCE — nearest real training examples with a different prediction
nnce = NNCEClassifierExplainer(model=sklearn_model, data=data)
nnce_cf = nnce.generate_counterfactuals(sample, n_counterfactuals=3)
```

### 4. Inspect results

Each result is a `Counterfactual` object. Use `highlighted_counterfactuals` to see only the features that changed (unchanged values appear as `"-"`):

```python
dice_cfs[0].highlighted_counterfactuals
```

```
  age  annual_income  credit_score  employment_years  debt_to_income_ratio  num_credit_lines  Prediction
0   -          116.8           644                 -                 0.316                 -     0.0 → 1
1   -          116.8           744                 -                 0.316                11     0.0 → 1
2   -          116.8           738                 -                 0.316                 -     0.0 → 1
```

Notice that `age` is always `"-"` — the immutable constraint is respected automatically.

---

## Documentation

Full documentation (user guide, examples, and API reference) is available at the [CELIA docs site](https://serval-uni-lu.github.io/celia/).

---

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the [MIT License](LICENSE).

Developed at the [Interdisciplinary Centre for Security, Reliability and Trust (SnT)](https://www.uni.lu/snt-en/), University of Luxembourg, within the [SerVal](https://serval-snt-uni-lu.github.io//) research group.
