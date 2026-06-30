# Contributing to CELIA

This guide walks you through adding a new **classifier explainer** method to CELIA. By the end you will have a working explainer with full unit tests that follow the project conventions.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

## Development Setup

```bash
# 1. Clone and enter the repository
git clone https://github.com/serval-uni-lu/celia
cd celia

# 2. Create a virtual environment and install all dependencies (including dev tools)
uv venv .venv && source .venv/bin/activate
uv sync --extra dev

# 3. Verify everything works
ruff check src/
ruff format --check src/
ty check src/celia
pytest tests/ -v
```

`uv sync --extra dev` installs ruff, ty, pytest, torch, and other dev tools. See `pyproject.toml` for the full dependency list.

---

## Adding a New Classifier Explainer

This section uses a fictional method called **"Foo"** as the running example. Replace `Foo` / `foo` with your method name throughout.

### Step 1: Create the Method Directory

Every method lives in its own directory under `src/celia/explainers/`:

```
src/celia/explainers/foo/
    __init__.py
    foo.py
```

The `__init__.py` re-exports your public class:

```python
from celia.explainers.foo.foo import FooClassifierExplainer

__all__ = ["FooClassifierExplainer"]
```

### Step 2: Implement the Explainer Class

Your explainer must inherit from `ClassifierExplainer` and implement three hooks. Here is the full skeleton:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from celia.counterfactuals import Counterfactual
from celia.data import Data
from celia.data._base import BaseData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.model import BaseModel, SklearnModel

if TYPE_CHECKING:
    pass  # Put type-only imports here (e.g., torch types)


class FooClassifierExplainer(ClassifierExplainer):
    """FOO: Full Name of Your Method.

    Brief description of what the method does and how it works.

    Parameters
    ----------
    model : SklearnModel
        A CELIA ``SklearnModel`` wrapping a fitted classifier.
    data : Data
        Training data with metadata (feature names, types, constraints).

    References
    ----------
    Author, A. (Year). Paper Title. Conference/Journal.
    """

    def __init__(self, model: BaseModel, data: BaseData, *args: object, **kwargs: object) -> None:
        # 1. Validate model type (if your method only supports specific backends)
        if not isinstance(model, SklearnModel):
            message = "FooClassifierExplainer requires a SklearnModel."
            raise ConfigurationError(
                message=message,
                param="model",
                config={"expected": "SklearnModel", "received": type(model).__name__},
                hint="Wrap your sklearn classifier with `SklearnModel(model=...)`.",
                source="FooClassifierExplainer.__init__",
            )

        # 2. Validate data type
        if not isinstance(data, Data):
            message = "FooClassifierExplainer requires data to be an instance of Data."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "Data", "received": type(data).__name__},
                source="FooClassifierExplainer.__init__",
            )

        # 3. Call super().__init__ LAST — it triggers _create_explainer()
        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(
        self,
        model: SklearnModel,
        data: BaseData,
        *args: object,
        **kwargs: object,
    ) -> Any:
        """Instantiate the underlying algorithm.

        Called automatically by ``BaseExplainer.__init__()``.
        The return value is stored as ``self.explainer``.
        """
        # Initialize your method's internal class here.
        # Example: return FooAlgorithm(model=model.model, data=data.data)
        ...

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        """Generate counterfactual explanations.

        Called by the public ``generate_counterfactuals()`` after validation.
        Must handle both single-row and multi-row inputs.
        """
        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T

        # Single instance → return a single Counterfactual
        if sample.shape[0] == 1:
            result = self._explain_single(sample.iloc[0])
            if result is None:
                raise NoCounterfactualsFoundError()
            return result

        # Multiple instances → return list[Counterfactual]
        cf_list: list[Counterfactual] = []
        for _, row in sample.iterrows():
            result = self._explain_single(row)
            if result is not None:
                cf_list.append(result)

        if not cf_list:
            raise NoCounterfactualsFoundError()

        return cf_list

    def _explain_single(self, row: pd.Series) -> Counterfactual | None:
        """Run the algorithm for one instance. Return None if no CF found."""
        # Your algorithm logic here.
        # Must return a Counterfactual object or None.
        ...

    # Optional: override _validate_sample for method-specific input checks
    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        """Add method-specific validation (called before _generate_counterfactuals)."""
        pass
```

#### Key Rules

1. **Call `super().__init__()` last** in your constructor. It calls `_create_explainer()`, so any validation or state setup must happen before it.

2. **`_create_explainer()`** runs once during initialization. The return value is accessible via `self.explainer`. Use it to set up any reusable internal state.

3. **`_generate_counterfactuals()`** receives pre-validated input. It must:
   - Handle both single-row and multi-row DataFrames.
   - Return a single `Counterfactual` for one instance, `list[Counterfactual]` for multiple.
   - Return results for instances where CFs were found, even if some failed.
   - Raise `NoCounterfactualsFoundError` **only** when zero CFs were found across all instances.

4. **`_validate_sample()`** is optional. Override it to add input checks specific to your method (e.g., rejecting non-numeric columns). It is called automatically before `_generate_counterfactuals()`.

#### Constructing `Counterfactual` Objects

```python
from celia.counterfactuals import Counterfactual

cf = Counterfactual(
    original_instance=row.to_frame().T,       # pd.DataFrame, single row
    counterfactual_instance=cf_df,             # pd.DataFrame, one or more rows
    original_prediction=current_prediction,    # int, float, or str
    counterfactual_prediction=cf_prediction,   # int, float, str, or list thereof
)
```

Rules:
- Both `original_instance` and `counterfactual_instance` must have the **same columns** (feature columns only, **no target column**).
- `original_instance` is always a single row.
- `counterfactual_instance` can be one or more rows.
- If `counterfactual_prediction` is a list, it must have one entry per counterfactual row.

See `src/celia/counterfactuals/counterfactuals.py` for full validation details.

#### Integrating an External Package

There are three approaches depending on the availability and state of the method's code:

| Approach | When to use | Example                                                                                     |
|----------|-------------|---------------------------------------------------------------------------------------------|
| **Install compatible package** | The method has a public Python package that works with CELIA's dependencies | e.g. OCEAN uses `oceanpy>=2.0.4` as a regular dependency in `pyproject.toml`                |
| **Fork and update** | The method has a package but it's outdated or incompatible | e.g. DiCE and CERTIFAI are installed from forks via `[tool.uv.sources]` in `pyproject.toml` |
| **Embed code directly** | The method has code but no package, or you want full control | e.g. GRACE implements the algorithm class directly in `src/celia/explainers/grace/grace.py` |

For the **fork approach**, add the dependency to `pyproject.toml`:

```toml
# In [project.dependencies]
dependencies = [
    "foo-method",
    # ...
]

# In [tool.uv.sources]
[tool.uv.sources]
foo-method = { git = "https://github.com/youruser/foo-method.git", rev = "main" }
```

#### Torch-Only Methods

If your method requires PyTorch:

1. Guard torch imports with `TYPE_CHECKING`:
   ```python
   from __future__ import annotations
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from torch import Tensor, nn
   ```

2. Validate that the model is a `TorchModel` in `__init__`.

3. Use the `@requires_torch_class` decorator if the **entire class** depends on torch (see `src/celia/_utils/dependencies.py`).

4. Reference `src/celia/explainers/grace/grace.py` as the canonical torch-only example.

### Step 3: Register the Explainer

Add your explainer to `src/celia/explainers/__init__.py`:

```python
from .foo import FooClassifierExplainer

__all__ = [
    # ... existing entries ...
    "FooClassifierExplainer",
]
```

### Step 4: Write Unit Tests

Create `tests/explainers/test_foo.py`. CELIA provides a **test mixin** that gives you a comprehensive test suite with minimal setup.

#### Minimal Test File

```python
from tests.explainers.classifier_test_suite import ClassifierExplainerTests
from celia.explainers.foo import FooClassifierExplainer


class TestFooClassifier(ClassifierExplainerTests):
    explainer_class = FooClassifierExplainer
    explainer_kwargs = {}          # Extra kwargs for the constructor
    generate_kwargs = {}           # Extra kwargs for generate_counterfactuals()

    # Set capability flags based on what your method supports
    supports_sklearn = True
    supports_torch = False
    supports_immutable_features = True
    supports_feasible_values = True
    supports_categorical_features = False
    requires_encoded_data = False
```

That's it. The mixin provides all core tests automatically.

#### Capability Flags Reference

| Flag | Default | Tests Provided |
|------|---------|----------------|
| *(always run)* | -- | `test_valid_initialization`, `test_invalid_data_raises_configuration_error`, `test_counterfactuals_exclude_target_column`, `test_no_counterfactuals_found_raises_error`, `test_single_instance_returns_single_counterfactual`, `test_multiple_instances_returns_list` |
| `supports_sklearn` | `True` | `test_sklearn_model_compatibility` |
| `supports_torch` | `False` | `test_torch_model_compatibility` |
| `rejects_sklearn` | `False` | `test_rejects_unsupported_sklearn_model` |
| `rejects_torch` | `False` | `test_rejects_unsupported_torch_model` |
| `supports_immutable_features` | `False` | `test_immutable_features_unchanged` |
| `supports_feasible_values` | `False` | `test_feasible_values_respected` |
| `supports_categorical_features` | `False` | `test_categorical_features_valid` |
| `requires_encoded_data` | `False` | `test_encoded_data_required` |

Only set a flag to `True` if your method **actually implements** that capability.

#### Overriding Model Helpers

Some methods need specific model types (e.g., OCEAN requires tree ensembles, not any sklearn classifier). Override `_get_model()` and `_get_dummy_model()`:

```python
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from celia.model import BaseModel, SklearnModel


class TestFooClassifier(ClassifierExplainerTests):
    explainer_class = FooClassifierExplainer
    # ... flags ...

    def _get_model(self, df: pd.DataFrame, request: pytest.FixtureRequest) -> BaseModel:
        """Override: use RandomForestClassifier instead of DecisionTreeClassifier."""
        X = df.drop(columns=["target"])
        y = df["target"]
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        rf.fit(X, y)
        return SklearnModel(rf)

    def _get_dummy_model(self, df: pd.DataFrame, request: pytest.FixtureRequest) -> BaseModel | None:
        """Return None if DummyClassifier is incompatible with the method."""
        return None
```

#### Overriding Core Tests

If a core test's default logic doesn't apply to your method, override it. For example, OCEAN overrides `test_no_counterfactuals_found_raises_error` because `DummyClassifier` isn't a tree ensemble:

```python
from celia.errors import NoCounterfactualsFoundError
from tests.explainers.classifier_test_suite import _make_public_data


class TestFooClassifier(ClassifierExplainerTests):
    # ... config ...

    def test_no_counterfactuals_found_raises_error(self, dummy_classification_dataframe, request):
        """Override: make all features immutable to force infeasibility."""
        model = self._get_model(dummy_classification_dataframe, request)
        all_features = dummy_classification_dataframe.drop(columns=["target"]).columns.tolist()
        public_data, X = _make_public_data(
            dummy_classification_dataframe,
            immutable=all_features,
        )

        explainer = self.explainer_class(model=model, data=public_data, **self.explainer_kwargs)
        sample = X.iloc[[0]]

        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample, **self.generate_kwargs)
```

#### Adding Method-Specific Tests

Add any additional tests as regular methods on the class:

```python
class TestFooClassifier(ClassifierExplainerTests):
    # ... mixin config ...

    def test_custom_parameter_validation(self, dummy_classification_dataframe, request):
        """Test that invalid custom_param raises ConfigurationError."""
        ...

    def test_specific_algorithm_behavior(self, dummy_classification_dataframe, request):
        """Test algorithm-specific behavior."""
        ...
```

#### Available Fixtures

These shared fixtures are available in all test files:

| Fixture | Source | Description |
|---------|--------|-------------|
| `dummy_classification_dataframe` | `tests/conftest.py` | 8-row DataFrame with 3 numeric features and binary target |
| `dummy_classification_dataframe_with_categories` | `tests/conftest.py` | 5-row DataFrame with 2 numeric + 1 categorical feature |
| `celia_public_data_classification` | `tests/conftest.py` | `Data` wrapping the numeric classification data |
| `model_trained_classifier` | `tests/conftest.py` | `SklearnModel` wrapping a `DummyClassifier` |
| `model_trained_classifier_stratified` | `tests/conftest.py` | `SklearnModel` wrapping a `DecisionTreeClassifier` |
| `torch_classification_model` | `tests/explainers/conftest.py` | Trained `TorchModel` (auto-skipped if torch unavailable) |

When using `ClassifierExplainerTests`, you typically don't need fixtures directly -- the mixin's internal helpers (`_make_public_data`, `_make_sklearn_model`) handle data and model creation. Use fixtures for additional standalone tests.

### Step 5: Run Quality Checks

These are the same checks CI runs on every PR:

```bash
# Lint (must pass with zero errors)
ruff check src/

# Format check (must produce no changes)
ruff format --check src/

# Auto-format if needed
ruff format src/

# Type check
ty check src/celia

# Run all tests
pytest tests/ -v

# Run only your method's tests during development
pytest tests/explainers/test_foo.py -v
```

---

## Code Style Reference

### Error Handling

Ruff's `EM` rule requires assigning error messages to a variable before `raise`:

```python
# Correct
message = f"Invalid parameter value: {value}"
raise ConfigurationError(
    message=message,
    param="my_param",
    hint="Use a value between 0 and 1.",
    config={"received": value},
    source="FooClassifierExplainer.__init__",
)

# Wrong — inline string literal
raise ConfigurationError(f"Invalid parameter value: {value}")
```

CELIA's error hierarchy (see `src/celia/errors.py`):
- `ConfigurationError` -- invalid user input (includes `message`, `param`, `hint`, `config`, `source`)
- `NoCounterfactualsFoundError` -- algorithm found zero CFs for any instance
- `MethodError` / `MethodValueError` -- internal algorithm errors

### Type Hints and Imports

```python
# Use PEP 604 union syntax (not Union[X, Y])
def foo(x: int | None) -> str | float:
    ...

# Use TYPE_CHECKING for heavy imports (torch, etc.)
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

# Absolute imports only — no relative imports
from celia.explainers import ClassifierExplainer   # correct
from ._base import ClassifierExplainer             # wrong
```

### Docstrings

Use NumPy-style format:

```python
def _explain_single(self, row: pd.Series) -> Counterfactual | None:
    """Run the algorithm for a single instance.

    Parameters
    ----------
    row : pd.Series
        Single instance to explain.

    Returns
    -------
    Counterfactual | None
        The counterfactual, or None if no feasible solution was found.

    Raises
    ------
    ConfigurationError
        If the instance has invalid feature values.
    """
```

### General Rules

- **Line length:** 120 characters
- **Formatter:** `ruff format` (double quotes, 4-space indent, magic trailing commas)
- **Private attributes/methods:** Prefix with `_`, expose via `@property` getters
- **No relative imports** -- use absolute imports only

---

## Git Workflow

```bash
# 1. Create a branch from dev, named after your method
git checkout dev
git pull origin dev
git checkout -b foo

# 2. Implement and test your method (Steps 1–5 above)

# 3. Before pushing, run all quality checks
ruff check src/ && ruff format --check src/ && ty check src/celia && pytest tests/ -v

# 4. Push and open a PR targeting the dev branch
git push -u origin foo
# Open PR: foo → dev
```

CI runs ruff and pytest automatically on PRs to `main` and `dev` (see `.github/workflows/ci.yml`).

---

## Checklist Before Submitting a PR

- [ ] Explainer class inherits from `ClassifierExplainer`
- [ ] Constructor validates `model` and `data` types, raising `ConfigurationError(param=...)`
- [ ] Implements `_create_explainer()` and `_generate_counterfactuals()`
- [ ] `_validate_sample()` overridden if method has input-specific constraints
- [ ] Returns `Counterfactual` for single instance, `list[Counterfactual]` for multiple
- [ ] Raises `NoCounterfactualsFoundError` when no CFs found for any instance
- [ ] `Counterfactual` objects contain feature columns only (no target column)
- [ ] Method `__init__.py` exports the explainer class
- [ ] `src/celia/explainers/__init__.py` updated with import and `__all__` entry
- [ ] Test class inherits from `ClassifierExplainerTests` with correct capability flags
- [ ] Method-specific tests added for custom parameters or behavior
- [ ] `ruff check src/` passes
- [ ] `ruff format --check src/` reports no changes
- [ ] `ty check src/celia` passes
- [ ] `pytest tests/ -v` passes
- [ ] Branch is based on `dev`, PR targets `dev`

---

## Reference Files

| File                                           | Purpose |
|------------------------------------------------|---------|
| `src/celia/explainers/_base.py`                | Base classes (`ClassifierExplainer`, `RegressorExplainer`) |
| `src/celia/explainers/ocean/ocean.py`          | Reference classifier explainer implementation |
| `tests/explainers/classifier_test_suite.py`    | Test mixin with capability flags |
| `tests/explainers/test_ocean.py`               | Reference test file using the mixin |
| `src/celia/explainers/__init__.py`             | Explainer registration (update when adding methods) |
| `src/celia/errors.py`                          | Error hierarchy |
| `src/celia/counterfactuals/counterfactuals.py` | `Counterfactual` dataclass |
| `src/celia/explainers/grace/grace.py`          | Reference for torch-only / embedded-code methods |
| `src/celia/_utils/dependencies.py`             | `@requires_torch_class` decorator |
