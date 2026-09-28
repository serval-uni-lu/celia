"""Explainer-specific fixtures that complement the root ``tests/conftest.py``."""

from __future__ import annotations

import pytest
import pandas as pd

from celia.data import Data

from tests.conftest import build_feasible_values


# ---------------------------------------------------------------------------
# TorchModel fixture — skipped entirely when torch is not installed
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_classification_model(dummy_classification_dataframe):
    """A simple 2-layer ``TorchModel`` trained on the classification dataset.

    Automatically skipped if PyTorch is not installed.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    from celia.model import TorchModel

    X = dummy_classification_dataframe.drop(columns=["target"])
    y = dummy_classification_dataframe["target"]

    n_features = X.shape[1]
    n_classes = int(y.nunique())

    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 16)
            self.fc2 = nn.Linear(16, n_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    torch.manual_seed(42)
    model = SimpleClassifier()

    # Quick training so predictions are non-trivial
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return TorchModel(model)


# ---------------------------------------------------------------------------
# Additional data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def celia_data_classification_no_immutable(dummy_classification_dataframe) -> Data:
    """Classification ``Data`` with **no** immutable features.

    Useful when a method needs full freedom to find counterfactuals.
    """
    data = dummy_classification_dataframe.drop(columns=["target"])
    targets = dummy_classification_dataframe["target"]
    return Data(
        data=data,
        targets=targets,
        target_name="target",
        column_names=data.columns.tolist(),
        continuous_column_names=data.columns.tolist(),
        categorical_column_names=[],
        immutable_column_names=[],
        feasible_values=build_feasible_values(dummy_classification_dataframe),
    )
