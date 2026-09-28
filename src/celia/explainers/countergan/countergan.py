from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from countergan import CounterGAN
from sklearn.model_selection import train_test_split

from celia.counterfactuals import Counterfactual
from celia.data import BaseData, Data
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.model import BaseModel, TorchModel


class CounterGANClassifierExplainer(ClassifierExplainer):
    """CounterGAN classifier explainer.

    This method trains a generative adversarial network (GAN) to produce counterfactuals.

    CounterGAN requires a ``TorchModel`` backend.  All features must be
    numeric — categorical features must be one-hot encoded before use.

    This method does not support feasible ranges.

    Parameters
    ----------
    model : TorchModel
        A CELIA ``TorchModel`` wrapper around a ``torch.nn.Module``.
    data : Data
        Training data with metadata (feature names, types, constraints).
        All features must be numeric.
    desired_class : int
        Target class index for counterfactuals. Must be present in
        ``data.targets``.
    strategy : str, default="countergan"
        One of ``"regular_gan"``, ``"countergan"``, or ``"countergan_wt"``.
    n_discriminator_steps : int, default=2
        Number of discriminator update steps per training iteration.
    n_generator_steps : int, default=4
        Number of generator update steps per training iteration.
    n_iterations : int, default=2000
        Total number of training iterations.
    backend : str or None, default=None
        CounterGAN backend: ``"torch"`` or ``"tensorflow"``. Auto-detected
        when ``None``.

    Raises
    ------
    ConfigurationError
        If ``model`` is not a ``TorchModel``, ``data`` is not a ``Data``
        instance, the data contains non-numeric columns, or
        ``desired_class`` is not found in the training targets.

    References
    ----------
    Nemirovsky, D., Thiebaut, N., Xu, Y., & Gupta, A.
    (2022, August). CounteRGAN: Generating counterfactuals for real-time
    recourse and interpretability using residual GANs. Uncertainty in
    Artificial Intelligence (pp. 1488-1497). PMLR.

    Note
    ----
    **Supported constraints:** immutable features.

    Examples
    --------
    >>> from celia import Data, TorchModel, CounterGANClassifierExplainer
    >>> explainer = CounterGANClassifierExplainer(
    ...     model=torch_model, data=data, desired_class=1,
    ... )
    >>> cf = explainer.generate_counterfactuals(sample)
    """

    def __init__(self, model: BaseModel, data: BaseData, *args: object, **kwargs: object) -> None:
        if not isinstance(data, Data):
            message = "CounterGANClassifierExplainer requires data to be an instance of Data."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "Data", "received": type(data).__name__},
                source="CounterGANClassifierExplainer.__init__",
            )

        non_numeric = data.data.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = (
                f"CounterGAN requires all features to be numeric. "
                f"Found non-numeric columns: {non_numeric}. "
                f"Categorical features must be one-hot encoded."
            )
            raise ConfigurationError(
                message=message,
                param="data",
                hint="One-hot encode categorical features before passing data to CounterGAN.",
                source="CounterGANClassifierExplainer.__init__",
            )

        if not isinstance(model, TorchModel):
            message = "CounterGANClassifierExplainer requires model to be a TorchModel instance."
            raise ConfigurationError(
                message=message,
                param="model",
                config={"expected": "TorchModel", "received": type(model).__name__},
                source="CounterGANClassifierExplainer.__init__",
            )

        self._validate_desired_class(data, **kwargs)

        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model: BaseModel, data: BaseData, *args, **kwargs) -> Any:
        immutable_cols = list(self.data.immutable_column_names or [])
        mutable_cols = [c for c in self.data.column_names if c not in set(immutable_cols)]
        # Store mutable-first ordering for use in _generate_counterfactuals
        self._ordered_columns: list[str] = mutable_cols + immutable_cols

        n_mutable = len(mutable_cols)
        n_classes = len(np.unique(self.data.targets))

        strategy = str(kwargs.get("strategy", "countergan"))
        n_discriminator_steps = int(kwargs.get("n_discriminator_steps", 2))
        n_generator_steps = int(kwargs.get("n_generator_steps", 4))
        n_iterations = int(kwargs.get("n_iterations", 2000))
        backend_val = kwargs.get("backend")
        backend = str(backend_val) if backend_val is not None else None

        gan = CounterGAN(
            strategy=strategy,
            classifier=model,
            n_mutable_features=n_mutable,
            n_discriminator_steps=n_discriminator_steps,
            n_generator_steps=n_generator_steps,
            n_iterations=n_iterations,
            desired_class=self.desired_class,
            number_of_classes=n_classes,
            backend=backend,
        )

        targets = self.data.targets
        y_arr = targets.to_numpy() if hasattr(targets, "to_numpy") else np.asarray(targets)
        y_ohe = np.eye(n_classes)[y_arr.astype(int)]

        x_data = self.data.data[self._ordered_columns].to_numpy().astype(np.float32)
        x_train, x_test, y_train, _ = train_test_split(x_data, y_ohe, test_size=0.2, random_state=42)
        gan.fit(x_train, y_train, x_test)
        return gan

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        frame = sample.to_frame().T if isinstance(sample, pd.Series) else sample
        is_single = isinstance(sample, pd.Series) or frame.shape[0] == 1

        # Reorder to mutable-first (same ordering used during fit)
        x_input = frame[self._ordered_columns].to_numpy().astype(np.float32)

        cf_array = self.explainer.generate_counterfactuals(x_input)

        if cf_array.size == 0:
            raise NoCounterfactualsFoundError()

        # Restore original column order so original_instance and
        # counterfactual_instance share the same column layout
        col_to_idx = {c: i for i, c in enumerate(self._ordered_columns)}
        restore_idx = [col_to_idx[c] for c in self.data.column_names]
        cf_restored = cf_array[:, restore_idx]
        orig_restored = frame[self.data.column_names].to_numpy()

        orig_preds = self.model.predict(orig_restored)
        cf_preds = self.model.predict(cf_restored)

        results: list[Counterfactual] = []
        for i in range(frame.shape[0]):
            if cf_preds[i] == orig_preds[i]:
                continue
            original_row = pd.DataFrame([orig_restored[i]], columns=self.data.column_names)
            cf_row = pd.DataFrame([cf_restored[i]], columns=self.data.column_names)
            results.append(
                Counterfactual(
                    original_instance=original_row,
                    counterfactual_instance=cf_row,
                    original_prediction=int(orig_preds[i]),
                    counterfactual_prediction=int(cf_preds[i]),
                )
            )

        if not results:
            message = "No counterfactuals found: the generated samples never reached the desired class."
            raise NoCounterfactualsFoundError(
                message=message,
                source="CounterGANClassifierExplainer._generate_counterfactuals",
            )

        return results[0] if is_single else results

    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        """Reject samples containing non-numeric columns."""
        frame = sample.to_frame().T if isinstance(sample, pd.Series) else sample
        non_numeric = frame.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = (
                f"CounterGANClassifierExplainer requires all features to be numeric. "
                f"Found non-numeric columns: {non_numeric}. "
                f"Categorical features must be one-hot encoded."
            )
            raise ConfigurationError(
                message=message,
                param="sample",
                hint="One-hot encode categorical features before passing data to CounterGANClassifierExplainer.",
                source="CounterGANClassifierExplainer._validate_sample",
            )

    def _validate_desired_class(self, data: Data, **kwargs: object) -> None:
        """Validate that 'desired_class' is provided and valid."""
        if "desired_class" not in kwargs:
            message = "CounterGANClassifierExplainer requires 'desired_class' to be specified as a keyword argument during initialization."
            raise ConfigurationError(
                message=message,
                param="desired_class",
                hint="Pass the target class index as 'desired_class' when initializing CounterGANClassifierExplainer.",
                source="CounterGANClassifierExplainer._validate_desired_class",
            )

        unique_classes = np.unique(data.targets)
        desired_class = kwargs["desired_class"]
        if desired_class not in unique_classes:
            message = (
                f"'desired_class' must be present in Data targets. "
                f"Expected values {unique_classes}. "
                f"Received: {desired_class}."
            )
            raise ConfigurationError(
                message=message,
                param="desired_class",
                hint="Ensure 'desired_class' is a valid class for the dataset.",
                source="CounterGANClassifierExplainer._validate_desired_class",
            )

        self.desired_class: int = desired_class
