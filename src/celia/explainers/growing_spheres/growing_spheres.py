from __future__ import annotations

import numpy as np
import pandas as pd

from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.data._base import BaseData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.explainers.growing_spheres.gsg import GSG
from celia.model import BaseModel


class GSGClassifierExplainer(ClassifierExplainer):
    """Growing Spheres Generation classifier explainer.

    Generates counterfactual explanations by iteratively expanding a
    hyperspherical shell around an input instance until a candidate that
    flips the model's predicted class is found.

    GSG supports any model backend (sklearn, torch) that exposes
    ``predict_proba``.  All features must be numeric — categorical
    features must be one-hot encoded before use.

    Parameters
    ----------
    model : BaseModel
        A CELIA model wrapper (``SklearnModel`` or ``TorchModel``).
    data : PublicData
        Training data with metadata (feature names, types, constraints).
        All features must be numeric.

    References
    ----------
    Laugel, T., Lesot, M.-J., Marsala, C., Renard, X., & Detyniecki, M.
    (2018). Comparison-based Inverse Classification for Interpretability
    in Machine Learning. IPMU.
    """

    def __init__(self, model: BaseModel, data: BaseData, *args: object, **kwargs: object) -> None:
        if not isinstance(data, PublicData):
            message = "GSGClassifierExplainer requires data to be an instance of PublicData."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "PublicData", "received": type(data).__name__},
                source="GSGClassifierExplainer.__init__",
            )

        non_numeric = data.data.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = (
                f"GSG requires all features to be numeric. "
                f"Found non-numeric columns: {non_numeric}. "
                f"Categorical features must be one-hot encoded."
            )
            raise ConfigurationError(
                message=message,
                param="data",
                hint="One-hot encode categorical features before passing data to GSG.",
                source="GSGClassifierExplainer.__init__",
            )

        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(
        self,
        model: BaseModel,
        data: BaseData,
        *args,
        **kwargs,
    ) -> GSG:
        data_public: PublicData = data  # type: ignore[assignment]

        feature_order = data_public.column_names
        immutable_features = data_public.immutable_column_names or []
        immutable_set = set(immutable_features)
        mutable_features = [c for c in feature_order if c not in immutable_set]
        continuous_features = data_public.continuous_column_names or []
        binary_features = [
            col
            for col in feature_order
            if data_public.data[col].nunique() == 2  # noqa: PLR2004
        ]

        n_samples = kwargs.get("n_samples", 1000)
        p_norm = kwargs.get("p_norm", 2)
        step_size = kwargs.get("step_size", 0.2)
        max_iterations = kwargs.get("max_iterations", 1000)
        max_shrink_iterations = kwargs.get("max_shrink_iterations", 50)
        seed = kwargs.get("seed", 42)

        return GSG(
            model=model,
            mutable_features=mutable_features,
            immutable_features=immutable_features,
            continuous_features=continuous_features,
            binary_features=binary_features,
            feature_order=feature_order,
            n_samples=n_samples,
            p_norm=p_norm,
            step_size=step_size,
            max_iterations=max_iterations,
            max_shrink_iterations=max_shrink_iterations,
            seed=seed,
        )

    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        """Reject samples containing non-numeric columns."""
        frame = sample.to_frame().T if isinstance(sample, pd.Series) else sample
        non_numeric = frame.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = (
                f"GSG requires all features to be numeric. "
                f"Found non-numeric columns: {non_numeric}. "
                f"Categorical features must be one-hot encoded."
            )
            raise ConfigurationError(
                message=message,
                param="sample",
                hint="One-hot encode categorical features before passing data to GSG.",
                source="GSGClassifierExplainer._validate_sample",
            )

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T

        # Ensure column order matches feature_order
        sample = sample[self.data.column_names]

        cf_df = self.explainer.generate(sample)

        # Detect rows where no CF was found (all NaN)
        nan_mask = cf_df.isna().all(axis=1)

        if nan_mask.all():
            raise NoCounterfactualsFoundError()

        # Build Counterfactual objects for successful rows
        cf_list: list[Counterfactual] = []
        for idx in sample.index:
            if nan_mask.loc[idx]:
                continue

            original_row = sample.loc[[idx]]
            cf_row = cf_df.loc[[idx]]

            original_prediction = int(np.argmax(self.model.predict_proba(original_row)))
            cf_prediction = int(np.argmax(self.model.predict_proba(cf_row)))

            cf_list.append(
                Counterfactual(
                    original_instance=original_row,
                    counterfactual_instance=cf_row,
                    original_prediction=original_prediction,
                    counterfactual_prediction=cf_prediction,
                )
            )

        if not cf_list:
            raise NoCounterfactualsFoundError()

        if sample.shape[0] == 1:
            return cf_list[0]

        return cf_list
