"""FastAR classifier explainer wrapper for CELIA.

FastAR is a reinforcement-learning-based counterfactual explainer that uses
PPO (stable-baselines3) to learn an amortized recourse policy. It fits at
construction time and then generates counterfactuals cheaply.

FastAR requires that all features are scaled to [-1, 1]. It is the user's
responsibility to provide pre-scaled data and to inverse-transform the
counterfactuals back to the original space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastar import Explanation, FastAR, FeatureSpec

from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.data._base import BaseData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers._base import ClassifierExplainer
from celia.model import BaseModel


class FastARClassifierExplainer(ClassifierExplainer):
    """FastAR: Amortized counterfactual recourse via reinforcement learning.

    Trains a PPO policy at construction time that learns to perturb instances
    until a black-box classifier flips to the target class. Supports immutable,
    monotonic increasing, and correlated feature constraints.

    Parameters
    ----------
    model : BaseModel
        A CELIA model wrapping any classifier with ``predict_proba``.
    data : PublicData
        Training data with metadata (feature names, types, constraints).
        Must contain numeric features scaled to ``[-1, 1]``.
    target_class : int
        The desired class label (encoded) the agent tries to reach.
    total_timesteps : int
        Number of environment steps for PPO training.
    policy_path : str | Path | None
        Path to a previously saved policy. If provided, the policy is loaded
        instead of training from scratch.
    **kwargs
        Forwarded to ``FastAR()`` constructor (e.g. ``dist_lambda``,
        ``max_episode_steps``, ``policy_kwargs``, ``seed``).

    References
    ----------
    Verma, S., Hines, K., & Dickerson, J. P. (2022, June).
    Amortized generation of sequential algorithmic recourses for black-box models.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 8, pp. 8512-8519).
    """

    def __init__(
        self,
        model: BaseModel,
        data: BaseData,
        target_class: int = 1,
        total_timesteps: int = 1_000_000,
        policy_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(data, PublicData):
            message = "FastARClassifierExplainer requires data to be an instance of PublicData."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "PublicData", "received": type(data).__name__},
                source="FastARClassifierExplainer.__init__",
            )

        if not isinstance(target_class, int):
            message = f"'target_class' must be an int, got {type(target_class).__name__}"
            raise ConfigurationError(
                message=message,
                param="target_class",
                config={"provided_type": type(target_class).__name__},
                hint="Provide the encoded target class label as an integer.",
                source="FastARClassifierExplainer.__init__",
            )

        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            message = f"'total_timesteps' must be a positive integer, got {total_timesteps}"
            raise ConfigurationError(
                message=message,
                param="total_timesteps",
                config={"provided": total_timesteps},
                hint="Provide a positive integer for the PPO training budget.",
                source="FastARClassifierExplainer.__init__",
            )

        if policy_path is not None:
            policy_path = Path(policy_path)
            if not policy_path.exists():
                message = f"Policy path does not exist: {policy_path}"
                raise ConfigurationError(
                    message=message,
                    param="policy_path",
                    config={"path": str(policy_path)},
                    hint="Provide a valid path to a saved PPO policy (.zip file).",
                    source="FastARClassifierExplainer.__init__",
                )

        self._target_class = target_class
        self._total_timesteps = total_timesteps
        self._policy_path = policy_path

        super().__init__(model, data, **kwargs)

    @property
    def target_class(self) -> int:
        """The target class label the agent tries to reach."""
        return self._target_class

    def _create_explainer(
        self,
        model: BaseModel,
        data: BaseData,
        **kwargs: Any,
    ) -> FastAR:
        """Build FeatureSpec from PublicData, create and train FastAR."""
        public_data: PublicData = data

        feature_spec = FeatureSpec(
            columns=public_data.column_names,
            continuous=public_data.continuous_column_names or (),
            immutable=public_data.immutable_column_names or (),
            monotonic_increasing=public_data.monotonic_increasing_column_names or (),
            correlated=public_data.correlated_features or (),
        )

        fastar = FastAR(
            classifier=model,
            feature_spec=feature_spec,
            target_class=self._target_class,
            **kwargs,
        )

        x_train = public_data.data.to_numpy().astype(np.float32)

        if self._policy_path is not None:
            fastar.load_policy(self._policy_path, x_train)
        else:
            fastar.fit(x_train, total_timesteps=self._total_timesteps)

        return fastar

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        """Generate counterfactual explanations using the trained FastAR policy."""
        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T

        sample = sample[self.data.column_names]
        cf_list: list[Counterfactual] = []

        for idx in sample.index:
            row = sample.loc[[idx]]
            row_numpy = row.to_numpy().astype(np.float32).reshape(-1)

            explanation: Explanation = self.explainer.explain(row_numpy)

            if not explanation.success:
                continue

            cf_array = explanation.counterfactual.reshape(1, -1)
            cf_df = pd.DataFrame(cf_array, columns=self.data.column_names, index=[idx])

            original_prediction = int(np.argmax(self.model.predict_proba(row)))
            cf_prediction = int(np.argmax(self.model.predict_proba(cf_df)))

            cf_list.append(
                Counterfactual(
                    original_instance=row,
                    counterfactual_instance=cf_df,
                    original_prediction=original_prediction,
                    counterfactual_prediction=cf_prediction,
                )
            )

        if not cf_list:
            raise NoCounterfactualsFoundError()

        if len(cf_list) == 1 and sample.shape[0] == 1:
            return cf_list[0]

        return cf_list

    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        """Reject samples containing non-numeric columns."""
        frame = sample.to_frame().T if isinstance(sample, pd.Series) else sample
        non_numeric = frame.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = f"FastAR requires all features to be numeric. Non-numeric columns found: {non_numeric}"
            raise ConfigurationError(
                message=message,
                param="sample",
                config={"non_numeric_columns": non_numeric},
                hint="Encode categorical features and scale all values to [-1, 1] before passing to FastAR.",
                source="FastARClassifierExplainer._validate_sample",
            )

    def save_policy(self, path: str | Path) -> None:
        """Save the trained PPO policy to disk.

        Parameters
        ----------
        path : str | Path
            Destination file path. Stable-Baselines3 appends ``.zip`` if not present.
        """
        self.explainer.save(path)
