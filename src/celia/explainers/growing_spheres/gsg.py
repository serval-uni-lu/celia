from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy import linalg


class GSG:
    """Growing Spheres Generation.

    Generates CFs by iteratively expanding a hyperspherical
    shell around an input instance until a candidate that flips the
    model's predicted class is found.

    Based on Laugel et al. (2018), "Comparison-based Inverse Classification
    for Interpretability in Machine Learning".

    Parameters
    ----------
    model : Any
        Classifier exposing a ``predict_proba(X)`` method that returns
        class probabilities as a 2-D array of shape (n_instances, n_classes).
    mutable_features : list[str]
        Feature names that may be modified during search.
    immutable_features : list[str]
        Feature names that must remain fixed.
    continuous_features : list[str]
        Names of continuous-valued features.
    binary_features : list[str]
        Names of binary-valued features.
    feature_order : list[str]
        Canonical column ordering expected by the model.
    n_samples : int, optional
        Number of candidates sampled per iteration (default: 1000).
    p_norm : {1, 2}, optional
        Norm for distance computation (default: 2).
    step_size : float, optional
        Radial increment per iteration (default: 0.2).
    max_iterations : int, optional
        Maximum number of search iterations (default: 1000).
    max_shrink_iterations : int, optional
        Maximum number of halving steps in the shrinking phase
        (default: 50).
    """

    def __init__(
        self,
        model: Any,
        mutable_features: list[str],
        immutable_features: list[str],
        continuous_features: list[str],
        binary_features: list[str],
        feature_order: list[str],
        n_samples: int = 1000,
        p_norm: Literal[1, 2] = 2,
        step_size: float = 0.2,
        max_iterations: int = 1000,
        max_shrink_iterations: int = 50,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.mutable_features = mutable_features
        self.immutable_features = immutable_features
        self.continuous_features = continuous_features
        self.binary_features = binary_features
        self.feature_order = feature_order
        self.n_samples = n_samples
        self.p_norm = p_norm
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.max_shrink_iterations = max_shrink_iterations
        self.rng = np.random.default_rng(seed)

        binary_set = set(binary_features)
        self.mutable_continuous: list[str] = [f for f in mutable_features if f not in binary_set]
        self.mutable_binary: list[str] = [f for f in mutable_features if f in binary_set]

    def _sample_hypersphere(
        self,
        instance: np.ndarray,
        upper_bound: float,
        lower_bound: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample candidate CFs uniformly from a hyperspherical shell.

        Points are sampled uniformly at random on a hypersphere and scaled
        to lie within the shell defined by [lower_bound, upper_bound). Based
        on Pawelczyk, Broelemann & Kasneci (2020), "Learning Counterfactual
        Explanations for Tabular Data", and insights from Wolfram MathWorld's
        Hypersphere Point Picking.

        Parameters
        ----------
        instance : np.ndarray
            Reference point array with shape (n_samples, n_features).
        upper_bound : float
            Outer radius of the sampling shell.
        lower_bound : float
            Inner radius of the sampling shell.

        Returns
        -------
        candidates : np.ndarray
            Candidate CF points with shape (n_samples, n_features).
        distances : np.ndarray
            Sampled radial distances with shape (n_samples,).
        """
        random_directions = self.rng.standard_normal((self.n_samples, instance.shape[1]))
        radial_distances = self.rng.uniform(lower_bound, upper_bound, size=self.n_samples)
        norms = linalg.norm(random_directions, ord=self.p_norm, axis=1)
        scale_factors = np.divide(radial_distances, norms).reshape(-1, 1)
        perturbations = np.multiply(random_directions, scale_factors)
        candidates = instance + perturbations

        return candidates, radial_distances

    def _sample_candidates(
        self,
        mutable_continuous_values: np.ndarray,
        immutable_values: np.ndarray,
        upper_bound: float,
        lower_bound: float,
    ) -> pd.DataFrame:
        """Sample a batch of candidate CFs within a hyperspherical shell.

        Combines continuous perturbations (hypersphere sampling) with
        random binary feature values, then assembles them with the
        immutable features in the correct column order.

        Parameters
        ----------
        mutable_continuous_values : np.ndarray
            Replicated continuous mutable values, shape (n_samples, n_continuous).
        immutable_values : np.ndarray
            Replicated immutable values, shape (n_samples, n_immutable).
        upper_bound : float
            Outer radius of the sampling shell.
        lower_bound : float
            Inner radius of the sampling shell.

        Returns
        -------
        pd.DataFrame
            Candidate CFs with columns in ``feature_order``.
        """
        continuous_candidates, _ = self._sample_hypersphere(mutable_continuous_values, upper_bound, lower_bound)

        binary_candidates = self.rng.binomial(n=1, p=0.5, size=self.n_samples * len(self.mutable_binary)).reshape(
            self.n_samples, -1
        )

        candidates = pd.DataFrame(
            np.c_[immutable_values, continuous_candidates, binary_candidates],
            columns=self.immutable_features + self.mutable_continuous + self.mutable_binary,
        )
        return candidates[self.feature_order]

    def _has_enemy(self, candidates: pd.DataFrame, instance_label: int) -> bool:
        """Check whether any candidate flips the predicted class.

        Parameters
        ----------
        candidates : pd.DataFrame
            Candidate CFs.
        instance_label : int
            Predicted class index of the original instance.

        Returns
        -------
        bool
            True if at least one candidate has a different predicted class.
        """
        predicted_labels = np.argmax(self.model.predict_proba(candidates.values), axis=1)
        return bool(np.any(predicted_labels != instance_label))

    def _shrink(
        self,
        mutable_continuous_values: np.ndarray,
        immutable_values: np.ndarray,
        instance_label: int,
    ) -> float:
        """Phase 1: shrink the sphere to find the tightest CF-containing radius.

        Starts with a sphere of radius ``step_size`` and halves it
        repeatedly while enemies (CFs) still exist among the samples,
        up to ``max_shrink_iterations`` halvings.

        Parameters
        ----------
        mutable_continuous_values : np.ndarray
            Replicated continuous mutable values.
        immutable_values : np.ndarray
            Replicated immutable values.
        instance_label : int
            Predicted class index of the original instance.

        Returns
        -------
        float
            The smallest radius η where CFs were no longer found,
            to be used as the starting point for the growing phase.
        """
        radius = self.step_size

        candidates = self._sample_candidates(mutable_continuous_values, immutable_values, radius, 0)

        for _ in range(self.max_shrink_iterations):
            if not self._has_enemy(candidates, instance_label):
                break
            radius /= 2
            candidates = self._sample_candidates(mutable_continuous_values, immutable_values, radius, 0)

        return radius

    def _grow(
        self,
        mutable_continuous_values: np.ndarray,
        immutable_values: np.ndarray,
        original_values: np.ndarray,
        instance_label: int,
        initial_radius: float,
    ) -> np.ndarray:
        """Phase 2: grow outward from the shrink boundary to find the closest CF.

        Expands a hyperspherical shell starting from the radius found by
        the shrinking phase until a CF is found or ``max_iterations`` is
        reached.

        Parameters
        ----------
        mutable_continuous_values : np.ndarray
            Replicated continuous mutable values.
        immutable_values : np.ndarray
            Replicated immutable values.
        original_values : np.ndarray
            Replicated original instance values for distance computation.
        instance_label : int
            Predicted class index of the original instance.
        initial_radius : float
            Starting radius η returned by the shrinking phase.

        Returns
        -------
        np.ndarray
            Closest CF found, or a NaN-filled array if none was found.
        """
        lower_bound = initial_radius
        upper_bound = 2 * initial_radius
        closest_cf = np.full(original_values.shape[1], np.nan)

        for _ in range(self.max_iterations):
            candidates = self._sample_candidates(mutable_continuous_values, immutable_values, upper_bound, lower_bound)

            if self.p_norm == 1:
                distances = np.abs(candidates.to_numpy() - original_values).sum(axis=1)
            elif self.p_norm == 2:
                distances = np.square(candidates.to_numpy() - original_values).sum(axis=1)
            else:
                msg = f"p_norm must be 1 or 2, got {self.p_norm}"
                raise ValueError(msg)

            predicted_labels = np.argmax(self.model.predict_proba(candidates.to_numpy()), axis=1)
            flipped_mask = np.where(predicted_labels != instance_label)
            flipped_candidates = candidates.to_numpy()[flipped_mask]
            flipped_distances = distances[flipped_mask]

            if len(flipped_distances) > 0:
                closest_cf = flipped_candidates[np.argmin(flipped_distances)]
                break

            lower_bound = upper_bound
            upper_bound = lower_bound + initial_radius

        return closest_cf

    def _find_closest_cf(self, instance: pd.Series) -> np.ndarray:
        """Find the closest CF for a single instance.

        Runs the full two-phase Growing Spheres algorithm: first shrinks
        to find the tightest CF-containing radius, then grows outward
        from that boundary.

        Parameters
        ----------
        instance : pd.Series
            Single input instance.

        Returns
        -------
        np.ndarray
            Closest CF found, or a NaN-filled array if none was found
            within ``max_iterations``.
        """
        immutable_values = np.repeat(
            instance[self.immutable_features].to_numpy().reshape(1, -1), self.n_samples, axis=0
        )
        original_values = np.repeat(instance.to_numpy().reshape(1, -1), self.n_samples, axis=0)
        mutable_continuous_values = np.repeat(
            instance[self.mutable_continuous].to_numpy().reshape(1, -1), self.n_samples, axis=0
        )

        instance_label = np.argmax(self.model.predict_proba(instance.to_numpy().reshape(1, -1)))

        initial_radius = self._shrink(mutable_continuous_values, immutable_values, instance_label)

        return self._grow(
            mutable_continuous_values,
            immutable_values,
            original_values,
            instance_label,
            initial_radius,
        )

    def generate(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """Generate CFs for one or more input instances.

        Parameters
        ----------
        factuals : pd.DataFrame
            Input instances, one per row, with columns matching
            ``feature_order``.

        Returns
        -------
        pd.DataFrame
            DataFrame of CFs with the same index and column order
            as ``factuals``. Rows where no CF was found contain NaN.
        """
        counterfactuals = []
        for _, row in factuals.iterrows():
            counterfactuals.append(self._find_closest_cf(row))

        return pd.DataFrame(counterfactuals, index=factuals.index, columns=self.feature_order)
