from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from celia._utils.dependecies import requires_ocean_class
from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.data._base import BaseData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.model import BaseModel, SklearnModel

if TYPE_CHECKING:
    from ocean import MixedIntegerProgramExplainer as OceanMIPExplainer
    from ocean.abc import Mapper
    from ocean.feature import Feature


@requires_ocean_class
class OCEANClassifierExplainer(ClassifierExplainer):
    """OCEAN: Optimal Counterfactual Explanations in Tree Ensembles.

    Generates counterfactual explanations for tree-ensemble classifiers by
    formulating the search as a Mixed-Integer Program (MIP) solved by Gurobi.

    OCEAN supports Tree Ensembles from scikit-learn / XGBoost.

    Parameters
    ----------
    model : SklearnModel
        A CELIA ``SklearnModel`` wrapping a fitted tree-ensemble classifier.
    data : PublicData
        Training data with metadata (feature names, types, constraints).
        All features must be numeric.

    References
    ----------
    Parmentier, A., & Vidal, T. (2021). Optimal Counterfactual Explanations
    in Tree Ensembles. Proceedings of the 38th International Conference on
    Machine Learning (ICML).
    """

    _ocean_mapper: Mapper[Feature]
    _binary_kept_value: dict[str, Any]

    def __init__(self, model: BaseModel, data: BaseData, *args: object, **kwargs: object) -> None:
        if not isinstance(model, SklearnModel):
            message = "OCEANClassifierExplainer requires a SklearnModel wrapping a fitted tree-ensemble classifier."
            raise ConfigurationError(
                message=message,
                param="model",
                config={"expected": "SklearnModel", "received": type(model).__name__},
                hint="Wrap your sklearn tree-ensemble classifier with `SklearnModel(model=...)`",
                source="OCEANClassifierExplainer.__init__",
            )

        if not isinstance(data, PublicData):
            message = "OCEANClassifierExplainer requires data to be an instance of PublicData."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "PublicData", "received": type(data).__name__},
                source="OCEANClassifierExplainer.__init__",
            )

        non_numeric = data.data.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = (
                f"OCEAN requires all features to be numeric. "
                f"Found non-numeric columns: {non_numeric}. "
                f"Categorical features must be one-hot encoded."
            )
            raise ConfigurationError(
                message=message,
                param="data",
                hint="One-hot encode categorical features before passing data to OCEAN.",
                source="OCEANClassifierExplainer.__init__",
            )

        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(
        self,
        model: SklearnModel,
        data: BaseData,
        *args: object,
        **kwargs: object,
    ) -> OceanMIPExplainer:
        from ocean import MixedIntegerProgramExplainer as OceanMIPExplainer
        from ocean.feature import parse_features


        raw_model = model.model
        data_public: PublicData = data  # type: ignore[assignment]

        _, mapper = parse_features(
            data_public.data,
            scale=False,
            drop_na=False,
            drop_constant=False,
        )

        self._ocean_mapper = mapper

        # Precompute binary feature mappings for row transformation
        self._binary_kept_value = {}
        for col in data_public.column_names:
            if data_public.data[col].nunique() == 2:  # noqa: PLR2004
                unique_sorted = sorted(data_public.data[col].unique())
                self._binary_kept_value[col] = unique_sorted[1]

        epsilon = kwargs.get("epsilon", 1.0 / (2.0**16))
        num_epsilon = kwargs.get("num_epsilon", 1.0 / (2.0**6))

        return OceanMIPExplainer(
            ensemble=raw_model,
            mapper=mapper,
            epsilon=epsilon,
            num_epsilon=num_epsilon,
        )

    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        """Reject samples containing non-numeric (unencoded categorical) columns.

        OCEAN operates on numeric data only. Categorical features must be
        one-hot encoded before use.
        """
        frame = sample.to_frame().T if isinstance(sample, pd.Series) else sample
        non_numeric = frame.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            message = (
                f"OCEAN requires all features to be numeric. "
                f"Found non-numeric columns: {non_numeric}. "
                f"Categorical features must be one-hot encoded."
            )
            raise ConfigurationError(
                message=message,
                param="sample",
                hint="One-hot encode categorical features before passing data to OCEAN.",
                source="OCEANClassifierExplainer._validate_sample",
            )

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        target_class = kwargs.get("target_class")
        norm = kwargs.get("norm", 1)
        max_time = kwargs.get("max_time", 60)
        verbose = kwargs.get("verbose", False)
        num_workers = kwargs.get("num_workers")
        random_seed = kwargs.get("random_seed", 42)

        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T

        # Single row
        if sample.shape[0] == 1:
            result = self._explain_single_row(
                row=sample.iloc[0],
                target_class=target_class,
                norm=norm,
                verbose=verbose,
                max_time=max_time,
                num_workers=num_workers,
                random_seed=random_seed,
            )
            if result is None:
                raise NoCounterfactualsFoundError()
            return result

        # Multiple rows
        cf_list: list[Counterfactual] = []
        for _, row in sample.iterrows():
            result = self._explain_single_row(
                row=row,
                target_class=target_class,
                norm=norm,
                verbose=verbose,
                max_time=max_time,
                num_workers=num_workers,
                random_seed=random_seed,
            )
            if result is not None:
                cf_list.append(result)

        if not cf_list:
            raise NoCounterfactualsFoundError()

        return cf_list

    def _explain_single_row(
        self,
        row: pd.Series,
        target_class: int | None,
        norm: int,
        verbose: bool,
        max_time: int,
        num_workers: int | None,
        random_seed: int,
    ) -> Counterfactual | None:
        """Run OCEAN's MIP solver for a single instance.

        Parameters
        ----------
        row : pd.Series
            Single instance to explain.
        target_class : int | None
            Desired counterfactual class. Auto-detected for binary classifiers.
        norm : int
            Distance metric (1 for L1, 2 for L2).
        verbose : bool
            Whether to show Gurobi solver output.
        max_time : int
            Solver time limit in seconds.
        num_workers : int | None
            Number of Gurobi threads.
        random_seed : int
            Gurobi random seed.

        Returns
        -------
        Counterfactual | None
            The counterfactual, or None if no feasible solution was found.
        """
        x_ocean = self._transform_row_to_ocean(row)
        current_prediction = int(self.model.predict(row.to_frame().T).item())
        y = self._determine_target_class(current_prediction) if target_class is None else int(target_class)

        custom_constrs = self._add_custom_constraints(x_ocean)

        try:
            explanation = self.explainer.explain(
                x=x_ocean,
                y=y,
                norm=norm,
                verbose=verbose,
                max_time=max_time,
                num_workers=num_workers,
                random_seed=random_seed,
            )
        finally:
            if custom_constrs:
                self.explainer.remove(custom_constrs)
            self.explainer.cleanup()

        if explanation is None:
            return None

        cf_df = self._transform_ocean_to_celia(explanation)
        # Use the MIP target class as the CF prediction. OCEAN's solver
        # guarantees the CF satisfies the majority-class constraint, but
        # sklearn's predict() may disagree at the exact decision boundary
        # due to tie-breaking rules.
        cf_prediction = y

        return Counterfactual(
            original_instance=row.to_frame().T,
            counterfactual_instance=cf_df,
            original_prediction=current_prediction,
            counterfactual_prediction=cf_prediction,
        )

    def _add_custom_constraints(self, x_ocean: np.ndarray) -> list:
        """Add Gurobi constraints for immutable features and feasible values.

        Parameters
        ----------
        x_ocean : np.ndarray
            The query instance in OCEAN's column space.

        Returns
        -------
        list
            List of Gurobi constraint objects to be removed after solving.
        """
        import gurobipy as gp

        data_public: PublicData = self.data  # type: ignore[assignment]
        custom_constrs: list[gp.Constr] = []

        # Immutable feature constraints
        immutable = data_public.immutable_column_names or []
        for feature_name in immutable:
            idx = self._ocean_mapper.idx.get(feature_name)
            var = self.explainer.vget(idx)
            constr = self.explainer.addConstr(var == x_ocean[idx])
            custom_constrs.append(constr)

        # Feasible value constraints
        feasible = data_public.feasible_values or {}
        for feature_name, bounds in feasible.items():
            if (
                isinstance(bounds, (list, tuple))
                and len(bounds) == 2  # noqa: PLR2004
                and all(isinstance(b, (int, float)) for b in bounds)
            ):
                idx = self._ocean_mapper.idx.get(feature_name)
                var = self.explainer.vget(idx)
                lo, hi = bounds
                custom_constrs.append(self.explainer.addConstr(var >= lo))
                custom_constrs.append(self.explainer.addConstr(var <= hi))

        return custom_constrs

    def _determine_target_class(self, current_prediction: int) -> int:
        """Auto-determine the target class for counterfactual generation.

        For binary classifiers, flips the current prediction. For multi-class
        classifiers, raises an error requiring the user to specify the target.

        Parameters
        ----------
        current_prediction : int
            The model's prediction on the original instance.

        Returns
        -------
        int
            The desired counterfactual class.

        Raises
        ------
        ConfigurationError
            If the classifier has more than 2 classes and no target is specified.
        """
        n_classes = self.explainer.n_classes
        if n_classes == 2:  # noqa: PLR2004
            return 1 - current_prediction

        message = (
            f"OCEAN requires an explicit `target_class` for multi-class classifiers "
            f"(found {n_classes} classes). Pass `target_class=<int>` to "
            f"`generate_counterfactuals()`."
        )
        raise ConfigurationError(
            message=message,
            param="target_class",
            hint="Specify the desired counterfactual class, e.g. `target_class=1`.",
            source="OCEANClassifierExplainer._determine_target_class",
        )

    def _transform_row_to_ocean(self, row: pd.Series) -> np.ndarray:
        """Convert a CELIA row to the 1D array expected by OCEAN's ``explain()``.

        For continuous features (>2 unique training values), values pass through
        unchanged. For binary features (exactly 2 unique training values), values
        are mapped to 0/1 matching ``parse_features``'s ``get_dummies(drop_first=True)``.

        Parameters
        ----------
        row : pd.Series
            Single instance in CELIA's feature space.

        Returns
        -------
        np.ndarray
            1D array of float64 values in OCEAN's column space.
        """
        values: list[float] = []
        for feature_name in self._ocean_mapper:
            feature = self._ocean_mapper[feature_name]
            val = row[feature_name]
            if feature.is_binary and feature_name in self._binary_kept_value:
                values.append(1.0 if val == self._binary_kept_value[feature_name] else 0.0)
            else:
                values.append(float(val))
        return np.array(values, dtype=np.float64)

    def _transform_ocean_to_celia(self, explanation: Any) -> pd.DataFrame:
        """Convert an OCEAN explanation back to a CELIA-compatible DataFrame.

        Parameters
        ----------
        explanation : ocean.mip.Explanation
            The OCEAN explanation containing counterfactual values.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with CELIA's original column names and values.
        """
        data_public: PublicData = self.data  # type: ignore[assignment]
        cf_values: dict[str, Any] = explanation.value

        result: dict[str, Any] = {}
        for feature_name in data_public.column_names:
            feature = self._ocean_mapper[feature_name]
            val = cf_values[feature_name]

            if feature.is_binary and feature_name in self._binary_kept_value:
                unique_sorted = sorted(data_public.data[feature_name].unique())
                result[feature_name] = unique_sorted[1] if np.isclose(float(val), 1.0) else unique_sorted[0]
            else:
                result[feature_name] = float(val)

        return pd.DataFrame([result], columns=data_public.column_names)
