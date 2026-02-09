import inspect
from typing import Any, Callable, cast

import dice_ml
import pandas as pd
from dice_ml.diverse_counterfactuals import (
    CounterfactualExamples as dice_CounterfactualExamples,
)

from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import RegressorExplainer
from celia.model import BaseModel


class DiceRegressorExplainer(RegressorExplainer):
    """
    Concrete implementation of RegressorExplainer for Diverse Counterfactuals (DiCE).
    This class wraps the DiCE method to generate counterfactual explanations
    for regression tasks using public training data. It enforces the use of PublicData
    and validates that the input data meets the expected structure required by the explainer.

    Parameters
    ----------
    model : BaseModel
        The predictive regression model to be explained. Must implement the BaseModel interface
        with a `predict` method.

    data : PublicData
        The public dataset object, containing the training data, target labels, and metadata
        such as feature types and feasible values.

    *args : Any
        Additional positional arguments passed to the parent RegressorExplainer class.

    **kwargs : Any
        Additional keyword arguments.

    Raises
    ------
    ValueError
        If the provided data is not an instance of PublicData.

    Attributes
    ----------
    model : BaseModel
        The regression model to be explained.

    data : PublicData
        The dataset used to generate counterfactual explanations.

    explainer : dice_ml.Dice
        Instance of the NearestNeighborCE class initialized with training data, model,
        and target variable for regression tasks.
    """

    def __init__(self, model: BaseModel, data: PublicData, *args: object, **kwargs: object) -> None:
        if not isinstance(data, PublicData):
            message = "DiceRegressorExplainer requires data to be an instance of PublicData."
            raise ConfigurationError(
                message=message,
                param="data",
                hint="Please provide a PublicData object with appropriate metadata.",
                config={"data_type": type(data).__name__},
            )
        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(
        self,
        model: BaseModel,
        data: PublicData,
        *args: object,
        **kwargs: object,
    ) -> Any:
        # DiCE requires that data.data includes the target variable as well.
        data_with_targets = data.data.copy()
        data_with_targets[data.target_name] = data.targets

        method = kwargs.pop("method", "random")

        # Filter kwargs for each component
        data_kwargs = self._filter_kwargs(dice_ml.Data.__init__, kwargs)
        model_kwargs = self._filter_kwargs(dice_ml.Model.__init__, kwargs)
        dice_kwargs = kwargs

        d = dice_ml.Data(
            dataframe=data_with_targets,
            continuous_features=data.continuous_column_names or None,
            permitted_range=self.data.feasible_values if self.data.feasible_values is not None else {},
            outcome_name=data.target_name,
            **data_kwargs,
        )

        m = dice_ml.Model(
            model=model,
            backend="sklearn",
            model_type="regressor",
            **model_kwargs,
        )

        return dice_ml.Dice(d, m, method=method, **dice_kwargs)

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        target_range: tuple[float, float] | list[float],
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual]:
        """
        Generate counterfactuals using DiCE.

        Parameters
        ----------
        sample : pd.DataFrame | pd.Series
            Input instance(s) to explain.
        target_range : tuple[float, float] | list[float]
            Desired prediction interval for regression CFs.

        Returns
        -------
        list[Counterfactual]
        """

        features_to_vary = (
            [col for col in self.data.column_names if col not in self.data.immutable_column_names]
            if self.data.immutable_column_names
            else "all"
        )

        total_cfs = kwargs.pop("total_CFs", 1)

        results = self.explainer.generate_counterfactuals(
            query_instances=sample,
            desired_range=target_range,
            total_CFs=total_cfs,
            features_to_vary=features_to_vary,
            permitted_range=self.data.feasible_values or None,
            *args,
            **kwargs,
        )

        if not results.cf_examples_list:
            message = "No counterfactuals generated. Check the input parameters and data."
            raise NoCounterfactualsFoundError(message)

        counterfactual_list: list[Counterfactual] = []

        for cf_group in results.cf_examples_list:
            # DiCE modifies predictions; compute true predictions
            original_pred, cf_preds = self._get_true_predictions(cf_group)

            cf_group.test_instance_df[self.data.target_name] = original_pred
            cf_group.final_cfs_df[self.data.target_name] = cf_preds

            ce = Counterfactual(
                original_instance=cf_group.test_instance_df,
                counterfactual_instance=cf_group.final_cfs_df,
            )
            counterfactual_list.append(ce)

        return counterfactual_list

    def _get_true_predictions(
        self,
        counterfactuals: dice_CounterfactualExamples,
    ) -> tuple[float, list[float]]:
        """Get the true model predictions for the original instance and counterfactuals."""

        original_instance = cast(pd.DataFrame, counterfactuals.test_instance_df).drop(
            columns=[self.data.target_name],
            errors="ignore",
        )
        counterfactual_instances = cast(pd.DataFrame, counterfactuals.final_cfs_df).drop(
            columns=[self.data.target_name],
            errors="ignore",
        )

        original_pred = self.model.predict(original_instance)
        cf_preds: list[float] = self.model.predict(counterfactual_instances)

        return float(original_pred[0]), cf_preds

    @staticmethod
    def _filter_kwargs(constructor: Callable[..., Any], all_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Filters a dictionary of kwargs to include only those
        accepted by the given constructor function.

        Parameters
        ----------
        constructor : Callable
            The class or function whose signature will be inspected.

        all_kwargs : dict
            A dictionary of keyword arguments to filter.

        Returns
        -------
        dict
            A dictionary containing only the arguments valid for the constructor.
        """
        sig = inspect.signature(constructor)
        valid_params = set(sig.parameters)
        return {key: value for key, value in all_kwargs.items() if key in valid_params}
