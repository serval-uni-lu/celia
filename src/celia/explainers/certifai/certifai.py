import numpy as np
import pandas as pd
from certifai import CERTIFAI

from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import RegressorExplainer
from celia.model import BaseModel


class CertifaiRegressorExplainer(RegressorExplainer):
    """
        Concrete implementation of RegressorExplainer for CERTIFAI method.
        This class wraps the CERTIFAI method to generate counterfactual explanations
        for regression tasks using public training data. It enforces the use of PublicData
        and validates that the input data meets the expected structure required by the explainer.
        Note: CERTIFAI requires that category data is one-hot encoded.

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
        Additional keyword arguments. For more details, refer to the CERTIFAI documentation.
        https://github.com/alanpar97/CERTIFAI/

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

    explainer : CERTIFAI
        Instance of the CERTIFAI class initialized with training data, model,
        and target variable for regression tasks.
    """

    def __init__(self, model: BaseModel, data: PublicData, *args: object, **kwargs: object) -> None:
        # Assert that data is an instance of PublicData
        if not isinstance(data, PublicData):
            message = "CertifaiRegressorExplainer requires data to be an instance of PublicData."
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
    ) -> CERTIFAI:
        pm = kwargs.pop("Pm", 0.1)
        pc = kwargs.pop("Pc", 0.1)

        exp = CERTIFAI(Pm=pm, Pc=pc, pandas_dataset=data.data)
        exp.set_constraints(fixed=data.immutable_column_names or None)
        return exp

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        target_range: tuple[float, float] | list[float],
        *args: object,
        **kwargs: object,
    ) -> Counterfactual | list[Counterfactual]:
        """
        Generate counterfactuals using CERTIFAI.

        Parameters
        ----------
        sample : pd.DataFrame | pd.Series
            The instance for which counterfactuals are generated.

        target_range : tuple[float, float] | list[float]
            Lower and upper bounds for desired prediction.

        *args, **kwargs :
            Additional CERTIFAI arguments.

        Returns
        -------
        Counterfactual | list[Counterfactual]
            One or multiple counterfactual explanation objects.
        """

        generations = kwargs.get("generations", 3)
        distance = kwargs.get("distance", "L1")
        final_k = kwargs.get("final_k", 1)
        trained_with_columns = kwargs.get("trained_with_columns", True)
        model_type = kwargs.get("model_type", "sklearn")
        select_retain = kwargs.get("select_retain", 1000)
        gen_retain = kwargs.get("gen_retain", 500)
        verbose = kwargs.get("verbose", False)

        target_lower = np.atleast_1d(target_range[0])
        target_upper = np.atleast_1d(target_range[1])

        if len(sample) > 1:
            target_lower = np.full(shape=(len(sample),), fill_value=target_lower.item())
            target_upper = np.full(shape=(len(sample),), fill_value=target_upper.item())

        self.explainer.fit(
            self.model,
            x=sample,
            generations=generations,
            distance=distance,
            final_k=final_k,
            classification=False,
            trained_with_columns=trained_with_columns,
            target_name=self.data.target_name,
            target_lower=target_lower,
            target_upper=target_upper,
            model_type=model_type,
            select_retain=select_retain,
            gen_retain=gen_retain,
            verbose=verbose,
        )

        if not self.explainer.results:
            message = "No counterfactuals generated. Check the input parameters and data."
            raise NoCounterfactualsFoundError(message)

        results = self.explainer.results
        columns = list(self.data.column_names) + [self.data.target_name]

        counterfactuals: list[Counterfactual] = []

        for res in results:
            original_instance = res[0]
            counterfactual_array = res[1]

            counterfactual_df = pd.DataFrame(counterfactual_array, columns=columns)
            cf = Counterfactual(
                original_instance=original_instance,
                counterfactual_instance=counterfactual_df,
            )
            counterfactuals.append(cf)

        return counterfactuals[0] if len(counterfactuals) == 1 else counterfactuals

    def _validate_sample(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> None:
        """CERTIFAI specific sample validation. Checks if categories are one-hot encoded."""
        non_numeric_columns = [
            col
            for col, dtype in sample.dtypes.items()
            if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype))
        ]
        if non_numeric_columns:
            message = "CERTIFAI requires that categorical features are one-hot encoded."
            raise ConfigurationError(
                message=message,
                param="sample",
                hint="Please ensure all categorical features are one-hot encoded before passing the sample.",
                config={"sample_dtypes": sample.dtypes.astype(str).to_dict(),
                        "non_numeric_columns": non_numeric_columns},
            )
