from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from celia.explainers import BaseExplainer, RegressorExplainer
from celia.data import PublicData
from certifai import CERTIFAI

from celia.model import BaseModel


class CertifaiRegressorExplainer(RegressorExplainer):
    """
        Concrete implementation of RegressorExplainer for CERTIFAI method.
        This class wraps the CERTIFAI method to generate counterfactual explanations
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
    def __init__(self, model: BaseModel, data: PublicData, *args, **kwargs):
        # Assert that data is an instance of PublicData
        if not isinstance(data, PublicData):
            raise ValueError("data must be an instance of PublicData")
        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model:BaseModel, data:PublicData, *args, **kwargs):
        Pm = kwargs.pop("Pm", 0.1)
        Pc = kwargs.pop("Pc", 0.1)
        exp = CERTIFAI(Pm=Pm, Pc=Pc, pandas_dataset=data.data)
        exp.set_constraints(fixed=data.immutable or None)
        return exp

    def _generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series],
                                  target_range: Union[List[float], Tuple[float, float]],
                                  *args, **kwargs) -> pd.DataFrame:

        # Check if user provided any additional parameters
        generations = kwargs.get('generations', 3)
        distance = kwargs.get('distance', 'L1')
        final_k = kwargs.get('final_k', 1)
        trained_with_columns = kwargs.get('trained_with_columns', True)
        model_type = kwargs.get('model_type', 'sklearn')
        select_retain = kwargs.get('select_retain', 1000)
        gen_retain = kwargs.get('gen_retain', 500)
        verbose = kwargs.get('verbose', False)

        self.explainer.fit(
            self.model,
            x=sample,
            generations=generations,
            distance=distance,
            final_k=final_k,
            classification=False,
            trained_with_columns=trained_with_columns,
            target_lower=np.atleast_1d(target_range[0]),
            target_upper=np.atleast_1d(target_range[1]),
            model_type=model_type,
            select_retain=select_retain,
            gen_retain=gen_retain,
            verbose=verbose
        )

        counterfactuals_list = self.explainer.results[0][1]
        columns = list(self.data.columns) + [self.data.target_name]
        counterfactuals_df = pd.DataFrame(counterfactuals_list, columns=columns)
        return counterfactuals_df

