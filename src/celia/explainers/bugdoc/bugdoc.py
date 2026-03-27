import logging
from typing import Union, List, Tuple
from bugdoc.algos.stacked_shortcut_standalone import StackedShortcutStandalone as StackedShortcut
from bugdoc.algos.debugging_decision_trees import DebuggingDecisionTrees
import bugdoc.utils.tree as _tree
from bugdoc.utils.quine_mccluskey import prune_tree

import numpy as np
import pandas as pd
from celia.counterfactuals import Counterfactual
from celia.explainers import RegressorExplainer
from celia.data import PublicData

from celia.model import BaseModel



class BugDocRegressorExplainer(RegressorExplainer):
    """
        Concrete implementation of RegressorExplainer for BugDoc method.
        This class wraps the BugDoc method to generate counterfactual explanations
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
    """
    def __init__(self, model: BaseModel, data: PublicData, *args, **kwargs):
        # Assert that data is an instance of PublicData
        if not isinstance(data, PublicData):
            raise ValueError("data must be an instance of PublicData")
        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model:BaseModel, data:PublicData, *args, **kwargs):
        self.budget = kwargs.pop("budget", 100)
        return self


    def _generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series],
                                  target_range: Union[List[float], Tuple[float, float]],
                                  *args, **kwargs) -> pd.DataFrame:
        
         # Check if user provided any additional parameters
        separator = kwargs.get('separator', "#")
        origin = kwargs.get('origin', "debug")
        
        

        sample_dict = {col: self.data.data[col].unique().tolist() if isinstance(self.data.data[col], (pd.Series, np.ndarray)) else [self.data.data[col]] for col in list(self.data.column_names)}

        def model_predict(input_dict):
            input_df = pd.DataFrame([input_dict])
            return  (self.model.predict(input_df)[0] <  target_range[0]) or (target_range[1] < self.model.predict(input_df)[0])

        
        counterfactuals = []
        for i,row in sample.iterrows():
            row_dict = row.to_dict()
            row_dict[self.data.target_name] = self.model.predict(row.to_frame().T)[0]
            original_instance=pd.DataFrame([row_dict])
            input_dict = {col: [row[col]] if col in self.data.immutable_column_names else sample_dict[col] for col in list(self.data.column_names)}
            autodebug = StackedShortcut(max_iter=self.budget,
                                        function=model_predict,
                                        separator=separator,
                                        origin=origin)
            
            results = autodebug.run('entry_point', input_dict)
            if len(results) == 0:
                autodebug = DebuggingDecisionTrees(max_iter=self.budget,
                                            function=model_predict,
                                            separator=separator,
                                            origin=origin)
                _, t, _ = autodebug.run('entry_point', input_dict)
                if _tree.get_depth(t) > 0:
                    keys = list(input_dict.keys())
                    results = prune_tree(t, keys)
            
            for res in results:
                for clause in res:
                    if " == " in clause:
                        feature, value = clause.split(" == ")
                        row_dict[feature] = [v for v in input_dict[feature] if v == value] [0]
                    elif " != " in clause:
                        feature, value = clause.split(" != ")
                        row_dict[feature] = [v for v in input_dict[feature] if v != value] [0]
                    elif " < " in clause:
                        feature, value = clause.split(" < ")
                        row_dict[feature] = [v for v in input_dict[feature] if v < float(value)] [0]
                    elif " >= " in clause:
                        feature, value = clause.split(" >= ")
                        row_dict[feature] = [v for v in input_dict[feature] if v >= float(value)] [0]
   
                row_dict[self.data.target_name] = self.model.predict(pd.DataFrame([row_dict])[list(self.data.column_names)])[0]
                counterfactual_df = pd.DataFrame([row_dict])
                
                counterfactual = Counterfactual(original_instance=original_instance.drop(columns=[self.data.target_name]),
                                                counterfactual_instance=counterfactual_df.drop(columns=[self.data.target_name]),
                                                original_prediction=original_instance[self.data.target_name].iloc[0],
                                                counterfactual_prediction=counterfactual_df[self.data.target_name].iloc[0])
                counterfactuals.append(counterfactual)

        return counterfactuals[0] if len(counterfactuals) == 1 else counterfactuals

