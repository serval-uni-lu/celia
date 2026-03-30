from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.data._base import BaseData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.explainers.counternet._module import CounterNet 
from celia.model import BaseModel, TorchModel  # Assuming PytorchModel exists in Celia



class CounterNetClassifierExplainer(ClassifierExplainer):
    """CounterNet: End-to-End Training of Prediction Aware Counterfactual Explanations."""

    def __init__(
        self, 
        model: BaseModel, 
        data: BaseData, 
        epochs: int = 100,
        batch_size: int = 64,
        *args: object, 
        **kwargs: object
    ) -> None:
        
        self.epochs = epochs
        self.batch_size = batch_size

        if not isinstance(model, TorchModel):
            raise ConfigurationError(
                message="CounterNetExplainer requires a PytorchModel.",
                param="model",
                config={"expected": "TorchModel", "received": type(model).__name__},
                source="CounterNetExplainer.__init__",
            )

        if not isinstance(data, PublicData):
            raise ConfigurationError(
                message="CounterNetExplainer requires data to be an instance of PublicData.",
                param="data",
                config={"expected": "PublicData", "received": type(data).__name__},
                source="CounterNetExplainer.__init__",
            )

        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model: TorchModel, data: PublicData, *args: object, **kwargs: object) -> Any:
        X_df = data.data.copy()
        y_df = data.targets.copy()

        X_train, X_val, y_train, y_val = train_test_split(
            X_df, y_df, test_size=0.2, random_state=42, stratify=y_df
        )

        X_train_tensor = torch.FloatTensor(X_train.to_numpy())
        y_train_tensor = torch.FloatTensor(y_train.to_numpy())

        X_val_tensor = torch.FloatTensor(X_val.to_numpy())
        y_val_tensor = torch.FloatTensor(y_val.to_numpy())

        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        val_ds = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        valid_values_dict = self._get_categorical_metadata(data)
        
        input_dim = X_train_tensor.shape[1]
        
        enc_dims = [input_dim, 50, 10]
        dec_dims = [10, 10]
        exp_dims = [10, 50] 

        config_dict = {
            "enc_dims": enc_dims,
            "dec_dims": dec_dims,
            "exp_dims": exp_dims,
            "lr": 0.01,
            "lambda_1": 1.0,
            "lambda_2": 0.01,
            "lambda_3": 1,
        }

        counternet_model = CounterNet(
            config=config_dict,
            valid_values_dict=valid_values_dict,
        ) 

        trainer = pl.Trainer(max_epochs=self.epochs, logger=False, enable_checkpointing=False)
        trainer.fit(counternet_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        counternet_model.eval()
        return counternet_model

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T

        cf_list: list[Counterfactual] = []
        for _, row in sample.iterrows():
            result = self._explain_single(row)
            if result is not None:
                cf_list.append(result)

        if not cf_list:
            raise NoCounterfactualsFoundError()

        if sample.shape[0] == 1:
            return cf_list[0]

        return cf_list

    def _explain_single(self, row: pd.Series) -> Counterfactual | None:
        x_tensor = torch.FloatTensor(row.to_numpy()).unsqueeze(0)
        
        with torch.no_grad():
            try:
                # 1. Forward pass for the original data point
                y_hat, cf_tensor = self.explainer(x_tensor, hard=True)
                original_pred = torch.round(y_hat).int().cpu().numpy().item()

                # 2. Forward pass for the generated counterfactual
                cf_y_hat, _ = self.explainer(cf_tensor)
                cf_preds = torch.round(cf_y_hat).int().cpu().numpy().item()

            except Exception as e:
                print(f"CounterNet generation failed: {e}")
                return None

        cf_numpy = cf_tensor.squeeze(0).cpu().numpy()
        cf_df = pd.DataFrame([cf_numpy], columns=row.index)

        return Counterfactual(
            original_instance=row.to_frame().T,
            counterfactual_instance=cf_df,
            original_prediction=original_pred,
            counterfactual_prediction=cf_preds
        )

    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        if not np.issubdtype(sample.to_numpy().dtype, np.number):
            raise ConfigurationError(
                message="CounterNetExplainer requires strictly numerical inputs.",
                source="CounterNetExplainer._validate_sample"
            )
    
    def _get_categorical_metadata(self, data: PublicData) -> dict[int, torch.Tensor]:
        all_cols = list(data.data.columns)
        cat_cols = data.categorical_column_names
        
        valid_values_dict = {}
        
        for col_name in cat_cols:
            if col_name in all_cols:
                col_idx = all_cols.index(col_name)
                unique_values = data.data[col_name].unique()
                valid_values_dict[col_idx] = torch.FloatTensor(unique_values)
            else:
                raise ValueError(f"Categorical column '{col_name}' not found in X_train.")
                
        return valid_values_dict

