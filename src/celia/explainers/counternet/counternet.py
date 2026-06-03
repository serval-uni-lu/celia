from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.data._base import BaseData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import ClassifierExplainer
from celia.explainers.counternet._module import CounterNet
from celia.model import BaseModel  # Assuming PytorchModel exists in Celia


class CounterNetClassifierExplainer(ClassifierExplainer):
    """CounterNet: End-to-End Training of Prediction Aware Counterfactual Explanations."""

    def __init__(
        self,
        model: PublicData,
        data: BaseData,
        epochs: int = 100,
        batch_size: int = 64,
        bb_setting: bool = True,
        *args: object,
        **kwargs: object,
    ) -> None:

        self.epochs = epochs
        self.batch_size = batch_size
        self.bb_setting = bb_setting

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

        if not isinstance(data, PublicData):
            raise ConfigurationError(
                message="CounterNetExplainer requires data to be an instance of PublicData.",
                param="data",
                config={"expected": "PublicData", "received": type(data).__name__},
                source="CounterNetExplainer.__init__",
            )

        super().__init__(model, data, *args, **kwargs)

    def _get_default_dims(self, input_dim: int) -> tuple[list[int], list[int], list[int]]:
        """
        Generates sensible architecture dimensions dynamically based on input size.
        """
        # 1. Encoder: Project upwards or maintain capacity
        # For small datasets, 32 or 64 is a safe latent bottleneck.
        latent_dim = max(32, int(2 ** np.ceil(np.log2(input_dim * 2))))
        enc_dims = [input_dim, latent_dim, latent_dim // 2]

        # 2. Predictor: Processes the latent representation
        # Must start with enc_dims[-1]
        dec_dims = [enc_dims[-1], max(16, enc_dims[-1])]

        # 3. Explainer: Needs capacity to generate modifications
        # Must start with dec_dims[-1] (it gets concat'd with enc_dims[-1] inside the module)
        exp_dims = [dec_dims[-1], latent_dim]

        return enc_dims, dec_dims, exp_dims

    def _create_explainer(self, model: BaseModel, data: PublicData, *args: object, **kwargs: object) -> Any:
        x_df = data.data.copy()

        if self.bb_setting:
            bb_predictions = model.predict(x_df.to_numpy())

            # Assuming binary classification, grab the probability of the positive class (class 1)
            if bb_predictions.ndim > 1 and bb_predictions.shape[1] == 2:
                y_df = pd.Series(bb_predictions[:, 1])
            else:
                y_df = pd.Series(bb_predictions)
        else:
            y_df = data.targets.copy()

        x_train, x_val, y_train, y_val = train_test_split(x_df, y_df, test_size=0.2, random_state=42, stratify=y_df)

        x_train_tensor = torch.FloatTensor(x_train.to_numpy())
        y_train_tensor = torch.FloatTensor(y_train.to_numpy())

        x_val_tensor = torch.FloatTensor(x_val.to_numpy())
        y_val_tensor = torch.FloatTensor(y_val.to_numpy())

        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        val_ds = TensorDataset(x_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        valid_values_dict = self._get_categorical_metadata(data)

        input_dim = x_train_tensor.shape[1]

        enc_dims = kwargs.get("enc_dims")
        dec_dims = kwargs.get("dec_dims")
        exp_dims = kwargs.get("exp_dims")

        if not all([enc_dims, dec_dims, exp_dims]):
            extracted = self._get_default_dims(input_dim)
            enc_dims = enc_dims or extracted[0]
            dec_dims = dec_dims or extracted[1]
            exp_dims = exp_dims or extracted[2]

        config_dict = {
            "enc_dims": enc_dims,
            "dec_dims": dec_dims,
            "exp_dims": exp_dims,
            "lr": kwargs.get("lr", 0.01),
            "lambda_1": kwargs.get("lambda_1", 1.0),
            "lambda_2": kwargs.get("lambda_2", 0.01),
            "lambda_3": kwargs.get("lambda_3", 1.0),
        }

        counternet_model = CounterNet(
            config=config_dict,
            valid_values_dict=valid_values_dict,
            bb_setting=self.bb_setting,
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
            # 1. Forward pass for the original data point
            y_hat, cf_tensor = self.explainer(x_tensor, hard=True)
            original_pred = torch.round(y_hat).int().cpu().numpy().item()

            # 2. OPTIMIZED: Bypass the explainer sub-network for the counterfactual prediction
            cf_y_hat = self.explainer.predict_only(cf_tensor)
            cf_preds = torch.round(cf_y_hat).int().cpu().numpy().item()

        cf_numpy = cf_tensor.squeeze(0).cpu().numpy()
        cf_df = pd.DataFrame([cf_numpy], columns=row.index)

        return Counterfactual(
            original_instance=row.to_frame().T,
            counterfactual_instance=cf_df,
            original_prediction=original_pred,
            counterfactual_prediction=cf_preds,
        )

    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        if not np.issubdtype(sample.to_numpy().dtype, np.number):
            raise ConfigurationError(
                message="CounterNetExplainer requires strictly numerical inputs.",
                source="CounterNetExplainer._validate_sample",
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
                msg = f"Categorical column '{col_name}' not found in X_train."
                raise ValueError(msg)

        return valid_values_dict
