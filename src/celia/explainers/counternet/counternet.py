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
#from celia.explainers.counternet._module import CounterNet 
from celia.model import BaseModel, TorchModel  # Assuming PytorchModel exists in Celia



class CounterNetClassifierExplainer(ClassifierExplainer):
    """CounterNet: End-to-End Training of Prediction Aware Counterfactual Explanations.

    This explainer natively supports PyTorch models. It uses the provided 
    `PytorchModel` to label the training data, and then jointly trains the 
    CounterNet predictor and counterfactual generator using PyTorch Lightning.

    Parameters
    ----------
    model : PytorchModel
        A CELIA ``PytorchModel`` wrapping a fitted PyTorch classifier.
    data : PublicData
        Training data with metadata (feature names, types, constraints).
    epochs : int
        Number of epochs to train the CounterNet model (default: 50).
    batch_size : int
        Batch size for training CounterNet (default: 64).
    """

    def __init__(
        self, 
        model: BaseModel, 
        data: BaseData, 
        epochs: int = 50,
        batch_size: int = 64,
        *args: object, 
        **kwargs: object
    ) -> None:
        
        self.epochs = epochs
        self.batch_size = batch_size

        # 1. Validate model type strictly for PyTorch
        if not isinstance(model, TorchModel):
            message = "CounterNetExplainer requires a PytorchModel."
            raise ConfigurationError(
                message=message,
                param="model",
                config={"expected": "PytorchModel", "received": type(model).__name__},
                hint="Wrap your PyTorch classifier with `PytorchModel(model=...)`.",
                source="CounterNetExplainer.__init__",
            )

        # 2. Validate data type
        if not isinstance(data, PublicData):
            message = "CounterNetExplainer requires data to be an instance of PublicData."
            raise ConfigurationError(
                message=message,
                param="data",
                config={"expected": "PublicData", "received": type(data).__name__},
                source="CounterNetExplainer.__init__",
            )

        # 3. Trigger _create_explainer via super()
        super().__init__(model, data, *args, **kwargs)

    def _create_explainer(self, model: TorchModel, data: PublicData, *args: object, **kwargs: object) -> Any:

        X_df = data.data.copy()
        y_df = data.targets.copy()

        # Split into train test
        X_train, X_val, y_train, y_val = train_test_split(
            X_df, y_df, test_size=0.2, random_state=42, stratify=y_df
        )

        X_train_tensor = torch.FloatTensor(X_train.to_numpy())
        y_train_tensor = torch.FloatTensor(y_train.to_numpy())

        X_val_tensor = torch.FloatTensor(X_val.to_numpy())
        y_val_tensor = torch.FloatTensor(y_val.to_numpy())

        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        val_ds = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        input_dim = X_train_tensor.shape[1]
        
        # Define the exact dimensions for the three blocks
        # (You can expose these as init parameters in CounterNetExplainer if you want)
        enc_dims = [input_dim, 50, 10]
        dec_dims = [10, 10]
        exp_dims = [10, 50] # Note: the first dimension will be adjusted internally

        config_dict = {
            "enc_dims": enc_dims,
            "dec_dims": dec_dims,
            "exp_dims": exp_dims,
            "lr": 1e-3,
            "lambda_1": 1.0,
            "lambda_2": .001,
            "lambda_3": .2,
        }

        counternet_model = CounterNet(
            config=config_dict
        ) 

        trainer = pl.Trainer(max_epochs=self.epochs, logger=False, enable_checkpointing=False)
        trainer.fit(counternet_model, train_dataloaders=train_loader)

        counternet_model.eval()
        return counternet_model

    def _generate_counterfactuals(
        self,
        sample: pd.DataFrame | pd.Series,
        *args: object,
        **kwargs: object,
    ) -> list[Counterfactual] | Counterfactual:
        """Generate counterfactual explanations."""
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
        """Run the CounterNet forward pass for one instance."""
        x_tensor = torch.FloatTensor(row.to_numpy()).unsqueeze(0)
        
        with torch.no_grad():
            try:
                # CounterNet typically returns (predictions, counterfactuals)
                _, cf_tensor = self.explainer(x_tensor)
            except Exception:
                return None

        # CounterNet might return a tensor attached to a compute graph or GPU,
        # so we ensure it is moved to CPU and detached before converting to NumPy.
        cf_numpy = cf_tensor.detach().cpu().numpy()
        cf_df = pd.DataFrame(cf_numpy, columns=row.index)

        return Counterfactual(
            original_instance=row.to_frame().T,
            counterfactuals=cf_df,
        )

    def _validate_sample(self, sample: pd.DataFrame | pd.Series, *args: object, **kwargs: object) -> None:
        """Ensure the input is numeric and ready for tensor conversion."""
        if not np.issubdtype(sample.to_numpy().dtype, np.number):
            message = "CounterNetExplainer requires strictly numerical inputs."
            raise ConfigurationError(
                message=message,
                source="CounterNetExplainer._validate_sample"
            )
