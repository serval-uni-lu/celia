from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from celia._utils.dependecies import requires_torch_class


class LinearBlock(nn.Module):
    """A standard building block for the MLPs."""

    def __init__(self, input_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MultilayerPerception(nn.Module):
    """Stacks multiple LinearBlocks based on a list of dimensions."""

    def __init__(self, dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(1, len(dims)):
            layers.append(LinearBlock(dims[i - 1], dims[i], dropout=dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class CategoricalNormalizer:
    """
    Post-processing step to enforce valid Label/CatBoost encoded values.
    Uses the Straight-Through Estimator (STE) to snap continuous outputs
    to the nearest valid category while preserving gradient flow.
    """

    def __init__(self, valid_values_dict: dict[int, Tensor]):
        """
        @param valid_values_dict: A dictionary where the key is the column index
                                  and the value is a 1D tensor of valid encoded numbers.
        """
        self.valid_values_dict = valid_values_dict

    def normalize(self, x: Tensor, hard: bool = False) -> Tensor:
        if not self.valid_values_dict:
            return x

        x_out = x.clone()

        for col_idx, valid_values in self.valid_values_dict.items():
            # Move valid_values to the same device as x
            valid_values = valid_values.to(x.device)

            # Extract the continuous predictions for this column: shape (batch_size, 1)
            col_preds = x[:, col_idx].unsqueeze(1)

            if hard:
                # 1. Calculate the distance between predictions and all valid values
                # distances shape: (batch_size, num_valid_values)
                distances = torch.abs(col_preds - valid_values.unsqueeze(0))

                # 2. Find the index of the closest valid value
                nearest_idx = torch.argmin(distances, dim=1)

                # 3. Get the actual nearest valid values
                nearest_values = valid_values[nearest_idx]

                # 4. The Straight-Through Estimator (STE) Trick!
                # We want to output 'nearest_values', but PyTorch can't backpropagate through argmin.
                # So we detach the hard values, subtract the detached soft values, and add the soft values back.
                # Forward pass result: nearest_values. Backward pass result: gradients of col_preds.
                snapped_values = nearest_values.detach() - x[:, col_idx].detach() + x[:, col_idx]

                x_out[:, col_idx] = snapped_values
            else:
                # During the soft training phase, we just let the network output continuous numbers
                # so the Proximity Loss (MSE) can pull it smoothly toward the original instance.
                pass

        return x_out


@requires_torch_class
class CounterNet(pl.LightningModule):
    """
    The complete CounterNet model and training loop.
    Includes the alternating dual-optimizer logic.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        valid_values_dict: dict,
        bb_setting: bool = False,
    ):
        super().__init__()
        # self.save_hyperparameters(config)
        self.bb_setting = bb_setting

        # Required when using multiple optimizers (GANs, CounterNet, etc.)
        self.automatic_optimization = False

        # Hyperparameters
        self.lr = config["lr"]
        self.lambda_1 = config["lambda_1"]  # Weight for Predictor Loss
        self.lambda_2 = config["lambda_2"]  # Weight for Cost/Proximity Loss
        self.lambda_3 = config["lambda_3"]  # Weight for Validity Loss

        enc_dims = config["enc_dims"]
        dec_dims = config["dec_dims"]
        exp_dims = config["exp_dims"]

        # 1. Encoder
        self.encoder_model = MultilayerPerception(enc_dims)

        # 2. Predictor
        self.predictor = MultilayerPerception(dec_dims)
        self.pred_linear = nn.Linear(dec_dims[-1], 1)

        # 3. Explainer (Generator)
        # The explainer concatenates the encoder output AND the predictor's hidden state
        # exp_input_dim = exp_dims[0] + dec_dims[-1]
        # adjusted_exp_dims = [exp_input_dim] + exp_dims[1:]
        exp_dims = list(exp_dims)
        exp_dims[0] = exp_dims[0] + dec_dims[-1]

        self.explainer = nn.Sequential(
            MultilayerPerception(exp_dims),
            nn.Linear(exp_dims[-1], enc_dims[0]),  # Output must match original input size
        )

        # Initialize the normalizer
        self.normalizer = CategoricalNormalizer(valid_values_dict)

    def forward(self, x: Tensor, hard: bool = False) -> Tuple[Tensor, Tensor]:
        encoded_x = self.encoder_model(x)
        pred_hidden = self.predictor(encoded_x)
        y_logits = self.pred_linear(pred_hidden)
        y_hat = torch.sigmoid(y_logits)

        concat_features = torch.cat((encoded_x, pred_hidden), dim=-1)
        c = self.explainer(concat_features)

        # Apply the normalizer to the generated counterfactual
        c_normalized = self.normalizer.normalize(c, hard=hard)

        return torch.squeeze(y_hat, dim=-1), c_normalized

    # Add this inside the CounterNet class
    def predict_only(self, x: Tensor) -> Tensor:
        """Bypass the explainer to save compute."""
        encoded_x = self.encoder_model(x)
        pred_hidden = self.predictor(encoded_x)
        y_logits = self.pred_linear(pred_hidden)
        return torch.squeeze(torch.sigmoid(y_logits), dim=-1)

    def _loss_functions(self, x: Tensor, c: Tensor, y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the three distinct losses."""
        # 1. Prediction Loss (Knowledge Distillation)
        # y is now the soft probability from the black box.
        # MSE is often better for distilling soft probabilities than BCE.
        l_1 = (
            nn.functional.mse_loss(y_hat, y.float())
            if self.bb_setting
            else nn.functional.binary_cross_entropy(y_hat, y.float())
        )

        # 2. Proximity/Cost Loss (Mean Squared Error between original and CF)
        l_2 = nn.functional.mse_loss(x, c)

        # 3. Validity Loss
        # What does the model predict for the generated counterfactual?
        c_y_hat = self.predict_only(c)

        # The target for validity is the exact opposite of what the model originally predicted
        y_prime = 1.0 - torch.round(y_hat.detach())
        l_3 = nn.functional.binary_cross_entropy(c_y_hat, y_prime)
        # l_3 = nn.functional.mse_loss(c_y_hat, y_prime)

        return l_1, l_2, l_3

    def configure_optimizers(self) -> list[torch.optim]:
        """Two distinct optimizers for alternating training steps."""
        # 1. Gather all predictor-related parameters
        raw_pred_params = (
            list(self.encoder_model.parameters())
            + list(self.predictor.parameters())
            + list(self.pred_linear.parameters())
        )

        # Filter and assign to Optimizer 1
        # pred_params = [p for p in self.parameters() if p.requires_grad]
        opt_1 = torch.optim.Adam(raw_pred_params, lr=self.lr)

        # 2. Gather explainer parameters, filter, and assign to Optimizer 2
        exp_params = [p for p in self.explainer.parameters() if p.requires_grad]
        opt_2 = torch.optim.Adam(exp_params, lr=self.lr)

        return [opt_1, opt_2]

    def training_step(self, batch: torch.utils.data.DataLoader, batch_idx: int) -> None:
        """Modern manual optimization step."""
        # Retrieve the two optimizers
        opt_predictor, opt_explainer = self.optimizers()

        x, y = batch

        # ====================================
        # Phase 1: Train the Predictor
        # ====================================
        y_hat, c = self(x)
        l_1, l_2, l_3 = self._loss_functions(x, c, y, y_hat)

        p_loss = self.lambda_1 * l_1

        opt_predictor.zero_grad()
        # retain_graph=True allows us to reuse the graph for the Explainer phase
        self.manual_backward(p_loss)
        opt_predictor.step()

        # ====================================
        # Phase 2: Train the Explainer
        # ====================================
        y_hat, c = self(x, hard=True)
        _, l_2_new, l_3_new = self._loss_functions(x, c, y, y_hat)
        e_loss = (self.lambda_2 * l_2_new) + (self.lambda_3 * l_3_new)

        opt_explainer.zero_grad()
        self.manual_backward(e_loss)
        opt_explainer.step()

        # Optional: Log the losses so you can track them
        self.log("train_p_loss", p_loss, prog_bar=True)
        self.log("train_e_loss", e_loss, prog_bar=True)

    def validation_step(self, batch: torch.utils.data.DataLoader, batch_idx: int) -> None:
        x, y = batch

        # 1. Forward pass
        y_hat, c = self(x, hard=True)

        # 2. Calculate the individual losses
        l_1, l_2, l_3 = self._loss_functions(x, c, y, y_hat)

        # 3. Combine them into a single validation loss metric
        val_loss = (self.lambda_1 * l_1) + (self.lambda_2 * l_2) + (self.lambda_3 * l_3)

        # 4. Calculate some human-readable metrics (Accuracy)
        # Predictor Accuracy: Did the model guess the original class correctly?
        pred_acc = (torch.round(y_hat) == y).float().mean()

        # Validity Accuracy: Did the generated counterfactual successfully flip the prediction?
        c_y_hat = self.predict_only(c)
        # The target for the CF is the flipped version of the ORIGINAL prediction
        y_prime = 1.0 - torch.round(y_hat)
        validity_acc = (torch.round(c_y_hat) == y_prime).float().mean()

        # 5. Log everything
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_pred_acc", pred_acc, prog_bar=True)
        self.log("val_validity_acc", validity_acc, prog_bar=True)
