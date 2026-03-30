import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any

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

    def forward(self, x):
        return self.block(x)

class MultilayerPerception(nn.Module):
    """Stacks multiple LinearBlocks based on a list of dimensions."""
    def __init__(self, dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(1, len(dims)):
            layers.append(LinearBlock(dims[i-1], dims[i], dropout=dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

@requires_torch_class
class CounterNet(pl.LightningModule):
    """
    The complete CounterNet model and training loop.
    Includes the alternating dual-optimizer logic.
    """
    def __init__(
        self, 
        config: Dict[str, Any]
    ):
        super().__init__()
        #self.save_hyperparameters(config)

        # Required when using multiple optimizers (GANs, CounterNet, etc.)
        self.automatic_optimization = False
        
        # Hyperparameters
        self.lr = config["lr"]
        self.lambda_1 = config["lambda_1"] # Weight for Predictor Loss
        self.lambda_2 = config["lambda_2"] # Weight for Cost/Proximity Loss
        self.lambda_3 = config["lambda_3"] # Weight for Validity Loss

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
        exp_dims = [x for x in exp_dims]
        exp_dims[0] = exp_dims[0] + dec_dims[-1]
        
        self.explainer = nn.Sequential(
            MultilayerPerception(exp_dims),
            nn.Linear(exp_dims[-1], enc_dims[0]) # Output must match original input size
        )

    def forward(self, x):
        """The forward pass exactly as defined in the original CounterNetModel."""
        # Encode
        encoded_x = self.encoder_model(x)
        
        # Predict
        pred_hidden = self.predictor(encoded_x)
        y_logits = self.pred_linear(pred_hidden)
        y_hat = torch.sigmoid(y_logits)
        
        # Explain (Generate Counterfactual)
        # Concatenate encoder output and predictor hidden state
        concat_features = torch.cat((encoded_x, pred_hidden), dim=-1)
        c = self.explainer(concat_features)
        
        return torch.squeeze(y_hat, dim=-1), c

    def _loss_functions(self, x, c, y, y_hat):
        """Calculates the three distinct losses."""
        # 1. Prediction Loss (Binary Cross Entropy)
        l_1 = F.binary_cross_entropy(y_hat, y.float())
        
        # 2. Proximity/Cost Loss (Mean Squared Error between original and CF)
        l_2 = F.mse_loss(x, c)
        
        # 3. Validity Loss
        # What does the model predict for the generated counterfactual?
        c_y_hat, _ = self(c)
        
        # The target for validity is the exact opposite of what the model originally predicted
        y_prime = 1.0 - torch.round(y_hat.detach())
        l_3 = F.binary_cross_entropy(c_y_hat, y_prime)
        
        return l_1, l_2, l_3

    def configure_optimizers(self):
        """Two distinct optimizers for alternating training steps."""
        opt_1 = torch.optim.Adam(self.parameters(), lr=self.lr) # Predictor Optimizer
        opt_2 = torch.optim.Adam(self.parameters(), lr=self.lr) # Explainer Optimizer
        return [opt_1, opt_2]

    def training_step(self, batch, batch_idx):
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
        y_hat, c = self(x)
        _, l_2_new, l_3_new = self._loss_functions(x, c, y, y_hat)
        e_loss = (self.lambda_2 * l_2_new) + (self.lambda_3 * l_3_new)

        opt_explainer.zero_grad()
        self.manual_backward(e_loss)
        opt_explainer.step()

        # Optional: Log the losses so you can track them
        self.log("train_p_loss", p_loss, prog_bar=True)
        self.log("train_e_loss", e_loss, prog_bar=True)
