import numpy as np
import torch
import torch.nn as nn

from ..nn.losses import masked_loss
from ..nn.vae import VAE
from sklearn.preprocessing import StandardScaler

class S11SearchCriterion(nn.Module):
    """
    Criterion for searching through the distribution of S11 curves.
    """

    MASK_DB_THRESHOLD = 0.1  # dB

    def __init__(
        self,
        vae: VAE,
        target_curve: torch.Tensor,
        curve_scaler: StandardScaler,
        lambda_reg: float = 1.0,
        device: str = "cpu",
    ):
        super(S11SearchCriterion, self).__init__()

        self.lambda_reg = lambda_reg
        self.vae = vae
        self.curve_scaler = curve_scaler

        self.mask = torch.ones_like(target_curve)
        self.mask[torch.abs(target_curve) < self.MASK_DB_THRESHOLD] = 0

        target_curve_scaled = self.curve_scaler.transform(target_curve.cpu().numpy().reshape(1, -1))
        self.target_curve = torch.FloatTensor(target_curve_scaled.astype(np.float32)).squeeze().to(device)

        self.recon_criterion = nn.MSELoss(reduction="sum")  # Masked loss averages over mask

    def forward(self, z: torch.Tensor):
        curve = self.vae.decode(z)
        loss = masked_loss(
            pred=curve.squeeze(),
            target=self.target_curve.squeeze(),
            mask=self.mask,
            loss_fn=self.recon_criterion,
        )
        reg_loss = torch.sum(z**2)
        return loss + self.lambda_reg * reg_loss
