import torch
import torch.nn as nn

from ..nn.losses import masked_loss
from ..nn.vae import VAE


class S11SearchCriterion(nn.Module):
    """
    Criterion for searching through the distribution of S11 curves.
    """

    MASK_DB_THRESHOLD = 0.1  # dB

    def __init__(
        self,
        vae: VAE,
        target_curve: torch.Tensor,
        lambda_reg: float = 1.0,
    ):
        super(S11SearchCriterion, self).__init__()

        self.lambda_reg = lambda_reg
        self.vae = vae
        self.target_curve = target_curve

        self.mask = torch.ones_like(self.target_curve)
        self.mask[torch.abs(self.target_curve) < self.MASK_DB_THRESHOLD] = 0

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
