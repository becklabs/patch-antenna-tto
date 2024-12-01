import torch
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def sigmoid_annealing(epoch, n_warmup_epochs, min_weight, max_weight):
    if n_warmup_epochs == 0:  # No scheduling case
        return max_weight
    x = 10 * (epoch - 0.5 * n_warmup_epochs) / n_warmup_epochs
    sigmoid = 1 / (1 + np.exp(-x))
    return min_weight + (max_weight - min_weight) * sigmoid

def s11_reconstruction_loss(
    pred_S11, true_S11, lambda1=1.0, lambda2=1.0, lambda_smooth=1.0, reduction="mean"
):
    """
    Compute the reconstruction loss for S11 curves

    Args:
        pred_S11 (torch.Tensor): Predicted S11 curves (batch_size, M)
        true_S11 (torch.Tensor): True S11 curves (batch_size, M)
        lambda1 (float): Weight for first-order difference loss
        lambda2 (float): Weight for second-order difference loss
        lambda_smooth (float): Weight for smoothness loss
        reduction (str): Reduction method for the loss
    """
    L_MSE = F.mse_loss(pred_S11, true_S11, reduction=reduction)
    pred_diff1 = pred_S11[:, 1:] - pred_S11[:, :-1]  # (batch_size, M-1)
    true_diff1 = true_S11[:, 1:] - true_S11[:, :-1]
    L_first = F.mse_loss(pred_diff1, true_diff1, reduction=reduction)
    pred_diff2 = (
        pred_S11[:, 2:] - 2 * pred_S11[:, 1:-1] + pred_S11[:, :-2]
    )  # (batch_size, M-2)
    true_diff2 = true_S11[:, 2:] - 2 * true_S11[:, 1:-1] + true_S11[:, :-2]
    L_second = F.mse_loss(pred_diff2, true_diff2, reduction=reduction)
    L_smooth = torch.mean(pred_diff2**2)
    total_loss = (
        L_MSE + lambda1 * L_first + lambda2 * L_second + lambda_smooth * L_smooth
    )
    return total_loss


class NLLHead(nn.Module):
    def __init__(self):
        super(NLLHead, self).__init__()

    def forward(self, x):
        mean = x[:, 0, :]
        variance = F.softplus(x[:, 1, :])  # Ensure variance is positive
        return mean, variance


def beta_nll_loss(mean, variance, target, beta=0.5, reduction="mean"):
    """
    Compute beta-NLL loss

    Args:
        mean (torch.Tensor): Predicted mean of shape B x D
        variance (torch.Tensor): Predicted variance of shape B x D
        target (torch.Tensor): Target of shape B x D
        beta (float): Parameter from range [0, 1] controlling relative
        weighting between data points, where `0` corresponds to
        high weight on low error points and `1` to an equal weighting.
    Returns:
        Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)

    if reduction == "mean":
        loss = loss.mean()  # Shape: scalar
    else:
        loss = loss.sum()  # Shape: scalar

    return loss


def vae_loss(mu, logvar, recon_batch, s11_input, beta=1.0):
    """
    Compute VAE loss

    Args:
        mu (torch.Tensor): Predicted mean of shape B x D
        logvar (torch.Tensor): Predicted log variance of shape B x D
        recon_batch (torch.Tensor): Reconstructed S11 curves of shape B x M
        s11_input (torch.Tensor): True S11 curves of shape B x M
        beta (float): Weight for KL divergence
    Returns:
        recon_loss (torch.Tensor): Reconstruction loss
        kl_divergence (torch.Tensor): KL divergence
        loss (torch.Tensor): Total loss
    """
    recon_loss = nn.functional.mse_loss(recon_batch, s11_input, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_divergence
    return recon_loss, kl_divergence, loss


class VAELoss(nn.Module):
    def __init__(
        self,
        kld_weight: float = 1.0,
        recon_criterion: nn.Module = nn.MSELoss(reduction="mean"),
    ):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight
        self.recon_criterion = recon_criterion

    def forward(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_weight: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss: reconstruction loss + KL divergence.

        Args:
            recon: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Tuple containing:
            - recon_loss: Reconstruction loss
            - kld_loss: KL divergence loss (batchmean)
            - total_loss: Combined loss (reconstruction + weighted KL divergence)
        """
        recon_loss = self.recon_criterion(recon, x)  # Reconstruction loss
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0
        )  # KL divergence between q(z|x) and p(z)

        if kld_weight is not None:
            kld_loss = kld_weight * kld_loss
        else:
            kld_loss = self.kld_weight * kld_loss
        total_loss = recon_loss + kld_loss

        return recon_loss, kld_loss, total_loss
