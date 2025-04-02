import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class AdversarialVAELoss(nn.Module):
    def __init__(
        self,
        kld_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        recon_x_criterion: nn.Module = nn.MSELoss(reduction="mean"),
        recon_y_criterion: nn.Module = nn.MSELoss(reduction="mean"),
    ):
        super(AdversarialVAELoss, self).__init__()
        self.kld_weight = kld_weight
        self.adversarial_weight = adversarial_weight
        self.recon_x_criterion = recon_x_criterion
        self.recon_y_criterion = recon_y_criterion

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        recon_y: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_weight: Optional[float] = None,
        adversarial_weight: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Adversarial VAE loss: reconstruction loss + KL divergence + adversarial loss.
        https://archives.ismir.net/ismir2020/paper/000099.pdf

        Args:
            recon_x: Reconstructed input
            x: Original input
            recon_y: Reconstructed condition from discriminator
            y: True condition
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Tuple containing:
            - recon_x_loss: Reconstruction loss for x
            - recon_y_loss: Reconstruction loss for condition y
            - kld_loss: KL divergence loss (batchmean)
        """
        recon_x_loss = self.recon_x_criterion(recon_x, x)  # Reconstruction loss on x
        recon_y_loss = self.recon_y_criterion(recon_y, y)  # Reconstruction loss on y
        adversarial_loss = -recon_y_loss  # Adversarial loss on y

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0
        )  # KL divergence between q(z|x) and p(z)

        kld_weight = kld_weight or self.kld_weight
        adversarial_weight = adversarial_weight or self.adversarial_weight

        kld_loss = kld_loss * kld_weight
        adversarial_loss = adversarial_loss * adversarial_weight

        return recon_x_loss, recon_y_loss, adversarial_loss, kld_loss


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: torch.nn.Module = torch.nn.MSELoss(reduction="sum"),
) -> torch.Tensor:
    """
    Computes loss only on unmasked elements, normalized by mask size.

    Args:
        pred (torch.Tensor): Predictions tensor (B x D)
        target (torch.Tensor): Ground truth tensor (B x D)
        mask (torch.Tensor): Binary mask (1=valid, 0=ignored) (D)
        loss_fn (torch.nn.Module): Loss function, defaults to MSE

    Returns:
        torch.Tensor: Normalized loss value for masked elements
    """
    masked_pred = pred * mask
    masked_target = target * mask
    loss = loss_fn(masked_pred, masked_target)

    num_valid = mask.sum()
    if num_valid > 0:
        loss = loss * (mask.numel() / num_valid)

    return loss


def s11_reconstruction_loss(
    pred_S11: torch.Tensor,
    true_S11: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda_smooth: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute the reconstruction loss for S11 curves

    Args:
        pred_S11 (torch.Tensor): Predicted S11 curves (batch_size, M)
        true_S11 (torch.Tensor): True S11 curves (batch_size, M)
        lambda1 (float): Weight for first-order difference loss
        lambda2 (float): Weight for second-order difference loss
        lambda_smooth (float): Weight for smoothness loss
        reduction (str): Reduction method for the loss

    Returns:
        torch.Tensor: Total reconstruction loss
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
    """Neural network head for predicting mean and variance for NLL loss."""

    def __init__(self):
        super(NLLHead, self).__init__()

    def forward(self, x):
        mean = x[:, 0, :]
        variance = F.softplus(x[:, 1, :])
        return mean, variance


def beta_nll_loss(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute beta-NLL loss

    Args:
        mean (torch.Tensor): Predicted mean of shape B x D
        variance (torch.Tensor): Predicted variance of shape B x D
        target (torch.Tensor): Target of shape B x D
        beta (float): Parameter from range [0, 1] controlling relative
        weighting between data points, where `0` corresponds to
        high weight on low error points and `1` to an equal weighting.
        reduction (str): Reduction method for the loss

    Returns:
        torch.Tensor: Computed loss value (scalar)
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)

    if reduction == "mean":
        loss = loss.mean()  # Shape: scalar
    else:
        loss = loss.sum()  # Shape: scalar

    return loss

def gaussian_nll(y_pred, mean, logvar):
    """
    Negative log likelihood for Gaussian distributions when using log-variance.
    
    Args:
        y_pred: Predicted values
        mean: Mean of the Gaussian distribution
        logvar: Log-variance of the Gaussian distribution
    
    Returns:
        Negative log likelihood
    """
    var = torch.exp(logvar)
    return 0.5 * (torch.log(2 * torch.pi) + logvar + ((y_pred - mean)**2 / var))