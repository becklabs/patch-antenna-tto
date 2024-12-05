import torch
import numpy as np
import torch.nn as nn   
from deepad.nn.vae import VAE
from deepad.nn.losses import masked_loss
from typing import Optional

def find_in_distribution_curve(
    vae: VAE,
    ideal_curve: torch.Tensor,
    latent_dim: int,
    device: str,
    n_steps: int = 1000,
    lr: float = 0.01,
    criterion: nn.Module = nn.MSELoss(reduction="sum"),
    telemetry: bool = False,
):
    """
    Find the in-distribution curve closest to the ideal curve.

    Gradient-based optimization is used to find the latent vector that
    corresponds to the ideal curve.

    Args:
        vae: VAE model.
        ideal_curve: Ideal curve to find the latent vector for.
        latent_dim: Dimension of the latent space.
        n_steps: Number of optimization steps.
        lr: Learning rate.

    Returns:
        A tuple of the reconstructed curve and the optimized latent vector.
    """
    ideal_curve = ideal_curve.to(device)

    MASK_DB_THRESHOLD = 0.1 # dB

    mask = torch.ones_like(ideal_curve)
    mask[torch.abs(ideal_curve) < MASK_DB_THRESHOLD] = 0

    y_hat, z, telemetry = optimize_latent(
        vae=vae,
        y=ideal_curve,
        latent_dim=latent_dim,
        device=device,
        n_steps=n_steps,
        lr=lr,
        mask=mask,
        criterion=criterion,
        telemetry=telemetry,
    )

    return y_hat, z, telemetry


def optimize_latent(
    vae: VAE,
    y: torch.Tensor,
    latent_dim: int,
    device: str,
    n_steps: int = 1000,
    lr: float = 0.01,
    criterion: nn.Module = nn.MSELoss(),
    mask: Optional[torch.Tensor] = None,
    telemetry: bool = False,
):
    """
    Find the optimal latent vector that generates an output closest to the target tensor.

    Args:
        vae: VAE model.
        y: Target tensor.
        latent_dim: Dimension of the latent space.
        n_steps: Number of optimization steps.
        lr: Learning rate.
        criterion: Loss function.
        mask: Mask for masked loss on y.
        telemetry: If True, return dictionary with optimization history.
    """
    y = y.to(device)
    telemetry_data = {"curves": [], "losses": []} if telemetry else None
    
    # z = torch.zeros((1, latent_dim), requires_grad=True, device=device)
    # Random initialization
    z = torch.randn((1, latent_dim), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([z], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        y_hat = vae.decoder(z)
        if mask is not None:
            loss = masked_loss(y_hat.squeeze(), y.squeeze(), mask=mask, loss_fn=criterion)
        else:
            loss = criterion(y_hat.squeeze(), y.squeeze())
        
        if telemetry:
            telemetry_data["curves"].append(y_hat.detach().cpu().numpy())
            telemetry_data["losses"].append(loss.item())
            
        loss.backward()
        optimizer.step()

    y_hat = vae.decoder(z.detach()).squeeze()

    if telemetry:
        telemetry_data["curves"] = np.array(telemetry_data["curves"])
        telemetry_data["losses"] = np.array(telemetry_data["losses"])
    

    return y_hat.detach(), z.detach(), telemetry_data

