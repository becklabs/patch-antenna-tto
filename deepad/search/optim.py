from typing import Optional, Tuple

import torch
import torch.nn as nn


def optimize_latent(
    z_init: torch.Tensor,
    criterion: nn.Module,
    device: str,
    n_steps: int = 1000,
    lr: float = 0.01,
    telemetry: bool = False,
) -> Tuple[torch.Tensor, Optional[dict]]:
    """
    Optimize a latent vector according to the given criterion.

    Args:
        z_init: Initial latent vector.
        criterion: Loss function on the latent vector.
        device: Device to run the optimization on.
        n_steps: Number of optimization steps.
        lr: Learning rate.
        telemetry: If True, return dictionary with optimization history.
    """

    telemetry_data = {"latents": [], "losses": []} if telemetry else None

    z = nn.Parameter(z_init.to(device))
    optimizer = torch.optim.Adam([z], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = criterion(z)
        loss.backward()
        optimizer.step()

        if telemetry:
            telemetry_data["losses"].append(loss.item())
            telemetry_data["latents"].append(z.detach().clone())

    return z.detach(), telemetry_data
