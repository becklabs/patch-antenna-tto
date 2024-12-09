import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..nn.datasets import RectangularPatchDataset
from ..nn.losses import masked_loss
from ..nn.vae import VAE


def sort_latents(
    vae: VAE,
    target_curve: torch.Tensor,
    dataset: RectangularPatchDataset,
    batch_size: int,
) -> torch.Tensor:
    """
    Compute latent vectors and corresponding losses for all curves in the dataset,
    then return the latents sorted by ascending loss.
    """

    MASK_DB_THRESHOLD = 0.1  # dB
    mask = torch.ones_like(target_curve)
    mask[torch.abs(target_curve) < MASK_DB_THRESHOLD] = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_losses = []
    all_latents = []

    with torch.no_grad():
        for _, curves in dataloader:
            latents, _ = vae.encode(curves)
            losses = masked_loss(
                pred=curves,
                target=target_curve.unsqueeze(0).expand_as(curves),
                mask=mask,
                loss_fn=nn.MSELoss(reduction="none"),
            ).mean(dim=1)

            all_losses.append(losses)
            all_latents.append(latents)

    all_losses = torch.cat(all_losses)  # shape: [N]
    all_latents = torch.cat(all_latents)  # shape: [N, latent_dim]

    sorted_indices = torch.argsort(all_losses)
    sorted_latents = all_latents[sorted_indices]

    return sorted_latents
