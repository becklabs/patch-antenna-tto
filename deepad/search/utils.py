import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union

from ..nn.datasets import RectangularPatchDataset
from ..nn.losses import masked_loss
from ..nn.vae import VAE


def target_curve_mask(
    target_curve: Union[np.ndarray, torch.Tensor], threshold: float = 0.1
) -> torch.Tensor:
    """
    Create a mask for the target curve based on the dB threshold.
    """
    if isinstance(target_curve, np.ndarray):
        mask_np = np.ones_like(target_curve)
        mask_np[np.abs(target_curve) < threshold] = 0
        return torch.from_numpy(mask_np.astype(np.float32))

    elif isinstance(target_curve, torch.Tensor):
        mask = torch.ones_like(target_curve)
        mask[torch.abs(target_curve) < threshold] = 0
        return mask
    else:
        raise ValueError(f"Unsupported type for target_curve: {type(target_curve)}")


def sort_latents(
    vae: VAE,
    target_curve: np.ndarray,
    dataset: RectangularPatchDataset,
    batch_size: int,
) -> torch.Tensor:
    """
    Compute latent vectors and corresponding losses for all curves in the dataset,
    then return the latents sorted by ascending loss.
    """

    device = next(vae.parameters()).device

    MASK_DB_THRESHOLD = 0.1  # dB
    mask = target_curve_mask(target_curve=target_curve, threshold=MASK_DB_THRESHOLD).to(device)

    target_curve_scaled = dataset.s11_curves_scaler.transform(
        target_curve.reshape(1, -1)
    )
    target_curve_scaled = torch.FloatTensor(target_curve_scaled).squeeze().to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_losses = []
    all_latents = []

    with torch.no_grad():
        for _, curves in dataloader:
            latents, _ = vae.encode(curves)
            losses = masked_loss(
                pred=curves,
                target=target_curve_scaled.unsqueeze(0).expand_as(curves),
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
