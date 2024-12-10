import torch
import numpy as np

from typing import Optional, Tuple

from ..nn.vae import VAE
from .optim import optimize_latent

from .initialization import InitializationStrategy, RandomInitialization
from .criterion import S11SearchCriterion


def find_curve(
    vae: VAE,
    ideal_curve: np.ndarray,
    curve_scaler: object,
    latent_dim: int,
    device: str,
    z_init_strategy: InitializationStrategy = RandomInitialization(),
    n_curves: int = 1,
    n_steps: int = 1000,
    lr: float = 0.01,
    lambda_reg: float = 1.0,
    telemetry: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
    """
    Find the in-distribution curve closest to the ideal curve.

    Gradient-based optimization is used to find the latent vector that
    corresponds to the ideal curve.

    Args:
        vae: VAE model.
        ideal_curve: Ideal curve to find the latent vector for.
        latent_dim: Dimension of the latent space.
        device: Device to run the optimization on.
        n_curves: Number of
        n_steps: Number of optimization steps.
        lr: Learning rate.
        lambda_reg: Regularization term weight.
        telemetry: If True, return dictionary with optimization history.

    Returns:
        A list of dictionaries, each containing the reconstructed curve, the optimized latent vector, and the telemetry dictionary.
    """
    ideal_curve = torch.FloatTensor(ideal_curve).to(device)

    criterion = S11SearchCriterion(
        vae=vae,
        target_curve=ideal_curve,
        curve_scaler=curve_scaler,
        lambda_reg=lambda_reg,
        device=device,
    )

    z_init = z_init_strategy(latent_dim, n_curves)
    assert z_init.shape == (
        n_curves,
        latent_dim,
    ), f"Expected shape (n_curves, latent_dim), got {z_init.shape}"

    results = []
    for i in range(n_curves):
        z, telemetry_dict = optimize_latent(
            z_init=z_init[i],
            criterion=criterion,
            device=device,
            n_steps=n_steps,
            lr=lr,
            telemetry=telemetry,
        )

        y_hat = vae.decode(z).squeeze()

        if telemetry_dict is not None:
            with torch.no_grad():
                telemetry_dict["curves"] = np.array(
                    [
                        vae.decode(z).squeeze().cpu().numpy()
                        for z in telemetry_dict["latents"]
                    ]
                )

        results.append(
            {
                "curve": 
                    curve_scaler.inverse_transform(
                    y_hat.detach().cpu().numpy().reshape(1, -1)
                ).flatten()
                .astype(np.float32),
                "latent": z.detach(),
                "telemetry": telemetry_dict,
            }
        )

    return results
