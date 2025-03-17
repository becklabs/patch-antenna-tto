import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from patchtto.nn.datasets import RectangularPatchDataset
from patchtto.nn.decoder import ConvDecoder
from patchtto.nn.encoder import ConvEncoder
from patchtto.nn.utils import load_checkpoint, load_config, load_data, set_device
from patchtto.nn.vae import VAE
from patchtto.search.initialization import (ClosestCurveInitialization,
                                          FixedRandomInitialization,
                                          KClosestCurveInitialization,
                                          RandomInitialization)
from patchtto.search.routines import find_curves
from patchtto.signal import generate_s11_curve


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Search for a S11 curve with the given config"
    )
    parser.add_argument(
        "--vae_config",
        default="config/train/s11_vae.yaml",
        help="Path to the VAE configuration file",
    )

    parser.add_argument(
        "--search_config",
        default="config/search/search_s11.yaml",
        help="Path to the search configuration file",
    )
    return parser.parse_args()


def create_model(config, device):
    encoder = ConvEncoder(
        latent_dim=config["model"]["latent_dim"],
    )

    decoder = ConvDecoder(
        latent_dim=config["model"]["latent_dim"],
        output_length=config["model"]["n_freqs"],
    )

    model = VAE(
        encoder=encoder, decoder=decoder, latent_dim=config["model"]["latent_dim"]
    )
    return model.to(device)


def create_animation(freqs, target_curve, all_telemetry, n_curves):
    """Create and save an animation of the optimization progress."""
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    for i in range(n_curves):
        (line,) = ax.plot([], [], alpha=0.5, label=f"Optimized Curve {i+1}")
        lines.append(line)
    (target_line,) = ax.plot(freqs, target_curve.flatten(), "r-", label="Target Curve")

    ax.set_xlim(freqs.min(), freqs.max())
    ax.set_ylim(target_curve.flatten().min() - 5, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("S11 (dB)")
    ax.set_title("Optimization Progress")
    ax.grid(True)
    ax.legend()

    def animate(frame):
        for i, (line, telemetry) in enumerate(zip(lines, all_telemetry)):
            line.set_data(freqs, telemetry["curves"][frame].flatten())
        return lines

    anim = animation.FuncAnimation(
        fig, animate, frames=len(all_telemetry[0]["curves"]), interval=20, blit=True
    )
    anim.save("figs/optimization_progress_4.gif", writer="pillow")
    plt.close()


def plot_curves_and_losses(target_curve, reconstructed_curves, criterion):
    """Plot the target vs reconstructed curves and their losses."""
    plt.figure(figsize=(15, 5))

    # Plot 1: Original vs Reconstructed curves
    plt.subplot(1, 2, 1)
    plt.plot(target_curve.flatten(), label="Target Curve", linewidth=2)
    for i, curve in enumerate(reconstructed_curves):
        plt.plot(curve.flatten(), "--", label=f"Found Curve {i+1}", alpha=0.5)

    # Plot 2: Final loss
    final_losses = [
        criterion(torch.FloatTensor(curve), torch.FloatTensor(target_curve)).item()
        for curve in reconstructed_curves
    ]
    loss_text = "\n".join(
        [f"Curve {i+1} Loss: {loss:.6f}" for i, loss in enumerate(final_losses)]
    )
    plt.subplot(1, 2, 2)
    plt.text(
        0.5,
        0.5,
        loss_text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()


def plot_latent_distribution(optimized_zs):
    """Plot histogram of the optimized latent vectors."""
    plt.figure()
    for i, z in enumerate(optimized_zs):
        plt.hist(z.cpu().numpy().flatten(), bins=50, alpha=0.5, label=f"Curve {i+1}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.vae_config)
    search_config = load_config(args.search_config)
    device = set_device(config["device"])
    design_params, freq_response = load_data(config)

    freqs = np.linspace(
        float(search_config["data"]["freq_start"]),
        float(search_config["data"]["freq_stop"]),
        int(search_config["data"]["n_freqs"]),
    )

    vae = create_model(config, device)

    checkpoint_path = os.path.join(
        config["checkpoint"]["checkpoint_dir"],
        config["checkpoint"]["resume_checkpoint"],
    )
    vae, _, _, y_scaler, _ = load_checkpoint(checkpoint_path, vae, None, device)
    vae.eval()

    dataset = RectangularPatchDataset(
        design_params=design_params,
        s11_curves=freq_response[:, :, 1],
        s11_curves_scaler=y_scaler,
        curves_device=device,
        design_device=device,
    )

    target_curve = generate_s11_curve(
        freq_range=freqs,
        resonant_freqs=[2.4e9],
        bandwidths=[100e6],
        depths_db=[-15],
    )

    random_init = RandomInitialization()
    fixed_init = FixedRandomInitialization()
    closest_init = ClosestCurveInitialization(
        vae=vae,
        target_curve=target_curve,
        dataset=dataset,
    )
    k_closest_init = KClosestCurveInitialization(
        vae=vae,
        target_curve=target_curve,
        dataset=dataset,
    )

    results = find_curves(
        vae=vae,
        ideal_curve=target_curve,
        curve_scaler=y_scaler,
        latent_dim=config["model"]["latent_dim"],
        device=device,
        z_init_strategy=k_closest_init,
        n_curves=search_config["hyperparameters"]["n_curves"],
        n_steps=search_config["hyperparameters"]["n_steps"],
        lr=search_config["hyperparameters"]["lr"],
        telemetry=True,
    )

    reconstructed_curves = [result["curve"] for result in results]
    optimized_zs = [result["latent"] for result in results]
    all_telemetry = [result["telemetry"] for result in results]

    criterion = nn.MSELoss()

    create_animation(
        freqs, target_curve, all_telemetry, search_config["hyperparameters"]["n_curves"]
    )
    plot_curves_and_losses(target_curve, reconstructed_curves, criterion)
    plot_latent_distribution(optimized_zs)
