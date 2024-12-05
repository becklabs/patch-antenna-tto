import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from deepad.nn.decoder import ConvDecoder
from deepad.nn.encoder import ConvEncoder
from deepad.nn.utils import (load_checkpoint, load_config, prepare_datasets,
                             set_device)
from deepad.nn.vae import VAE
from deepad.optim import find_in_distribution_curve
from deepad.signal import generate_s11_curve


def find_closest_curve(dataset, target_curve, device, criterion=nn.MSELoss()):
    """Find the closest curve in the dataset to the target curve."""
    best_loss = float("inf")
    best_curve = None
    best_idx = None

    for idx in range(len(dataset)):
        design, curve = dataset[idx]
        curve = curve.to(device)
        loss = criterion(curve, target_curve)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_curve = curve
            best_idx = idx

    return best_curve, best_idx, best_loss


def main():
    config_path = "config/train/s11_vae.yaml"
    config = load_config(config_path)
    device = set_device(config["device"])

    design_params = np.load(
        os.path.join(config["data"]["data_dir"], "design_params.npy")
    )
    freq_response = np.load(
        os.path.join(config["data"]["data_dir"], "freq_response.npy")
    )

    freqs = freq_response[0, :, 0]

    encoder = ConvEncoder(latent_dim=config["model"]["latent_dim"])
    decoder = ConvDecoder(
        latent_dim=config["model"]["latent_dim"],
        output_length=config["model"]["n_freqs"],
    )
    model = VAE(
        encoder=encoder, decoder=decoder, latent_dim=config["model"]["latent_dim"]
    ).to(device)

    checkpoint_path = os.path.join(
        config["checkpoint"]["checkpoint_dir"],
        config["checkpoint"]["resume_checkpoint"],
    )
    model, _, _, _, _ = load_checkpoint(checkpoint_path, model, None, device)
    model.eval()

    _, val_dataset = prepare_datasets(design_params, freq_response, config, device)

    target_curve = generate_s11_curve(freqs, [2.4e9], [100e6], [-40])
    target_curve = torch.FloatTensor(target_curve).to(device)

    n_steps = 300
    lr = 0.01
    criterion = nn.MSELoss()

    reconstructed_curve, optimized_z, telemetry = find_in_distribution_curve(
        vae=model,
        ideal_curve=target_curve,
        latent_dim=config["model"]["latent_dim"],
        device=device,
        n_steps=n_steps,
        lr=lr,
        criterion=criterion,
        telemetry=True,
    )

    plt.plot(telemetry["losses"])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot([], [], "b-", label="Optimized Curve")
    (target_line,) = ax.plot(
        freqs, target_curve.cpu().numpy().flatten(), "r-", label="Target Curve"
    )
    ax.set_xlim(freqs.min(), freqs.max())
    ax.set_ylim(-40, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("S11 (dB)")
    ax.set_title("Optimization Progress")
    ax.grid(True)
    ax.legend()

    def animate(frame):
        line.set_data(freqs, telemetry["curves"][frame].flatten())
        return (line,)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(telemetry["curves"]), interval=20, blit=True
    )

    anim.save("figs/optimization_progress.gif", writer="pillow")
    plt.close()

    # Find closest curve in dataset
    # reconstructed_curve, _, closest_loss = find_closest_curve(val_dataset, target_curve, device, criterion)

    plt.figure(figsize=(15, 5))

    # Plot 1: Original vs Reconstructed curves
    plt.subplot(1, 2, 1)
    plt.plot(target_curve.cpu().numpy().flatten(), label="Target Curve", linewidth=2)
    plt.plot(reconstructed_curve.cpu().numpy().flatten(), "--", label="Found Curve")
    plt.title("Target vs Found Curve")
    plt.xlabel("Frequency Index")
    plt.ylabel("S11 (dB)")
    plt.legend()
    plt.grid(True)

    # Plot 2: Final loss
    final_loss = criterion(reconstructed_curve, target_curve).item()
    plt.subplot(1, 2, 2)
    plt.text(
        0.5,
        0.5,
        f"Final Loss: {final_loss:.6f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        fontsize=12,
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Plot 3: Histogram of the optimized z
    plt.hist(optimized_z.cpu().numpy().flatten(), bins=50)
    plt.show()


if __name__ == "__main__":
    main()
