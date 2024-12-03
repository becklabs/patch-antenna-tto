import os
import torch    
import numpy as np
import matplotlib.pyplot as plt
from deepad.nn.utils import load_config, set_device, load_checkpoint
from deepad.nn.datasets import RectangularPatchDataset
from deepad.nn.vae import VAE
from deepad.nn.decoder import FeedForwardDecoder, ConvDecoder
from deepad.nn.encoder import FeedForwardEncoder, ConvEncoder

def prepare_datasets(design_params, freq_response, config, device):
    nx, nd = design_params.shape
    ny, nf, nc = freq_response.shape

    assert nx == ny, "design_params and s11_curves must have the same number of samples"
    assert (
        nf == config["model"]["n_freqs"]
    ), "s11_curves must have the same number of frequency points as specified in the config"
    assert (
        nd == config["model"]["n_design_params"]
    ), "design_params must have the same number of design parameters as specified in the config"
    assert nc == 2, "s11_curves must have 2 channels (value and frequency)"

    s11_curves = freq_response[:, :, 1]

    np.random.seed(config["seed"])
    train_inds = np.random.choice(
        nx, int(nx * config["data"]["train_split"]), replace=False
    )
    test_inds = np.setdiff1d(np.arange(nx), train_inds)

    train_dataset = RectangularPatchDataset(
        design_params=design_params[train_inds],
        s11_curves=s11_curves[train_inds],
        curves_device=device,
    )
    test_dataset = RectangularPatchDataset(
        design_params=design_params[test_inds],
        s11_curves=s11_curves[test_inds],
        design_params_scaler=train_dataset.design_params_scaler,
        s11_curves_scaler=train_dataset.s11_curves_scaler,
        curves_device=device,
    )

    return train_dataset, test_dataset

def plot_reconstructed_curves(model, val_dataset, config, scaler, device):
    num_examples = config["plots"]["num_examples"]
    indices = np.random.choice(len(val_dataset), num_examples, replace=False)
    model.eval()
    for i in indices:
        _, s11_curve_scaled = val_dataset[i]  # (s11_length,)
        s11_curve_scaled = s11_curve_scaled.unsqueeze(0).to(device)  # (1, s11_length)
        with torch.no_grad():
            output_scaled, _, _ = model(s11_curve_scaled)  # (1, s11_length)
        output_scaled = output_scaled.squeeze(-1)  # (1, s11_length)
        output_scaled = output_scaled.cpu().numpy().flatten()

        s11_curve_scaled = s11_curve_scaled.cpu().numpy()
        s11_curve = (
            scaler["s11_curves_scaler"].inverse_transform(s11_curve_scaled).flatten()
        )
        output_curve = (
            scaler["s11_curves_scaler"]
            .inverse_transform(output_scaled.reshape(1, -1))
            .flatten()
        )

        plt.figure(figsize=(12, 6))
        plt.plot(s11_curve, label="Original S11 (dB)", linewidth=2)
        plt.plot(output_curve, label="Reconstructed S11 (dB)", linestyle="--")
        plt.title(f"Example {i+1}")
        plt.xlabel("Frequency Index")
        plt.ylabel("S11 (dB)")
        plt.legend()
        plt.grid(True)

        # Save plot to a buffer
        plt.tight_layout()
        image_path = f"figs/reconstructed_example_{i+1}.png"
        plt.savefig(image_path)
        plt.close()

        # Optionally, remove the saved image file
        # os.remove(image_path)

def main():
    config_path = "config/train/s11_vae.yaml"
    config = load_config(config_path)
    device = set_device(config["device"])

    design_params = np.load(os.path.join(config["data"]["data_dir"], "design_params.npy"))
    freq_response = np.load(os.path.join(config["data"]["data_dir"], "freq_response.npy"))
    _, val_dataset = prepare_datasets(design_params, freq_response, config, device)

    # encoder = FeedForwardEncoder(
    #     input_dim=config["model"]["n_freqs"],
    #     latent_dim=config["model"]["latent_dim"]
    # )

    encoder = ConvEncoder(
        latent_dim=config["model"]["latent_dim"]
    )

    # decoder = FeedForwardDecoder(
    #     latent_dim=config["model"]["latent_dim"],
    #     output_length=config["model"]["n_freqs"]
    # )
    decoder = ConvDecoder(
        latent_dim=config["model"]["latent_dim"],
        output_length=config["model"]["n_freqs"]
    )
    model = VAE(encoder=encoder, decoder=decoder, latent_dim=config["model"]["latent_dim"]).to(device)

    checkpoint_path = os.path.join(config["checkpoint"]["checkpoint_dir"], config["checkpoint"]["resume_checkpoint"])
    model, _, _, _, _ = load_checkpoint(checkpoint_path, model, None, device)

    plot_reconstructed_curves(model, val_dataset, config, {"s11_curves_scaler": val_dataset.s11_curves_scaler}, device)

if __name__ == "__main__":
    main()