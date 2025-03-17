import os
import torch    
import numpy as np
import matplotlib.pyplot as plt
from patchtto.nn.utils import load_config, set_device, load_checkpoint, prepare_datasets
from patchtto.nn.vae import VAE
from patchtto.nn.decoder import FeedForwardDecoder, ConvDecoder
from patchtto.nn.encoder import FeedForwardEncoder, ConvEncoder


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

        plt.tight_layout()
        image_path = f"figs/reconstructed_example_{i+1}.png"
        plt.savefig(image_path)
        plt.close()


def main():
    config_path = "config/train/s11_vae.yaml"
    config = load_config(config_path)
    device = set_device(config["device"])

    design_params = np.load(os.path.join(config["data"]["data_dir"], "design_params.npy"))
    freq_response = np.load(os.path.join(config["data"]["data_dir"], "freq_response.npy"))
    _, val_dataset = prepare_datasets(design_params, freq_response, config, device, "cpu")

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