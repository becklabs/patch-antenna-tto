import argparse
import logging
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from patchtto.nn.utils import (
    load_checkpoint,
    load_config,
    load_data,
    prepare_datasets,
    create_dataloaders,
    set_device,
)
from patchtto.nn.vae import AdversarialVAE

from patchtto.nn.encoder import SimpleFeedForwardEncoder
from patchtto.nn.decoder import SimpleFeedForwardDecoder
from patchtto.nn.encoder import ConvEncoder
from patchtto.nn.decoder import ConvDecoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CVAE model for antenna design"
    )
    parser.add_argument(
        "--config",
        default="config/train/design_cvae.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()

def create_model(config, device):
    condition_head = ConvEncoder(
        latent_dim=int(config["model"]["decoder"]["condition_features"] / 2),
    )

    encoder = SimpleFeedForwardEncoder(
        x_dim=config["model"]["n_design_params"],
        latent_dim=config["model"]["latent_dim"] * 2,
    )

    decoder = SimpleFeedForwardDecoder(
        latent_dim=config["model"]["latent_dim"]
        + config["model"]["decoder"]["condition_features"],
        output_length=config["model"]["n_design_params"],
    )

    # discriminator = FeedForwardDecoder(
    #     latent_dim=config["model"]["latent_dim"],
    #     output_length=config["model"]["n_freqs"],
    #     output_channels=1,
    # )

    discriminator = ConvDecoder(
        latent_dim=config["model"]["latent_dim"],
        output_length=config["model"]["n_freqs"],
        output_channels=1,
        transpose=True,
    )

    cvae = AdversarialVAE(
        encoder=encoder,
        condition_head=condition_head,
        decoder=decoder,
        discriminator=discriminator,
        latent_dim=config["model"]["latent_dim"],
    )

    return cvae.to(device)


def plot_3d_comparison(actual_designs, predicted_designs, save_path=None):
    """
    Create a 3D scatter plot comparing actual vs predicted designs
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot actual designs in blue
    ax.scatter(
        actual_designs[:, 0],
        actual_designs[:, 1],
        actual_designs[:, 2],
        c='blue',
        marker='o',
        label='Actual Designs',
        alpha=0.6
    )
    
    # Plot predicted designs in red
    ax.scatter(
        predicted_designs[:, 0],
        predicted_designs[:, 1],
        predicted_designs[:, 2],
        c='red',
        marker='^',
        label='Predicted Designs',
        alpha=0.6
    )
    
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Parameter 3')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluate_model(model, test_dataloader, device, num_samples=100):
    """
    Evaluate model reconstruction capability
    """
    model.eval()
    actual_designs = []
    predicted_designs = []

    # Sample random latent vectors
    z = torch.randn(1, model.latent_dim).to(device)
    
    with torch.no_grad():
        for i, (x_batch, condition) in enumerate(test_dataloader):
            if i >= num_samples:
                break
                
            predicted_design = model.decode(z.repeat(x_batch.size(0), 1), condition)
            
            actual_designs.append(x_batch.cpu().numpy())
            predicted_designs.append(predicted_design.cpu().numpy())
    
    actual_designs = np.concatenate(actual_designs, axis=0)
    predicted_designs = np.concatenate(predicted_designs, axis=0)
    
    return actual_designs, predicted_designs


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(config_path=args.config)
    device = set_device(device_preference=config["device"])
    logger.info("Using device: %s", device)
    
    # Load data
    design_params, freq_response = load_data(config=config)
    train_dataset, test_dataset = prepare_datasets(
        design_params=design_params,
        freq_response=freq_response,
        config=config,
        curves_device=device,
        design_device=device,
    )

    _, test_dataloader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
    )
    
    model = create_model(config=config, device=device)
    checkpoint_path = os.path.join(config["checkpoint"]["checkpoint_dir"], config["checkpoint"]["resume_checkpoint"])

    model, _, _, scalers, _ = load_checkpoint(
        checkpoint_path, model, None, device
    )
    
    actual_designs, predicted_designs = evaluate_model(
        model, test_dataloader, device
    )
    
    if hasattr(test_dataset, 'design_params_scaler'):
        actual_designs = test_dataset.design_params_scaler.inverse_transform(
            actual_designs
        )
        predicted_designs = test_dataset.design_params_scaler.inverse_transform(
            predicted_designs
        )
    
    for i in range(actual_designs.shape[1]):
        mse = np.sqrt(np.mean((actual_designs[:, i] - predicted_designs[:, i]) ** 2))
        mae = np.mean(np.abs(actual_designs[:, i] - predicted_designs[:, i]))
        logger.info(f"Dimension {i}:")
        logger.info(f"  MSE: %.4f", mse)
        logger.info(f"  MAE: %.4f", mae)
    
    os.makedirs("figs", exist_ok=True)
    
    plot_3d_comparison(
        actual_designs,
        predicted_designs,
        save_path="figs/design_reconstruction_comparison.png"
    )
