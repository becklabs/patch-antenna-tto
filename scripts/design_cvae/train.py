import argparse
import logging
import os

import torch
import torch.nn as nn

import wandb
from deepad.nn.decoder import ConvDecoder, FeedForwardDecoder
from deepad.nn.encoder import ConvEncoder, FeedForwardEncoder
from deepad.nn.losses import (VAELoss, s11_reconstruction_loss,
                              sigmoid_annealing)
from deepad.nn.utils import (create_dataloaders, load_checkpoint, load_config,
                             load_data, prepare_datasets, save_checkpoint,
                             set_device)
from deepad.nn.vae import CVAE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a CVAE model for antenna design with the given config"
    )
    parser.add_argument(
        "--config",
        default="config/train/design_cvae.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def create_model(config, device):
    condition_head = ConvEncoder(
        latent_dim=int(config["model"]["encoder"]["condition_hidden_dim"] / 2),
    )

    encoder = FeedForwardEncoder(
        input_dim=config["model"]["n_design_params"]
        + config["model"]["encoder"]["condition_hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    )

    decoder = FeedForwardDecoder(
        latent_dim=config["model"]["latent_dim"]
        + config["model"]["encoder"]["condition_hidden_dim"],
        output_length=config["model"]["n_design_params"],
    )

    model = CVAE(
        encoder=encoder,
        condition_head=condition_head,
        decoder=decoder,
        latent_dim=config["model"]["latent_dim"],
    )
    return model.to(device)


def train(
    model,
    train_dataloader,
    test_dataloader,
    config,
    device,
    start_epoch=0,
):
    kld_weight = config["hyperparameters"]["kld_weight"]
    max_kld_weight = config["hyperparameters"]["kld_weight"]
    min_kld_weight = 0.0
    n_warmup_epochs = config["hyperparameters"].get("kld_warmup_epochs", 0)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["hyperparameters"]["learning_rate"])
    )

    recon_criterion = nn.MSELoss(reduction="sum")
    criterion = VAELoss(
        kld_weight=config["hyperparameters"]["kld_weight"],
        recon_criterion=recon_criterion,
    )

    if config["checkpoint"]["resume_from_checkpoint"]:
        checkpoint_path = os.path.join(
            config["checkpoint"]["checkpoint_dir"],
            config["checkpoint"]["resume_checkpoint"],
        )
        model, optimizer, _, _, start_epoch = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )

    for epoch in range(start_epoch, config["hyperparameters"]["num_epochs"]):
        kld_weight = sigmoid_annealing(
            epoch, n_warmup_epochs, min_kld_weight, max_kld_weight
        )

        model.train()
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_total_loss = 0.0

        for x_batch, condition in train_dataloader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch, condition)
            recon_batch = recon_batch.squeeze(-1)
            recon_loss, kl_loss, total_loss = criterion(
                recon_batch, x_batch, mu, logvar, kld_weight
            )
            total_loss.backward()
            optimizer.step()

            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_loss.item()
            running_total_loss += total_loss.item()

        avg_train_recon_loss = running_recon_loss / len(train_dataloader.dataset)
        avg_train_kl_loss = running_kl_loss / len(train_dataloader.dataset)
        avg_train_loss = running_total_loss / len(train_dataloader.dataset)

        avg_test_recon_loss, avg_test_kl_loss, avg_test_loss = evaluate(
            model, test_dataloader, criterion, kld_weight
        )

        if epoch % config["wandb"]["log_interval"] == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_recon_loss": avg_train_recon_loss,
                    "train_kl_loss": avg_train_kl_loss,
                    "train_loss": avg_train_loss,
                    "test_recon_loss": avg_test_recon_loss,
                    "test_kl_loss": avg_test_kl_loss,
                    "test_loss": avg_test_loss,
                    "kld_weight": kld_weight,
                }
            )

        if (epoch + 1) % config["wandb"]["save_interval"] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                X_scaler=train_dataloader.dataset.design_params_scaler,
                y_scaler=train_dataloader.dataset.s11_curves_scaler,
                config=config,
            )

        logger.info(
            "Epoch %d: Train Loss: %.4f (Recon: %.4f, KL: %.4f), Test Loss: %.4f (Recon: %.4f, KL: %.4f)",
            epoch,
            avg_train_loss,
            avg_train_recon_loss,
            avg_train_kl_loss,
            avg_test_loss,
            avg_test_recon_loss,
            avg_test_kl_loss,
        )


def evaluate(model, dataloader, criterion, kld_weight):
    model.eval()
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    running_total_loss = 0.0

    with torch.no_grad():
        for x_batch, condition in dataloader:
            recon_batch, mu, logvar = model(x_batch, condition)
            recon_batch = recon_batch.squeeze(-1)
            recon_loss, kl_loss, total_loss = criterion(
                recon_batch, x_batch, mu, logvar, kld_weight
            )

            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_loss.item()
            running_total_loss += total_loss.item()

    avg_recon_loss = running_recon_loss / len(dataloader.dataset)
    avg_kl_loss = running_kl_loss / len(dataloader.dataset)
    avg_total_loss = running_total_loss / len(dataloader.dataset)

    return avg_recon_loss, avg_kl_loss, avg_total_loss


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(config_path=args.config)
    device = set_device(device_preference=config["device"])
    logger.info("Using device: %s", device)

    wandb.init(
        project=config["wandb"]["project"],
        config=config,
        resume=config["checkpoint"]["resume_from_checkpoint"],
    )

    design_params, freq_response = load_data(config=config)
    train_dataset, test_dataset = prepare_datasets(
        design_params=design_params,
        freq_response=freq_response,
        config=config,
        curves_device=device,
        design_device=device,
    )
    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
    )

    model = create_model(config=config, device=device)
    wandb.watch(model)

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        config=config,
        device=device,
    )

    wandb.finish()
