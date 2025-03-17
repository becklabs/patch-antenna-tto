import argparse
import logging
import os
import torch
import torch.nn as nn

import wandb
from patchtto.nn.decoder import ConvDecoder, FeedForwardDecoder
from patchtto.nn.losses import (
    s11_reconstruction_loss,
    beta_nll_loss,
    NLLHead,
)

from patchtto.nn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

from patchtto.nn.utils import (
    create_dataloaders,
    load_checkpoint,
    load_config,
    load_data,
    prepare_datasets,
    save_checkpoint,
    set_device,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a forward model with the given config"
    )
    parser.add_argument(
        "--config",
        default="config/train/surrogate_nll.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def create_model(config, device):
    decoder = ConvDecoder(
        latent_dim=config["model"]["n_design_params"],
        output_length=config["model"]["n_freqs"],
        output_channels=2,
        transpose=False
    )
    model = nn.Sequential(decoder, NLLHead())
    return model.to(device)


def train(
    model,
    train_dataloader,
    test_dataloader,
    config,
    device,
    start_epoch=0,
):

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["hyperparameters"]["learning_rate"])
    )
    logger.info("Model: %s", model)
    logger.info("Optimizer: %s", optimizer)


    recon_criterion = lambda outputs, target: s11_reconstruction_loss(
        outputs, target,
        lambda1=config["hyperparameters"]["lambda1"],
        lambda2=config["hyperparameters"]["lambda2"],
        lambda_smooth=config["hyperparameters"]["lambda_smooth"],
    )

    if config["checkpoint"]["resume_from_checkpoint"]:
        logger.info("Resuming from checkpoint: %s", config['checkpoint']['resume_checkpoint'])
        checkpoint_path = os.path.join(
            config["checkpoint"]["checkpoint_dir"],
            config["checkpoint"]["resume_checkpoint"],
        )
        model, optimizer, _, _, start_epoch = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )

    for epoch in range(start_epoch, config["hyperparameters"]["num_epochs"]):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0

        for design, curve in train_dataloader:
            optimizer.zero_grad()
            output = model(design)
            mean, variance = output
            loss = beta_nll_loss(mean=mean, variance=variance, target=curve, beta=config["hyperparameters"]["beta"])

            with torch.no_grad():
                recon_loss = recon_criterion(mean, curve)


            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon_loss += recon_loss.item()

        avg_train_loss = running_loss / len(train_dataloader.dataset)
        avg_train_recon_loss = running_recon_loss / len(train_dataloader.dataset)
        avg_test_loss, avg_test_recon_loss = evaluate(
            model, test_dataloader, recon_criterion, beta=config["hyperparameters"]["beta"]
        )

        if epoch % config["wandb"]["log_interval"] == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                    "train_recon_loss": avg_train_recon_loss,
                    "test_recon_loss": avg_test_recon_loss,
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
            "Epoch %d: Train Loss: %.4f, Test Loss: %.4f, Train Recon Loss: %.4f, Test Recon Loss: %.4f",
            epoch,
            avg_train_loss,
            avg_test_loss,
            avg_train_recon_loss,
            avg_test_recon_loss,
        )


def evaluate(model, dataloader, recon_criterion, beta):
    model.eval()
    running_loss = 0.0
    running_recon_loss = 0.0

    with torch.no_grad():
        for design, curve in dataloader:
            output = model(design)
            mean, variance = output
            loss = beta_nll_loss(mean=mean, variance=variance, target=curve, beta=beta)

            running_loss += loss.item() 

            with torch.no_grad():
                recon_loss = recon_criterion(mean, curve)
                running_recon_loss += recon_loss.item()

    avg_loss = running_loss / len(dataloader.dataset)
    avg_recon_loss = running_recon_loss / len(dataloader.dataset)

    return avg_loss, avg_recon_loss


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
        s11_curves_scaler=StandardScaler(),
        design_params_scaler=MinMaxScaler(),
    )

    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
    )

    model = create_model(config=config, device=device)

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        config=config,
        device=device,
    )

    wandb.finish()
