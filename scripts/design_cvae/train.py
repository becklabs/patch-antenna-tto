import argparse
import logging
import os

import torch
import torch.nn as nn

import wandb
from patchtto.nn.decoder import ConvDecoder, SimpleFeedForwardDecoder
from patchtto.nn.encoder import ConvEncoder, SimpleFeedForwardEncoder
from patchtto.nn.losses import AdversarialVAELoss
from patchtto.nn.utils import (create_dataloaders, load_checkpoint, load_config,
                             load_data, prepare_datasets, save_checkpoint,
                             set_device, sigmoid_annealing)
from patchtto.nn.vae import AdversarialVAE

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

def tracker_avg(tracker, key):
    return tracker[key]['sum'] / max(tracker[key]['count'], 1)

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

    optimizer_dec = torch.optim.Adam(
        model.decoder_params(), lr=float(config["hyperparameters"]["learning_rate"])
    )

    optimizer_disc = torch.optim.Adam(
        model.discriminator_params(), lr=float(config["hyperparameters"]["learning_rate"])
    )

    optimizer_enc = torch.optim.Adam(
        model.encoder_params(), lr=float(config["hyperparameters"]["learning_rate"])
    )

    criterion = AdversarialVAELoss(
        kld_weight=config["hyperparameters"]["kld_weight"],
        adversarial_weight=config["hyperparameters"]["adversarial_weight"],
        recon_x_criterion=nn.MSELoss(reduction="mean"),
        recon_y_criterion=nn.MSELoss(reduction="mean"),
    )

    if config["checkpoint"]["resume_from_checkpoint"]:
        checkpoint_path = os.path.join(
            config["checkpoint"]["checkpoint_dir"],
            config["checkpoint"]["resume_checkpoint"],
        )
        model, _, _, _, start_epoch = load_checkpoint(
            checkpoint_path, model, None, device
        )

    for epoch in range(start_epoch, config["hyperparameters"]["num_epochs"]):
        loss_trackers = {
            'recon': {'sum': 0.0, 'count': 0},
            'kl': {'sum': 0.0, 'count': 0},
            'y_hat': {'sum': 0.0, 'count': 0},
            'adversarial': {'sum': 0.0, 'count': 0},
            'encoder': {'sum': 0.0, 'count': 0},
            'vae': {'sum': 0.0, 'count': 0},
        }

        vae_updates = 0
        discriminator_updates = 0
        encoder_updates = 0

        kld_weight = sigmoid_annealing(
            epoch, n_warmup_epochs, min_kld_weight, max_kld_weight
        )

        model.train()

        for i, (x_batch, condition) in enumerate(train_dataloader):
            optimizer_dec.zero_grad()
            optimizer_enc.zero_grad()
            optimizer_disc.zero_grad()

            recon_x, y_hat, mu, logvar = model(x_batch, condition)

            recon_x_loss, y_hat_loss, adversarial_loss, kl_loss = criterion(
                recon_x, x_batch, y_hat, condition, mu, logvar, kld_weight=kld_weight
            )

            loss_trackers['recon']['sum'] += recon_x_loss.item()
            loss_trackers['recon']['count'] += 1

            loss_trackers['kl']['sum'] += kl_loss.item()
            loss_trackers['kl']['count'] += 1

            loss_trackers['y_hat']['sum'] += y_hat_loss.item()
            loss_trackers['y_hat']['count'] += 1

            loss_trackers['adversarial']['sum'] += adversarial_loss.item()
            loss_trackers['adversarial']['count'] += 1

            # VAE Update
            if (i // 10) % 2 == 0: # i = n + 0,1,2,3,4,5,6,7,8,9
                vae_loss = recon_x_loss + kl_loss
                vae_loss.backward()
                optimizer_dec.step()
                optimizer_enc.step()

                loss_trackers['vae']['sum'] += vae_loss.item()
                loss_trackers['vae']['count'] += 1

                vae_updates += 1

            # Discriminator Update
            elif (i // 5) % 2 == 0: # i = n + 10,11,12,13,14
                y_hat_loss.backward()
                optimizer_disc.step()

                discriminator_updates += 1

            # Encoder Update
            else: # i = n + 15,16,17,18,19
                encoder_loss = adversarial_loss + kl_loss
                encoder_loss.backward()
                optimizer_enc.step()

                loss_trackers['encoder']['sum'] += encoder_loss.item()
                loss_trackers['encoder']['count'] += 1

                encoder_updates += 1


        avg_train_recon_loss = tracker_avg(loss_trackers, 'recon')
        avg_train_kl_loss = tracker_avg(loss_trackers, 'kl')
        avg_train_y_hat_loss = tracker_avg(loss_trackers, 'y_hat')
        avg_train_adversarial_loss = tracker_avg(loss_trackers, 'adversarial')
        avg_train_encoder_loss = tracker_avg(loss_trackers, 'encoder')
        avg_train_vae_loss = tracker_avg(loss_trackers, 'vae')

        test_metrics = evaluate(
            model, test_dataloader, criterion, kld_weight
        )

        if epoch % config["wandb"]["log_interval"] == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_recon_loss": avg_train_recon_loss,
                    "train_kl_loss": avg_train_kl_loss,
                    "train_y_hat_loss": avg_train_y_hat_loss,
                    "train_adversarial_loss": avg_train_adversarial_loss,
                    "train_encoder_loss": avg_train_encoder_loss,
                    "train_vae_loss": avg_train_vae_loss,
                    "test_recon_loss": test_metrics["recon_loss"],
                    "test_kl_loss": test_metrics["kl_loss"],
                    "test_y_hat_loss": test_metrics["y_hat_loss"],
                    "test_adversarial_loss": test_metrics["adversarial_loss"],
                    "kld_weight": kld_weight,
                }
            )
        logger.info(
            "Epoch %d: Train VAE Loss: %.4f (Recon: %.4f, KL: %.4f), "
            "Y-hat Loss: %.4f, Adversarial Loss: %.4f, Encoder Loss: %.4f | "
            "Test VAE Loss: %.4f (Recon: %.4f, KL: %.4f), Y-hat Loss: %.4f, Adversarial Loss: %.4f | "
            "Update Spread: (%d/%d/%d)",
            epoch,
            avg_train_vae_loss,
            avg_train_recon_loss,
            avg_train_kl_loss,
            avg_train_y_hat_loss,
            avg_train_adversarial_loss,
            avg_train_encoder_loss,
            test_metrics["vae_loss"],
            test_metrics["recon_loss"],
            test_metrics["kl_loss"],
            test_metrics["y_hat_loss"],
            test_metrics["adversarial_loss"],
            int(100 * vae_updates / len(train_dataloader)),
            int(100 * discriminator_updates / len(train_dataloader)),
            int(100 * encoder_updates / len(train_dataloader)),
        )

        if (epoch + 1) % config["wandb"]["save_interval"] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer_dec,
                epoch=epoch,
                X_scaler=train_dataloader.dataset.design_params_scaler,
                y_scaler=train_dataloader.dataset.s11_curves_scaler,
                config=config,
            )


def evaluate(model, dataloader, criterion, kld_weight):
    model.eval()
    running_recon_loss = 0.0
    running_y_hat_loss = 0.0
    running_adversarial_loss = 0.0

    running_kl_loss = 0.0
    running_vae_loss = 0.0
    running_encoder_loss = 0.0

    with torch.no_grad():
        for x_batch, condition in dataloader:
            recon_x, y_hat, mu, logvar = model(x_batch, condition)
            
            recon_x_loss, y_hat_loss, adversarial_loss, kl_loss = criterion(
                recon_x, x_batch, y_hat, condition, mu, logvar, kld_weight=kld_weight
            )
            
            vae_loss = recon_x_loss + kl_loss
            encoder_loss = adversarial_loss + kl_loss

            running_recon_loss += recon_x_loss.item()
            running_kl_loss += kl_loss.item()
            running_y_hat_loss += y_hat_loss.item()
            running_adversarial_loss += adversarial_loss.item()
            running_vae_loss += vae_loss.item()
            running_encoder_loss += encoder_loss.item()

    return {
        "recon_loss": running_recon_loss / len(dataloader),
        "kl_loss": running_kl_loss / len(dataloader),
        "y_hat_loss": running_y_hat_loss / len(dataloader),
        "adversarial_loss": running_adversarial_loss / len(dataloader),
        "vae_loss": running_vae_loss / len(dataloader),
        "encoder_loss": running_encoder_loss / len(dataloader),
    }


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
    logger.info("Model info: %s", model)

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        config=config,
        device=device,
    )

    wandb.finish()
