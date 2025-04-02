import yaml
import torch
import numpy as np
import os
import wandb
from typing import Dict, Tuple
import torch.nn as nn
import logging
from patchtto.nn.datasets import RectangularPatchDataset
from torch.utils.data import DataLoader
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_device(device_preference):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_preference)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    X_scaler: object,
    y_scaler: object,
    config: dict,
) -> None:
    """
    Save model checkpoint to disk.

    Args:
        model (nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        epoch (int): Current epoch number.
        X_scaler (object): Scaler for design parameters.
        y_scaler (object): Scaler for S11 curves.
        config (dict): Configuration dictionary containing checkpoint settings.
    """
    checkpoint_dir = config["checkpoint"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
        },
        checkpoint_path,
    )
    wandb.save(checkpoint_path)
    logger.info("Saved checkpoint for epoch %d", epoch)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load a checkpoint from a file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state dictionary into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state dictionary into.
        device (torch.device): The device to load the model and optimizer onto.

    Returns:
        model (torch.nn.Module): The loaded model.
        optimizer (torch.optim.Optimizer): The loaded optimizer.
        X_scaler (sklearn.preprocessing.StandardScaler): The loaded design parameters scaler.
        y_scaler (sklearn.preprocessing.StandardScaler): The loaded S11 curves scaler.
        start_epoch (int): The epoch number at which to resume training.
    """
    if not os.path.exists(checkpoint_path):
        logger.info(
            "Checkpoint not found at %s. Starting from scratch.", checkpoint_path
        )
        return model, optimizer, None, None, 0
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    X_scaler = checkpoint.get("X_scaler", None)
    y_scaler = checkpoint.get("y_scaler", None)
    start_epoch = checkpoint["epoch"]
    logger.info("Resumed training from epoch %d", start_epoch)
    return model, optimizer, X_scaler, y_scaler, start_epoch


def load_wandb_checkpoint(
    run_path: str,
    checkpoint_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    download_dir: str = "wandb_downloads",
):
    """
    Load a checkpoint from Weights & Biases.

    Args:
        run_path (str): Path to the W&B run (format: "entity/project/run_id")
        checkpoint_name (str): Name of the checkpoint file
        model (torch.nn.Module): The model to load the state dictionary into
        optimizer (torch.optim.Optimizer): The optimizer to load the state dictionary into
        device (torch.device): The device to load the model and optimizer onto
        download_dir (str): Directory to download the checkpoint to

    Returns:
        Same as load_checkpoint(): (model, optimizer, X_scaler, y_scaler, start_epoch)
    """
    api = wandb.Api()
    try:
        os.makedirs(download_dir, exist_ok=True)

        run = api.run(run_path)
        checkpoint_path = Path(download_dir) / checkpoint_name

        run.file(checkpoint_name).download(root=download_dir, replace=True)

        result = load_checkpoint(checkpoint_path, model, optimizer, device)

        checkpoint_path.unlink()

        return result

    except Exception as e:
        logger.error(f"Failed to load W&B checkpoint: {e}")
        return model, optimizer, None, None, 0


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def load_data(config):
    design_params = np.load(
        os.path.join(config["data"]["data_dir"], "design_params.npy")
    )
    freq_response = np.load(
        os.path.join(config["data"]["data_dir"], "freq_response.npy")
    )

    if config["data"]["resonance_threshold"] is not None:
        resonance_mask = np.any(freq_response[:, :, 1] < float(config["data"]["resonance_threshold"]), axis=1)
        design_params = design_params[resonance_mask]
        freq_response = freq_response[resonance_mask]

    if config["data"]["feed_threshold"] is not None:
        feed_mask = np.abs(design_params[:, 2]) > float(config["data"]["feed_threshold"])
        design_params = design_params[feed_mask]
        freq_response = freq_response[feed_mask]
    return design_params, freq_response


def prepare_datasets(
    design_params: np.ndarray,
    freq_response: np.ndarray,
    config: dict,
    curves_device: str,
    design_device: str,
    s11_curves_scaler: object = None,
    design_params_scaler: object = None,
) -> Tuple[RectangularPatchDataset, RectangularPatchDataset]:
    """
    Prepare training and testing datasets from input data.

    Args:
        design_params (np.ndarray): Array of design parameters.
        freq_response (np.ndarray): Array of frequency responses.
        config (dict): Configuration dictionary.
        curves_device (str): Device for S11 curves.
        design_device (str): Device for design parameters.
        s11_curves_scaler (object, optional): Scaler for S11 curves. Defaults to None.
        design_params_scaler (object, optional): Scaler for design parameters. Defaults to None.

    Returns:
        tuple[RectangularPatchDataset, RectangularPatchDataset]: Training and testing datasets.
    """
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

    if s11_curves_scaler is not None:
        s11_curves_scaler.fit(s11_curves[train_inds])

    if design_params_scaler is not None:
        design_params_scaler.fit(design_params[train_inds])

    train_dataset = RectangularPatchDataset(
        design_params=design_params[train_inds],
        s11_curves=s11_curves[train_inds],
        design_params_scaler=design_params_scaler,
        s11_curves_scaler=s11_curves_scaler,
        curves_device=curves_device,
        design_device=design_device,
    )
    test_dataset = RectangularPatchDataset(
        design_params=design_params[test_inds],
        s11_curves=s11_curves[test_inds],
        design_params_scaler=train_dataset.design_params_scaler,
        s11_curves_scaler=train_dataset.s11_curves_scaler,
        curves_device=curves_device,
        design_device=design_device,
    )

    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset: RectangularPatchDataset,
    test_dataset: RectangularPatchDataset,
    config: dict,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader objects for training and testing.

    Args:
        train_dataset (RectangularPatchDataset): Training dataset.
        test_dataset (RectangularPatchDataset): Testing dataset.
        config (dict): Configuration dictionary containing hyperparameters.

    Returns:
        tuple[DataLoader, DataLoader]: Training and testing dataloaders.
    """
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["hyperparameters"]["batch_size"]
    )
    return train_dataloader, test_dataloader


def sigmoid_annealing(
    epoch: int, n_warmup_epochs: int, min_weight: float, max_weight: float
) -> float:
    """Compute sigmoid-based annealing weight for loss scheduling.

    Args:
        epoch (int): Current training epoch
        n_warmup_epochs (int): Number of epochs for warmup period
        min_weight (float): Minimum weight value
        max_weight (float): Maximum weight value

    Returns:
        float: Annealed weight value between min_weight and max_weight
    """
    if n_warmup_epochs == 0:  # No scheduling case
        return max_weight
    x = 10 * (epoch - 0.5 * n_warmup_epochs) / n_warmup_epochs
    sigmoid = 1 / (1 + np.exp(-x))
    return min_weight + (max_weight - min_weight) * sigmoid
