import yaml
import torch
import numpy as np
import os
import wandb
from typing import Dict
import torch.nn as nn
import logging
from deepad.nn.datasets import RectangularPatchDataset

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


def save_checkpoint(model, optimizer, epoch, X_scaler, y_scaler, config):
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    X_scaler = checkpoint.get("X_scaler", None)
    y_scaler = checkpoint.get("y_scaler", None)
    start_epoch = checkpoint["epoch"]
    logger.info("Resumed training from epoch %d", start_epoch)
    return model, optimizer, X_scaler, y_scaler, start_epoch


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


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
