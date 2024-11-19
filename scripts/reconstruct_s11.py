import os
import yaml
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import wandb

# Load configuration from YAML file
with open("config/train/reconstruct_s11.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Weights & Biases
wandb.init(project=config["wandb"]["project"])

# Set device
device_config = config.get("device", "auto")
if device_config == "cpu":
    device = torch.device("cpu")
elif device_config == "cuda":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("CUDA is not available.")
elif device_config == "mps":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise ValueError("MPS is not available.")
elif device_config == "auto":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
else:
    raise ValueError(f"Invalid device setting: {device_config}")
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = config.get("seed", 42)
torch.manual_seed(seed)
np.random.seed(seed)

# Hyperparameters
hyperparams = config.get("hyperparameters", {})
num_epochs = hyperparams.get("num_epochs", 500)
batch_size = hyperparams.get("batch_size", 32)
learning_rate = hyperparams.get("learning_rate", 1e-3)
design_param_dim = hyperparams.get("design_param_dim", 3)
s11_length = hyperparams.get("s11_length", 1000)

import torch.nn as nn
from typing import Dict


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


loss_params = config.get("loss_parameters", {})
lambda1 = loss_params.get("lambda1", 1.0)
lambda2 = loss_params.get("lambda2", 1.0)
lambda_smooth = loss_params.get("lambda_smooth", 1.0)


def s11_reconstruction_loss(
    pred_S11, true_S11, lambda1=lambda1, lambda2=lambda2, lambda_smooth=lambda_smooth
):
    # pred_S11, true_S11: (batch_size, M)
    mse_loss_fn = nn.MSELoss()
    L_MSE = mse_loss_fn(pred_S11, true_S11)
    pred_diff1 = pred_S11[:, 1:] - pred_S11[:, :-1]  # (batch_size, M-1)
    true_diff1 = true_S11[:, 1:] - true_S11[:, :-1]
    L_first = mse_loss_fn(pred_diff1, true_diff1)
    pred_diff2 = (
        pred_S11[:, 2:] - 2 * pred_S11[:, 1:-1] + pred_S11[:, :-2]
    )  # (batch_size, M-2)
    true_diff2 = true_S11[:, 2:] - 2 * true_S11[:, 1:-1] + true_S11[:, :-2]
    L_second = mse_loss_fn(pred_diff2, true_diff2)
    L_smooth = torch.mean(pred_diff2**2)
    total_loss = (
        L_MSE + lambda1 * L_first + lambda2 * L_second + lambda_smooth * L_smooth
    )
    return total_loss


def load_data():
    data_dir = config["data"]["data_dir"]
    design_params_file = os.path.join(data_dir, config["data"]["design_params_file"])
    freq_response_file = os.path.join(data_dir, config["data"]["freq_response_file"])
    design_params = np.load(design_params_file)  # (num_samples, design_param_dim)
    freq_response = np.load(freq_response_file)  # (num_samples, N, 2)
    s11_curves = freq_response[:, :, 1]  # (num_samples, s11_length)
    return design_params, s11_curves


design_params, s11_curves = load_data()
num_samples = design_params.shape[0]
print(f"Design Parameters Shape: {design_params.shape}")
print(f"S11 Curves Shape: {s11_curves.shape}")

# Split dataset into training and validation sets
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size
indices = np.arange(num_samples)
np.random.shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_design = design_params[train_indices]
train_s11 = s11_curves[train_indices]
val_design = design_params[val_indices]
val_s11 = s11_curves[val_indices]

# Initialize scalers
design_scaler = MinMaxScaler()
s11_scaler = StandardScaler()

# Fit scalers on training data
train_design_scaled = design_scaler.fit_transform(train_design)
train_s11_scaled = s11_scaler.fit_transform(train_s11)

# Transform validation data
val_design_scaled = design_scaler.transform(val_design)
val_s11_scaled = s11_scaler.transform(val_s11)

# Save scalers for future use
scaler_dir = config["data"]["scaler_dir"]
os.makedirs(scaler_dir, exist_ok=True)
joblib.dump(
    design_scaler, os.path.join(scaler_dir, config["data"]["scalers"]["design_scaler"])
)
joblib.dump(
    s11_scaler, os.path.join(scaler_dir, config["data"]["scalers"]["s11_scaler"])
)
print("Scalers saved.")


# Create custom Dataset
class AntennaDataset(Dataset):
    def __init__(self, design_params, s11_curves):
        # design_params: (num_samples, design_param_dim)
        # s11_curves: (num_samples, s11_length)
        self.design_params = torch.tensor(design_params, dtype=torch.float32)
        self.s11_curves = torch.tensor(s11_curves, dtype=torch.float32)

    def __len__(self):
        return len(self.design_params)

    def __getitem__(self, idx):
        design_param = self.design_params[idx]
        s11_curve = self.s11_curves[idx]
        return design_param, s11_curve


# Create Dataset instances
train_dataset = AntennaDataset(train_design_scaled, train_s11_scaled)
val_dataset = AntennaDataset(val_design_scaled, val_s11_scaled)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define the Decoder Model
class S11Decoder(nn.Module):
    def __init__(self, latent_dim, output_length=1000, output_channels=1):
        super(S11Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.output_channels = output_channels
        self.fc = nn.Linear(self.latent_dim, 512 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=latent_dim,
                out_channels=512,
                kernel_size=4,
                stride=4,
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=512, out_channels=256, kernel_size=5, stride=5, padding=0
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=256, out_channels=128, kernel_size=5, stride=5, padding=0
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=10, stride=10, padding=0
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=self.output_channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, x):
        # x: (batch_size, latent_dim)
        x = x.unsqueeze(-1)  # (batch_size, latent_dim, 1)
        x = self.decoder(x)  # (batch_size, output_channels, output_length)
        x = x.squeeze(1)  # (batch_size, output_length)
        return x


latent_dim = config["model"]["latent_dim"]
output_length = config["model"]["output_length"]
output_channels = config["model"]["output_channels"]
model = S11Decoder(
    latent_dim=latent_dim, output_length=output_length, output_channels=output_channels
).to(device)
print("parameters", count_parameters(model))
print(model)

# Define the Loss Function
criterion = nn.MSELoss()

# Define the Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for design_param, s11_curve in train_loader:
        design_param = design_param.to(device)
        s11_curve = s11_curve.to(device)

        optimizer.zero_grad()
        outputs = model(design_param)
        loss = s11_reconstruction_loss(outputs, s11_curve)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * design_param.size(0)

    # Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for design_param, s11_curve in val_loader:
            design_param = design_param.to(device)
            s11_curve = s11_curve.to(device)
            outputs = model(design_param)
            loss = criterion(outputs, s11_curve)
            val_loss += loss.item() * design_param.size(0)

    # Calculate average losses
    train_loss = train_loss / train_size
    val_loss = val_loss / val_size

    # Log metrics to Weights & Biases
    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    # Print training and validation loss
    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Train Loss: {train_loss:.6f}, "
        f"Val Loss: {val_loss:.6f}"
    )

# Save the trained model
model_save_path = config["model"]["model_save_path"]
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print("Model saved.")

# Finish the wandb run
wandb.finish()


# Plot some reconstructed S11 curves
def unscale_design(design_scaled, scaler):
    return scaler.inverse_transform(design_scaled)


def unscale_s11(s11_scaled, scaler):
    return scaler.inverse_transform(s11_scaled)


# Select some examples from validation set
num_examples = config["plots"]["num_examples"]
indices = np.random.choice(len(val_dataset), num_examples, replace=False)

model.eval()
with torch.no_grad():
    for i in indices:
        design_param_scaled, s11_curve_scaled = val_dataset[i]
        design_param_scaled = design_param_scaled.unsqueeze(0).to(
            device
        )  # (1, design_param_dim)
        output_scaled = model(design_param_scaled).cpu().numpy().flatten()
        design_param = design_scaler.inverse_transform(
            design_param_scaled.cpu().numpy()
        )  # (1, design_param_dim)
        design_length, design_width, design_feed = design_param.flatten()
        s11_curve = s11_scaler.inverse_transform(
            s11_curve_scaled.unsqueeze(0).numpy()
        ).flatten()
        output_curve = s11_scaler.inverse_transform(
            output_scaled.reshape(1, -1)
        ).flatten()
        plt.figure(figsize=(12, 6))
        plt.plot(s11_curve, label="Original S11 (dB)", linewidth=2)
        plt.plot(output_curve, label="Reconstructed S11 (dB)", linestyle="--")
        plt.title(
            f"Example {i+1} - Design Parameters:\nLength={design_length:.2f} mm, "
            f"Width={design_width:.2f} mm, Feed Position={design_feed:.2f} mm"
        )
        plt.xlabel("Frequency Index")
        plt.ylabel("S11 (dB)")
        plt.legend()
        plt.grid(True)
        plt.show()
