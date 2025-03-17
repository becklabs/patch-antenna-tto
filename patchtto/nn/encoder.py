from pytorch_tcn import TCN
from typing import List, Optional
import torch.nn as nn


class TCNEncoder(nn.Module):
    def __init__(
        self, latent_dim: int = 32, depth: int = 8, channels: Optional[List[int]] = None
    ):
        super(TCNEncoder, self).__init__()

        if channels is None:
            channels = [latent_dim] * depth

        self.encoder = nn.Sequential(
            TCN(
                num_inputs=1,  # Since S11 is a single value per frequency point
                num_channels=channels,  # 8 residual blocks with latent_dim channels each
                kernel_size=4,  
                dilations=None,  
                dilation_reset=None,  # No reset, to cover the entire sequence
                dropout=0.1,  
                causal=False,  # Not a causal problem; we consider the full sequence
                use_norm="weight_norm",  
                activation="relu",  # Standard activation function
                kernel_initializer="xavier_uniform",  # Good default initializer
                use_skip_connections=True, 
                input_shape="NCL",  # Input shape is (batch_size, channels, length)
                embedding_shapes=None,  
                use_gate=False,  
                output_projection=None,  
                output_activation=None,  
            ),
            nn.AdaptiveAvgPool1d(
                1
            ),  # Pool over the sequence length to get a fixed-size vector
            nn.Flatten(),  # Flatten the output to (batch_size, latent_dim)
        )

    def forward(self, x):
        # x is of shape (batch_size, 1000, 1)
        if x.dim() == 2:
            x = x.unsqueeze(2)
        x = x.transpose(1, 2)  # Shape: (batch_size, 1, 1000)
        x = self.encoder(x)
        return x  # Output shape: (batch_size, latent_dim)

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),  # Output: (batch_size, 32, 500)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # Output: (batch_size, 64, 250)
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # Output: (batch_size, 128, 125)
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),  # Output: (batch_size, 256, 63)
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),  # Output: (batch_size, 512, 32)
            nn.ReLU(),
            nn.Flatten(),  # Output: (batch_size, 512 * 32)
            nn.Linear(512 * 32, latent_dim * 2)  # Output: (batch_size, latent_dim * 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        return x

class FeedForwardEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(FeedForwardEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim * 2),
        )

    def forward(self, x):
        return self.encoder(x)

class SimpleFeedForwardEncoder(nn.Module):
    def __init__(self, x_dim: int, latent_dim: int):
        super(SimpleFeedForwardEncoder, self).__init__()
        self.encoder = nn.Sequential(   
            nn.Linear(x_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)
