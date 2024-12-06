import numpy as np
import torch
import torch.nn as nn

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU)
    Smoother alternative that approximates ReLU with a smooth curve
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_length=1000, output_channels=1, transpose=True):
        super(ConvDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.output_channels = output_channels
        self.fc = nn.Linear(self.latent_dim, 64 * 4)
        self.transpose = transpose

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=512,
                kernel_size=5,
                stride=5,
                padding=0,
            ), # (batch_size, 512, 20)
            # nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.ConvTranspose1d(
                in_channels=512, out_channels=256, kernel_size=5, stride=5, padding=0
            ), # (batch_size, 256, 100)
            nn.Dropout(0.1),
            nn.BatchNorm1d(256),

            # nn.LeakyReLU(),
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels=256, out_channels=128, kernel_size=5, stride=5, padding=0
            ), # (batch_size, 128, 500)
            nn.Dropout(0.1),
            nn.BatchNorm1d(128),

            # nn.LeakyReLU(),
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0
            ), # (batch_size, 64, 1000)
            nn.Dropout(0.1),
            nn.BatchNorm1d(64),

            # nn.LeakyReLU(),
            nn.GELU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=self.output_channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, x):
        # x: (batch_size, latent_dim)

        x = self.fc(x)  # (batch_size, 64 * 4)
        x = x.view(-1, 64, 4)  # (batch_size, 64, 4)
        x = self.decoder(x)  # (batch_size, output_channels, output_length)

        if self.transpose:
            x = x.transpose(1, 2) # (batch_size, output_length, output_channels)
        return x

class ConvDecoderSimple(nn.Module):
    def __init__(self, latent_dim, output_length=1000, output_channels=1):
        super(ConvDecoderSimple, self).__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.output_channels = output_channels
        self.fc = nn.Linear(self.latent_dim, 64 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=512,
                kernel_size=5,
                stride=5,
                padding=0,
            ), # (batch_size, 512, 20)
            # nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.ConvTranspose1d(
                in_channels=512, out_channels=256, kernel_size=5, stride=5, padding=0
            ), # (batch_size, 256, 100)
            nn.Dropout(0.1),
            nn.BatchNorm1d(256),

            # nn.LeakyReLU(),
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels=256, out_channels=128, kernel_size=5, stride=5, padding=0
            ), # (batch_size, 128, 500)
            nn.Dropout(0.1),
            nn.BatchNorm1d(128),

            # nn.LeakyReLU(),
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0
            ), # (batch_size, 64, 1000)
            nn.Dropout(0.1),
            nn.BatchNorm1d(64),

            # nn.LeakyReLU(),
            nn.GELU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=self.output_channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, x):
        # x: (batch_size, latent_dim)

        x = self.fc(x)  # (batch_size, 64 * 4)
        x = x.view(-1, 64, 4)  # (batch_size, 64, 4)
        x = self.decoder(x)  # (batch_size, output_channels, output_length)

        x = x.transpose(1, 2) # (batch_size, output_length, output_channels)
        return x

class ConvDecoder2(nn.Module):
    def __init__(self, latent_dim, output_length=1000, output_channels=1):
        super(ConvDecoder2, self).__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.output_channels = output_channels

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=latent_dim,
                out_channels=512,
                kernel_size=4,
                stride=4,
                padding=0,
            ), # (batch_size, 512, 4)
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=512, out_channels=256, kernel_size=5, stride=5, padding=0
            ), # (batch_size, 256, 20)
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=256, out_channels=128, kernel_size=5, stride=5, padding=0
            ), # (batch_size, 128, 100)
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=10, stride=10, padding=0
            ), # (batch_size, 64, 1000)
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
        x = x.transpose(1, 2) # (batch_size, output_length, output_channels)
        return x


class FeedForwardDecoder(nn.Module):
    def __init__(self, latent_dim, output_length=1000, output_channels=1):
        super(FeedForwardDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.output_channels = output_channels

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, output_length),
        )


    def forward(self, x):
        # x: (batch_size, latent_dim)
        x = self.decoder(x) # (batch_size, output_length)
        x = x.unsqueeze(-1) # (batch_size, output_length, 1)
        return x

class SimpleFeedForwardDecoder(nn.Module):
    def __init__(self, latent_dim, output_length=1000, output_channels=1):
        super(SimpleFeedForwardDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.output_channels = output_channels

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, output_length),
        )

    def forward(self, x):
        # x: (batch_size, latent_dim)
        x = self.decoder(x) # (batch_size, output_length)
        x = x.unsqueeze(-1) # (batch_size, output_length, 1)
        return x
