from typing import Tuple

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        """
        Variational Autoencoder that uses provided encoder and decoder networks.

        Args:
            encoder: Neural network that outputs 2*latent_dim features (mu and logvar)
            decoder: Neural network that takes latent_dim features as input
            latent_dim: Dimension of the latent space
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to get mean and log variance of the latent distribution.
        """
        h = self.encoder(x)  # Encoder outputs concatenated mu and logvar
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Perform the reparameterization trick to enable backpropagation through sampling.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Returns:
            Tuple of (reconstruction, mean, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from the prior distribution.
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


class CVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,

        latent_dim: int,

        x_head: nn.Module = nn.Identity(),
        condition_head: nn.Module = nn.Identity(),
    ):
        """
        Conditional Variational Autoencoder that uses provided encoder and decoder networks.

        Args:
            encoder: Neural network that outputs 2*latent_dim features (mu and logvar)
            decoder: Neural network that takes latent_dim + condition_dim features as input
            latent_dim: Dimension of the latent space
            x_head: Head for processing x
            condition_head: Head for processing conditional information
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.x_head = x_head
        self.condition_head = condition_head

    def encode(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to get mean and log variance of the latent distribution.

        Args:
            x: Input tensor
            c: Condition tensor
        """
        # Concatenate input and condition along feature dimension
        encoder_input = torch.cat([self.x_head(x), self.condition_head(c)], dim=1)
        h = self.encoder(encoder_input)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Perform the reparameterization trick to enable backpropagation through sampling.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector and condition to reconstruction.

        Args:
            z: Latent vector
            c: Condition tensor
        """
        # Concatenate latent vector and condition
        decoder_input = torch.cat([z, self.condition_head(c)], dim=1)
        return self.decoder(decoder_input)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CVAE.

        Args:
            x: Input tensor
            c: Condition tensor

        Returns:
            Tuple of (reconstruction, mean, logvar)
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(
        self, num_samples: int, condition: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Generate samples from the prior distribution conditioned on given conditions.

        Args:
            num_samples: Number of samples to generate
            condition: Condition tensor to generate samples for
            device: Device to generate samples on
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        if condition.size(0) == 1: # If condition is singular, repeat it for all samples
            condition = condition.repeat(num_samples, 1)
        return self.decode(z, condition)


class AdversarialVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        discriminator: nn.Module,

        latent_dim: int,

        x_head: nn.Module = nn.Identity(),
        condition_head: nn.Module = nn.Identity(),
    ):
        """
        Adversarial Variational Autoencoder that uses provided encoder, decoder, and discriminator networks.
        https://archives.ismir.net/ismir2020/paper/000099.pdf

        Args:
            encoder: Neural network that outputs 2*latent_dim features (mu and logvar)
            decoder: Neural network that takes latent_dim + condition_dim features as input
            discriminator: Neural network that takes latent_dim features as input and produces y
            latent_dim: Dimension of the latent space
            x_head: Head for processing x
            condition_head: Head for processing conditional information
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        self.latent_dim = latent_dim

        self.x_head = x_head
        self.condition_head = condition_head
    
    def encoder_params(self):
        return self.encoder.parameters()
    
    def decoder_params(self):
        return self.decoder.parameters()
    
    def discriminator_params(self):
        return self.discriminator.parameters()

    def encode(
        self, x: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to get mean and log variance of the latent distribution.

        Args:
            x: Input tensor
        """
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector and condition to reconstruction.

        Args:
            z: Latent vector
            c: Condition tensor
        """
        # Concatenate latent vector and condition
        decoder_input = torch.cat([z, self.condition_head(c)], dim=1)
        x = self.decoder(decoder_input)
        if x.dim() == 3:
            x = x.squeeze(2)
        return x
    
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict y-hat with the discriminator for a given latent vector.
        """
        y = self.discriminator(z)
        if y.dim() == 3:
            y = y.squeeze(2)
        return y

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CVAE.

        Args:
            x: Input tensor
            c: Condition tensor

        Returns:
            Tuple of (reconstruction, mean, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.predict(z)
        recon_x = self.decode(z, c)
        return recon_x, y, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Perform the reparameterization trick to enable backpropagation through sampling.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    @torch.no_grad()
    def sample(
        self, num_samples: int, condition: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Generate samples from the prior distribution conditioned on given conditions.

        Args:
            num_samples: Number of samples to generate
            condition: Condition tensor to generate samples for
            device: Device to generate samples on
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        if condition.size(0) == 1: # If condition is singular, repeat it for all samples
            condition = condition.repeat(num_samples, 1)
        return self.decode(z, condition)
