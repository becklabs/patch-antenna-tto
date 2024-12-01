import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List
from .decoder import ConvDecoder
from .encoder import TCNEncoder

from typing import Tuple

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
        h = self.encoder(x) # Encoder outputs concatenated mu and logvar
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Returns:
            Tuple of (reconstruction, mean, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(self, recon: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor,
                     kld_weight: float = 1.0) -> torch.Tensor:
        """
        Compute VAE loss: reconstruction loss + KL divergence.
        
        Args:
            recon: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            kld_weight: Weight for the KL divergence term
            
        Returns:
            Total loss (reconstruction + weighted KL divergence)
        """
        recon_loss = F.mse_loss(recon, x, reduction='sum') # Reconstruction loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence between q(z|x) and p(z)
        
        return recon_loss + kld_weight * kld_loss

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from the prior distribution.
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)