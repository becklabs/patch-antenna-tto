from abc import ABC, abstractmethod

import numpy as np
import torch

from ..nn.datasets import RectangularPatchDataset
from ..nn.vae import VAE
from .utils import sort_latents


class InitializationStrategy(ABC):
    """
    Base class defining interface for initializing latent vectors in VAE search.
    """

    @abstractmethod
    def __call__(self, latent_dim: int, n_curves: int) -> torch.Tensor: ...


class RandomInitialization(InitializationStrategy):
    """
    Initialize latent vectors by sampling from a standard normal distribution.
    """

    def __call__(self, latent_dim: int, n_curves: int) -> torch.Tensor:
        return torch.randn((n_curves, latent_dim))


class FixedInitialization(InitializationStrategy):
    """
    Initialize all latent vectors by repeating a single pre-defined latent vector.
    """

    def __init__(self, latent: torch.Tensor):
        assert latent.ndim == 2
        assert latent.size(0) == 1
        self.latent = latent

    def __call__(self, latent_dim: int, n_curves: int) -> torch.Tensor:
        assert latent_dim == self.latent.size(1)
        return self.latent.repeat(n_curves, 1)


class FixedRandomInitialization(InitializationStrategy):
    """
    Initialize latent vectors by repeating a random latent vector sampled from a standard normal distribution.
    """

    def __call__(self, latent_dim: int, n_curves: int) -> torch.Tensor:
        latent = RandomInitialization()(latent_dim, 1)
        return FixedInitialization(latent)(latent_dim, n_curves)


class ClosestCurveInitialization(InitializationStrategy):
    """
    Initialize latent vectors using the encoding of the training curve closest to a target.
    """

    def __init__(
        self,
        vae: VAE,
        target_curve: np.ndarray,
        dataset: RectangularPatchDataset,
        batch_size: int = 32,
    ):
        self.vae = vae
        self.target_curve = target_curve
        self.dataset = dataset
        self.batch_size = batch_size

    def __call__(self, latent_dim: int, n_curves: int) -> torch.Tensor:
        sorted_latents = sort_latents(
            self.vae, self.target_curve, self.dataset, self.batch_size
        )
        closest_latent = sorted_latents[0]
        return closest_latent.unsqueeze(0).repeat(n_curves, 1)


class KClosestCurveInitialization(ClosestCurveInitialization):
    """
    Initialize latent vectors using the encodings of the n_curves closest curves to a target.
    """

    def __call__(self, latent_dim: int, n_curves: int) -> torch.Tensor:
        sorted_latents = sort_latents(
            self.vae, self.target_curve, self.dataset, self.batch_size
        )
        selected_latents = sorted_latents[:n_curves]  # Take the top n_curves
        return selected_latents
