import numpy as np
import torch
import torch.nn as nn

from ..nn.losses import masked_loss, gaussian_nll
from ..nn.vae import VAE, AdversarialVAE
from ..nn.preprocessing import StandardScaler, MinMaxScaler
from ..simulation.harness import SimulationHarness
from .utils import target_curve_mask


class S11SearchCriterion(nn.Module):
    """
    Criterion for searching through the distribution of S11 curves.
    """

    MASK_DB_THRESHOLD = 0.1  # dB

    def __init__(
        self,
        vae: VAE,
        target_curve: torch.Tensor,
        curve_scaler: object,
        lambda_reg: float = 1.0,
        device: str = "cpu",
    ):
        super(S11SearchCriterion, self).__init__()

        self.lambda_reg = lambda_reg
        self.vae = vae
        self.curve_scaler = curve_scaler

        self.mask = target_curve_mask(
            target_curve=target_curve, threshold=self.MASK_DB_THRESHOLD
        )

        target_curve_scaled = self.curve_scaler.transform(
            target_curve.cpu().numpy().reshape(1, -1)
        )
        self.target_curve = (
            torch.FloatTensor(target_curve_scaled.astype(np.float32))
            .squeeze()
            .to(device)
        )

        self.recon_criterion = nn.MSELoss(
            reduction="sum"
        )  # Masked loss averages over mask

    def forward(self, z: torch.Tensor):
        curve = self.vae.decode(z)
        loss = masked_loss(
            pred=curve.squeeze(),
            target=self.target_curve.squeeze(),
            mask=self.mask,
            loss_fn=self.recon_criterion,
        )
        reg_loss = torch.sum(z**2)
        return loss + self.lambda_reg * reg_loss


class DesignSearchCriteria(nn.Module):
    def __init__(
        self, cvae: AdversarialVAE, condition: torch.Tensor, design_scaler: object, lambda_reg: float = 1.0
    ):
        """
        Manufacturability criterion for the rectangular patch design space.
        """
        super(DesignSearchCriteria, self).__init__()

        if not isinstance(design_scaler, nn.Module):
            self.design_scaler = MinMaxScaler.from_sklearn(design_scaler)
        else:
            self.design_scaler = design_scaler

        self.cvae = cvae
        self.condition = condition
        self.lambda_reg = lambda_reg

    def forward(self, z: torch.Tensor):
        """
        Compute the penalty for the batch of normalized designs.
        """
        x_scaled = self.cvae.decode(z, self.condition)

        x = self.design_scaler.inverse_transform(x_scaled)

        length = x[:, 0]
        width = x[:, 1]
        feed_pos = x[:, 2]

        # Condition 1: length < 0
        length_penalty = torch.relu(-length) ** 2

        # Condition 2: width < 0
        width_penalty = torch.relu(-width) ** 2

        # Condition 3: feed_position outside (-length/2, 0)
        lower_bound = -length / 2.0
        lower_penalty = torch.relu(lower_bound - feed_pos) ** 2
        upper_penalty = torch.relu(feed_pos) ** 2
        feed_penalty = lower_penalty + upper_penalty

        total_penalty = length_penalty + width_penalty + feed_penalty

        reg_loss = torch.sum(z**2)
        return total_penalty + self.lambda_reg * reg_loss


class OracleDesignScorer:
    """
    Scores the design according to a target curve using an EM solver.
    """

    def __init__(self, target_curve: np.ndarray, sim_harness: SimulationHarness):

        self.target_curve = torch.from_numpy(target_curve)
        self.harness = sim_harness

        self.mask = target_curve_mask(target_curve=target_curve)

        print("Mask size:", self.mask.size())

    def __call__(self, x: np.ndarray):
        """
        Score a single design with the surrogate.
        """
        if len(x.shape) == 1:  # Add a batch dimension if not already present
            x = x[np.newaxis, ...]

        s11 = torch.from_numpy(self.harness.simulate(x))

        loss = masked_loss(
            pred=s11.squeeze(),
            target=self.target_curve.squeeze(),
            mask=self.mask,
        )
        return loss, s11


class SurogateDesignScorer:
    """
    Scores the design according to a target curve using a differentiable surrogate model.

    TODO:
    - Implement version of this that handles normalized designs.
    """

    def __init__(
        self,
        target_curve: np.ndarray,
        surrogate: nn.Module,
        design_scaler: object,
        curve_scaler: object,
        device: str = "cpu",
    ):

        self.mask = target_curve_mask(target_curve=target_curve).to(device)

        target_curve_scaled = curve_scaler.transform(target_curve.reshape(1, -1))
        self.target_curve = target_curve_scaled.squeeze().to(device)
        # self.target_curve = (
        #     torch.from_numpy(target_curve_scaled.astype(np.float32))
        #     .squeeze()
        #     .to(device)
        # )

        self.curve_scaler = curve_scaler
        self.design_scaler = design_scaler

        self.surrogate = surrogate

        self.device = device

    def __call__(self, x: np.ndarray):
        """
        Score a single design with the surrogate.
        """
        if len(x.shape) == 1:  # Add a batch dimension if not already present
            x = x[np.newaxis, ...]

        x_scaled = self.design_scaler.transform(x).to(self.device)
        # x_scaled = torch.from_numpy(
        #     self.design_scaler.transform(x).astype(np.float32)
        # ).to(self.device)

        mean, variance = self.surrogate(x_scaled)
        mean = mean.squeeze()
        variance = variance.squeeze()

        nll = gaussian_nll(y_pred=self.target_curve, mean=mean, logvar=variance)
        # score = torch.sum(nll * self.mask)

        target_masked = self.target_curve * self.mask
        mean_masked = mean * self.mask

        score = torch.sum((mean_masked - target_masked) ** 2)

        return score.item()
