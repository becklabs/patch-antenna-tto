import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


class RectangularPatchDataset(Dataset):
    """
    Dataset for Rectangular Patch design parameters (length, width, feed_pos) and S11 curves.
    """

    def __init__(
        self,
        design_params: np.ndarray,
        s11_curves: np.ndarray,
        design_params_scaler=None,
        s11_curves_scaler=None,
        design_device: str = "cpu",
        curves_device: str = "cpu",
    ):
        """
        Dataset for Rectangular Patch design parameters (length, width, feed_pos) and S11 curves.

        Args:
            design_params (numpy.ndarray): Design parameters (num_samples, 3).
            s11_curves (numpy.ndarray): S11 curves (num_samples, s11_length).
            design_params_scaler (StandardScaler, optional): Pre-fitted scaler for design parameters.
            s11_curves_scaler (StandardScaler, optional): Pre-fitted scaler for S11 curves.
            design_device (str, optional): Device to store design parameters. Defaults to "cpu".
            curves_device (str, optional): Device to store S11 curves. Defaults to "cpu".
        """
        if design_params_scaler is None:
            self.design_params_scaler = MinMaxScaler()
            design_params_scaled = self.design_params_scaler.fit_transform(
                design_params
            )
        else:
            self.design_params_scaler = design_params_scaler
            design_params_scaled = self.design_params_scaler.transform(design_params)

        if s11_curves_scaler is None:
            self.s11_curves_scaler = StandardScaler()
            s11_curves_scaled = self.s11_curves_scaler.fit_transform(s11_curves)
        else:
            self.s11_curves_scaler = s11_curves_scaler
            s11_curves_scaled = self.s11_curves_scaler.transform(s11_curves)

        self.design_device = design_device
        self.curves_device = curves_device

        if isinstance(design_params_scaled, np.ndarray):
            self.design_params = torch.from_numpy(design_params_scaled).to(
                torch.float32
            )
        else:
            self.design_params = design_params_scaled.to(torch.float32)

        if isinstance(s11_curves_scaled, np.ndarray):
            self.s11_curves = torch.from_numpy(s11_curves_scaled).to(torch.float32)
        else:
            self.s11_curves = s11_curves_scaled.to(torch.float32)

        self.design_params = self.design_params.to(self.design_device)
        self.s11_curves = self.s11_curves.to(self.curves_device)

    def __len__(self):
        return len(self.design_params)

    def __getitem__(self, idx):
        design_param = self.design_params[idx]
        s11_curve = self.s11_curves[idx]
        return design_param, s11_curve
