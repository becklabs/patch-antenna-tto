import torch
import torch.nn as nn
import numpy as np


class StandardScaler(nn.Module):
    def __init__(self, dim=0, eps=1e-6):
        super(StandardScaler, self).__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self.fitted = False

    @classmethod
    def from_sklearn(cls, sklearn_scaler):
        """
        Create a StandardScaler from a fitted sklearn.preprocessing.StandardScaler.

        Parameters
        ----------
        sklearn_scaler : sklearn.preprocessing.StandardScaler
            A fitted sklearn StandardScaler.
        """
        if not hasattr(sklearn_scaler, "mean_") or not hasattr(
            sklearn_scaler, "scale_"
        ):
            raise ValueError(
                "The provided sklearn scaler is not a fitted StandardScaler."
            )

        obj = cls()

        mean_t = torch.tensor(sklearn_scaler.mean_, dtype=torch.float32).unsqueeze(0)
        std_t = torch.tensor(sklearn_scaler.scale_, dtype=torch.float32).unsqueeze(0)

        obj.register_buffer("mean", mean_t)
        obj.register_buffer("std", std_t)
        obj.fitted = True
        return obj

    def fit(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")
        x = x.detach()  # Ensure gradients are not tracked during fitting

        self.mean = x.mean(dim=self.dim, keepdim=True)
        self.std = x.std(dim=self.dim, keepdim=True, unbiased=False) + self.eps
        self.fitted = True

    def transform(self, x):
        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted yet. Call 'fit' with training data first."
            )

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")

        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x_scaled):
        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted yet. Call 'fit' with training data first."
            )

        if isinstance(x_scaled, np.ndarray):
            x_scaled = torch.from_numpy(x_scaled.astype(np.float32))
        elif not isinstance(x_scaled, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")

        return x_scaled * self.std.to(x_scaled.device) + self.mean.to(x_scaled.device)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def forward(self, x):
        return self.transform(x)


class MinMaxScaler(nn.Module):
    def __init__(self, dim=0, eps=1e-6):
        super(MinMaxScaler, self).__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("x_min", None)
        self.register_buffer("x_max", None)
        self.fitted = False

    @classmethod
    def from_sklearn(cls, sklearn_scaler):
        """
        Create a MinMaxScaler from a fitted sklearn.preprocessing.MinMaxScaler.

        NOTE: Only works for MinMaxScaler with feature_range=(0, 1)

        Parameters
        ----------
        sklearn_scaler : sklearn.preprocessing.MinMaxScaler
            A fitted sklearn MinMaxScaler.
        """
        if not hasattr(sklearn_scaler, "data_min_") or not hasattr(
            sklearn_scaler, "data_max_"
        ):
            raise ValueError(
                "The provided sklearn scaler is not a fitted MinMaxScaler."
            )

        obj = cls()

        min_t = torch.tensor(sklearn_scaler.data_min_, dtype=torch.float32).unsqueeze(0)
        max_t = torch.tensor(sklearn_scaler.data_max_, dtype=torch.float32).unsqueeze(0)

        obj.register_buffer("x_min", min_t)
        obj.register_buffer("x_max", max_t)
        obj.fitted = True
        return obj

    def fit(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")
        x = x.detach()  # Ensure gradients are not tracked during fitting

        self.x_min = x.min(dim=self.dim, keepdim=True)[0]
        self.x_max = x.max(dim=self.dim, keepdim=True)[0]
        self.fitted = True

    def transform(self, x):
        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted yet. Call 'fit' with training data first."
            )

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")

        return (x - self.x_min.to(x.device)) / (
            self.x_max.to(x.device) - self.x_min.to(x.device) + self.eps
        )

    def inverse_transform(self, x_scaled):
        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted yet. Call 'fit' with training data first."
            )

        if isinstance(x_scaled, np.ndarray):
            x_scaled = torch.from_numpy(x_scaled.astype(np.float32))
        elif not isinstance(x_scaled, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")

        return x_scaled * (
            self.x_max.to(x_scaled.device) - self.x_min.to(x_scaled.device) + self.eps
        ) + self.x_min.to(x_scaled.device)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def forward(self, x):
        return self.transform(x)
