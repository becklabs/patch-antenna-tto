import torch
import torch.nn as nn
import numpy as np

class StandardScaler(nn.Module):
    def __init__(self, dim=0, eps=1e-6):
        super(StandardScaler, self).__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        self.fitted = False

    def fit(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")
        x = x.detach()  # Ensure gradients are not tracked during fitting

        self.mean = x.mean(dim=self.dim, keepdim=True)
        self.std = x.std(dim=self.dim, keepdim=True, unbiased=False) + self.eps
        self.fitted = True

    def transform(self, x):
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' with training data first.")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")

        return (x - self.mean) / self.std

    def inverse_transform(self, x_scaled):
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' with training data first.")

        if isinstance(x_scaled, np.ndarray):
            x_scaled = torch.from_numpy(x_scaled)
        elif not isinstance(x_scaled, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")
        
        return x_scaled * self.std + self.mean

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
        self.register_buffer('x_min', None)
        self.register_buffer('x_max', None)
        self.fitted = False

    def fit(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")
        x = x.detach()  # Ensure gradients are not tracked during fitting
        
        
        self.x_min = x.min(dim=self.dim, keepdim=True)[0]
        self.x_max = x.max(dim=self.dim, keepdim=True)[0]
        self.fitted = True

    def transform(self, x):
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' with training data first.")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")

        return (x - self.x_min) / (self.x_max - self.x_min + self.eps)

    def inverse_transform(self, x_scaled):
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' with training data first.")
        
        if isinstance(x_scaled, np.ndarray):
            x_scaled = torch.from_numpy(x_scaled)
        elif not isinstance(x_scaled, torch.Tensor):
            raise TypeError("Input x must be a numpy array or a torch tensor.")

        return x_scaled * (self.x_max - self.x_min + self.eps) + self.x_min

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def forward(self, x):
        return self.transform(x)
