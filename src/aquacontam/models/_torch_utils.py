"""PyTorch utility classes for tabular deep learning models."""

from __future__ import annotations

import copy
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def get_device(gpu_id: int | None = None) -> torch.device:
    """Select the best available compute device.

    Resolution order:
      1. ``gpu_id`` argument when CUDA is available → ``cuda:gpu_id``.
      2. ``AQUACONTAM_GPU_ID`` env var when CUDA is available.
      3. CUDA available → ``cuda:0`` (honors ``CUDA_VISIBLE_DEVICES``).
      4. MPS (Apple Silicon).
      5. CPU.

    Note: on multi-GPU Windows setups, callers should also set
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so torch's index numbering matches
    ``nvidia-smi -L``. See ``scripts/reproduce.py`` for the wiring.
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")
        env = os.environ.get("AQUACONTAM_GPU_ID")
        if env is not None:
            return torch.device(f"cuda:{int(env)}")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement.
    min_delta : float
        Minimum decrease in loss to qualify as improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False
        self._best_state_dict: dict[str, torch.Tensor] | None = None

    def step(self, val_loss: float, model: nn.Module | None = None) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Current validation loss.
        model : nn.Module or None
            If provided, saves a deep copy of the model state dict
            when validation loss improves.

        Returns
        -------
        bool
            True if training should stop.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self._best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model: nn.Module) -> bool:
        """Restore the best model weights.

        Parameters
        ----------
        model : nn.Module
            Model to restore weights into.

        Returns
        -------
        bool
            True if weights were restored, False if no best state was saved.
        """
        if self._best_state_dict is not None:
            model.load_state_dict(self._best_state_dict)
            return True
        return False


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch Tensor to a numpy array.

    Works around PyTorch/NumPy version incompatibility where
    ``Tensor.numpy()`` may fail with newer NumPy versions.

    Parameters
    ----------
    t : torch.Tensor
        Tensor to convert (must be on CPU).

    Returns
    -------
    np.ndarray
    """
    try:
        return t.detach().cpu().numpy()  # type: ignore[no-any-return]
    except RuntimeError:
        return np.array(t.detach().cpu().tolist())  # type: ignore[no-any-return]


def to_numpy(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert feature input to float64 numpy array.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Float64 array.
    """
    if isinstance(X, pd.DataFrame):
        return X.to_numpy().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def to_array(y: pd.Series | np.ndarray) -> np.ndarray:
    """Convert labels to 1D numpy array.

    Parameters
    ----------
    y : pd.Series | np.ndarray
        Label vector.

    Returns
    -------
    np.ndarray
        1D array.
    """
    if isinstance(y, pd.Series):
        return y.to_numpy()
    return np.asarray(y)


def compute_validation_loss(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute average validation loss over a DataLoader.

    Parameters
    ----------
    model : nn.Module
        PyTorch model (set to eval mode internally).
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device for computation.

    Returns
    -------
    float
        Mean validation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            out = model(X_b).squeeze(-1)
            total_loss += criterion(out, y_b).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular (feature matrix + label) data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    y : np.ndarray
        Labels, shape ``(n_samples,)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
