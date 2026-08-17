"""Deep Tobit MLP: censoring-aware neural network for environmental monitoring data.

Replaces standard BCE/MSE loss with a Tobit likelihood that properly models
left-censored observations (non-detects). Outputs (mu, sigma) per sample:
- Detected observations contribute normal log-PDF: -log N(y | mu, sigma)
- Censored observations contribute normal log-CDF: -log Phi((DL - mu) / sigma)

A single trained model simultaneously supports classification
(P(detected) = 1 - Phi((DL - mu) / sigma)) and regression (E[Y | Y > DL]).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from aquacontam.models._torch_utils import (
    EarlyStopping,
    get_device,
    tensor_to_numpy,
    to_array,
    to_numpy,
)
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class _TobitDataset(Dataset):
    """Dataset that carries features, targets, censoring flags, and detection limits."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        censored: np.ndarray,
        detection_limits: np.ndarray,
    ) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.censored = torch.as_tensor(censored, dtype=torch.float32)
        self.detection_limits = torch.as_tensor(detection_limits, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.censored[idx], self.detection_limits[idx]


class _DeepTobitNetwork(nn.Module):
    """MLP with dual heads for mu and log_sigma."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_size, 1)
        self.log_sigma_head = nn.Linear(in_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1).clamp(-5.0, 5.0)
        return mu, log_sigma


def tobit_nll(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    y: torch.Tensor,
    censored: torch.Tensor,
    detection_limits: torch.Tensor,
) -> torch.Tensor:
    """Compute Tobit negative log-likelihood.

    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean, shape (N,).
    log_sigma : torch.Tensor
        Predicted log standard deviation, shape (N,).
    y : torch.Tensor
        Observed values, shape (N,).
    censored : torch.Tensor
        Binary censoring indicator (1 = censored), shape (N,).
    detection_limits : torch.Tensor
        Detection limits, shape (N,).

    Returns
    -------
    torch.Tensor
        Scalar mean negative log-likelihood.
    """
    sigma = torch.exp(log_sigma).clamp(min=1e-6)

    # Detected: normal log-PDF at observed value
    z_obs = (y - mu) / sigma
    log_pdf = -0.5 * z_obs**2 - log_sigma - 0.5 * np.log(2 * np.pi)

    # Censored: normal log-CDF at detection limit
    z_dl = (detection_limits - mu) / sigma
    # Use log_ndtr for numerical stability (available in torch via erfc)
    log_cdf = torch.log(0.5 * torch.erfc(-z_dl / np.sqrt(2)) + 1e-10)

    # Combine: detected samples use log_pdf, censored use log_cdf
    nll = -(1.0 - censored) * log_pdf - censored * log_cdf
    result: torch.Tensor = nll.mean()
    return result


def _prepare_censoring_arrays(
    y: np.ndarray,
    n_samples: int,
    censored: np.ndarray | None,
    detection_limits: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and prepare censoring arrays, using defaults if not provided."""
    if censored is not None:
        cens = np.asarray(censored, dtype=np.float64).ravel()
    else:
        # Default: no censoring (all observed)
        cens = np.zeros(n_samples, dtype=np.float64)

    if detection_limits is not None:
        dl = np.asarray(detection_limits, dtype=np.float64).ravel()
    else:
        # Default: use y values as detection limits for censored observations
        dl = np.where(cens > 0.5, y, 0.0).astype(np.float64)

    return cens, dl


class DeepTobitClassifier(BaseModel):
    """Censoring-aware MLP classifier using Tobit likelihood.

    Predicts P(detected) = 1 - Phi((DL - mu) / sigma) where mu, sigma
    are learned by the network. This properly accounts for left-censored
    (non-detect) observations common in environmental monitoring data.

    Parameters
    ----------
    config : dict, optional
        Hyperparameters:
        - ``hidden_sizes``: list of hidden layer sizes (default ``[128, 64, 32]``)
        - ``dropout``: dropout rate (default ``0.3``)
        - ``learning_rate``: Adam LR (default ``0.001``)
        - ``epochs``: max epochs (default ``200``)
        - ``batch_size``: mini-batch size (default ``256``)
        - ``patience``: early stopping patience (default ``20``)
        - ``random_state``: seed (default ``42``)
        - ``default_detection_limit``: DL used at predict time (default ``0.004``)
        - ``log_transform``: apply log1p transform to targets (default ``False``)
    """

    requires_censoring_metadata: bool = True

    @property
    def name(self) -> str:
        return "deep_tobit_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the Deep Tobit classifier.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training labels (0/1 for classification; continuous for Tobit fit).
        **kwargs
            Optional: ``X_val``, ``y_val``, ``censored``, ``detection_limits``,
            ``censored_val``, ``detection_limits_val``.
        """
        cfg = dict(self.config)
        hidden_sizes = cfg.get("hidden_sizes", [128, 64, 32])
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.001)
        epochs = cfg.get("epochs", 200)
        batch_size = cfg.get("batch_size", 256)
        patience = cfg.get("patience", 20)
        seed = cfg.get("random_state", 42)
        self._default_dl = cfg.get("default_detection_limit", 0.004)
        self._log_transform = cfg.get("log_transform", False)

        torch.manual_seed(seed)
        self._device = get_device()

        X_np = to_numpy(X_train)
        # Use actual concentrations for Tobit loss when available (classification
        # tasks pass binary y_train labels which are unsuitable for the Tobit
        # likelihood — the loss needs continuous concentrations for detected samples).
        y_conc = kwargs.get("y_concentration")
        if y_conc is not None:
            y_np = np.asarray(y_conc, dtype=np.float64).ravel()
        else:
            y_np = to_array(y_train).astype(np.float64)

        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_: list[str] = list(X_train.columns)

        # Apply log1p transform to targets if requested (stabilises skewed concentrations)
        if self._log_transform:
            y_np = np.log1p(np.clip(y_np, 0.0, None))

        cens_np, dl_np = _prepare_censoring_arrays(
            y_np, len(y_np), kwargs.get("censored"), kwargs.get("detection_limits")
        )
        # Also log-transform detection limits to match target scale
        if self._log_transform:
            dl_np = np.log1p(np.clip(dl_np, 0.0, None))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)

        n_features = X_scaled.shape[1]
        self._model = _DeepTobitNetwork(n_features, hidden_sizes, dropout).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_ds = _TobitDataset(X_scaled, y_np, cens_np, dl_np)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Validation setup
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        early_stop = None
        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_np = self._scaler.transform(to_numpy(X_val))
            y_val_np = to_array(y_val).astype(np.float64)
            y_conc_val = kwargs.get("y_concentration_val")
            if y_conc_val is not None:
                y_val_np = np.asarray(y_conc_val, dtype=np.float64).ravel()
            if self._log_transform:
                y_val_np = np.log1p(np.clip(y_val_np, 0.0, None))
            cens_val, dl_val = _prepare_censoring_arrays(
                y_val_np,
                len(y_val_np),
                kwargs.get("censored_val"),
                kwargs.get("detection_limits_val"),
            )
            if self._log_transform:
                dl_val = np.log1p(np.clip(dl_val, 0.0, None))
            val_ds = _TobitDataset(X_val_np, y_val_np, cens_val, dl_val)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            early_stop = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            self._model.train()
            for X_b, y_b, c_b, dl_b in train_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                c_b = c_b.to(self._device)
                dl_b = dl_b.to(self._device)
                optimizer.zero_grad()
                mu, log_sigma = self._model(X_b)
                loss = tobit_nll(mu, log_sigma, y_b, c_b, dl_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            if early_stop is not None and val_loader is not None:
                vl = self._compute_val_loss(val_loader)
                if early_stop.step(vl, self._model):
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if early_stop is not None:
            early_stop.restore(self._model)

    def _compute_val_loss(self, val_loader: DataLoader) -> float:
        """Compute Tobit NLL on validation data."""
        self._model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X_b, y_b, c_b, dl_b in val_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                c_b = c_b.to(self._device)
                dl_b = dl_b.to(self._device)
                mu, log_sigma = self._model(X_b)
                total_loss += tobit_nll(mu, log_sigma, y_b, c_b, dl_b).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _predict_mu_sigma(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get mu and sigma predictions for input features."""
        self._check_fitted()
        X_scaled = self._scaler.transform(to_numpy(X))
        ds = _TobitDataset(
            X_scaled,
            np.zeros(len(X_scaled)),
            np.zeros(len(X_scaled)),
            np.zeros(len(X_scaled)),
        )
        loader = DataLoader(ds, batch_size=512, shuffle=False)

        self._model.eval()
        all_mu: list[np.ndarray] = []
        all_sigma: list[np.ndarray] = []
        with torch.no_grad():
            for X_b, _, _, _ in loader:
                X_b = X_b.to(self._device)
                mu, log_sigma = self._model(X_b)
                all_mu.append(tensor_to_numpy(mu))
                all_sigma.append(tensor_to_numpy(torch.exp(log_sigma)))

        return np.concatenate(all_mu), np.concatenate(all_sigma)

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels (0/1)."""
        probs = self.predict_proba(X, **kwargs)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities using the Tobit model.

        P(detected) = 1 - Phi((DL - mu) / sigma)

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.
        **kwargs
            Optional: ``detection_limits`` array. If not provided, uses
            ``default_detection_limit`` from config.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)`` — [P(not detected), P(detected)].
        """
        mu, sigma = self._predict_mu_sigma(X)
        dl = kwargs.get("detection_limits")
        if dl is not None:
            dl = np.asarray(dl, dtype=np.float64).ravel()
        else:
            dl = np.full(len(mu), self._default_dl)

        # If model was trained with log-transform, DL must also be log-transformed
        if self._log_transform:
            dl = np.log1p(np.clip(dl, 0.0, None))

        # P(Y > DL) = 1 - Phi((DL - mu) / sigma)
        z = (dl - mu) / np.clip(sigma, 1e-6, None)
        p_detected = 1.0 - norm.cdf(z)
        p_detected = np.clip(p_detected, 0.0, 1.0)
        return np.column_stack([1.0 - p_detected, p_detected])


class DeepTobitRegressor(BaseModel):
    """Censoring-aware MLP regressor using Tobit likelihood.

    Predicts E[Y | Y > DL] = mu + sigma * phi(z) / (1 - Phi(z))
    where z = (DL - mu) / sigma. This truncated expectation properly
    handles left-censored environmental monitoring data.

    Parameters
    ----------
    config : dict, optional
        Same hyperparameters as DeepTobitClassifier.
    """

    requires_censoring_metadata: bool = True

    @property
    def name(self) -> str:
        return "deep_tobit_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the Deep Tobit regressor.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training target values (concentrations).
        **kwargs
            Optional: ``X_val``, ``y_val``, ``censored``, ``detection_limits``,
            ``censored_val``, ``detection_limits_val``.
        """
        # Delegate to classifier's training (same Tobit loss)
        self._clf = DeepTobitClassifier(config=self.config)
        self._clf.fit(X_train, y_train, **kwargs)
        self._model = self._clf._model
        self._scaler = self._clf._scaler
        self._device = self._clf._device
        self._default_dl = self._clf._default_dl
        self._log_transform = self._clf._log_transform
        # Store training range for output clamping
        y_np = to_array(y_train).astype(np.float64)
        self._y_train_max = float(np.nanmax(y_np))
        if hasattr(self._clf, "feature_names_in_"):
            self.feature_names_in_ = self._clf.feature_names_in_

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict concentration values.

        Returns E[Y | Y > DL] = mu + sigma * phi(z) / (1 - Phi(z)),
        the truncated mean of the learned Tobit distribution.

        For numerical stability, uses the asymptotic approximation
        E[Y | Y > DL] ≈ DL + sigma / z when z > 6 (where standard
        formula suffers from catastrophic cancellation in the survival
        function).
        """
        mu, sigma = self._clf._predict_mu_sigma(X)
        dl = kwargs.get("detection_limits")
        if dl is not None:
            dl = np.asarray(dl, dtype=np.float64).ravel()
        else:
            dl = np.full(len(mu), self._default_dl)

        # If model was trained with log-transform, DL must also be log-transformed
        if self._log_transform:
            dl = np.log1p(np.clip(dl, 0.0, None))

        safe_sigma = np.clip(sigma, 1e-6, None)
        z = (dl - mu) / safe_sigma

        # Standard formula where it's numerically safe (z <= 6)
        phi_z = norm.pdf(z)
        survival = 1.0 - norm.cdf(z)
        safe_survival = np.clip(survival, 1e-30, None)
        standard_result = mu + safe_sigma * phi_z / safe_survival

        # Asymptotic approximation for large z: E[Y|Y>DL] ≈ DL + sigma/z
        # Valid when z >> 0 (model predicts value very likely censored)
        safe_z = np.clip(z, 1e-6, None)
        asymptotic_result = dl + safe_sigma / safe_z

        # Blend: use standard formula for z <= 6, asymptotic for z > 6
        truncated_mean = np.where(z <= 6.0, standard_result, asymptotic_result)

        # For very negative z (detection almost certain), truncated mean ≈ mu
        truncated_mean = np.where(z < -6.0, mu, truncated_mean)

        # Back-transform from log space if needed
        if self._log_transform:
            # Clamp log-space values to prevent overflow in expm1.
            max_log = np.log1p(self._y_train_max * 2.0) if self._y_train_max > 0 else 10.0
            truncated_mean = np.clip(truncated_mean, 0.0, max_log)
            truncated_mean = np.expm1(truncated_mean)

        # Concentrations cannot be negative; cap at 2x training max
        y_cap = self._y_train_max * 2.0 if self._y_train_max > 0 else np.inf
        truncated_mean = np.clip(truncated_mean, 0.0, y_cap)

        return np.asarray(truncated_mean)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")
