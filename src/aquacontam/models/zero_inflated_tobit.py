"""Zero-Inflated Deep Tobit (ZIDT): gate network + Tobit likelihood.

Extends the Deep Tobit by adding a gate head that separates structural zeros
(systems with genuinely no PFAS) from censored positives (systems with
concentrations below the detection limit). The standard Tobit conflates
these two populations, leading to poor predictions under extreme censoring.

Architecture::

    SharedTrunk (BatchNorm -> Linear -> ReLU -> Dropout) x N
        GateHead: Linear(hidden, 1) -> Sigmoid -> P(structural_zero)
        MuHead: Linear(hidden, 1)
        LogSigmaHead: Linear(hidden, 1)

Loss (Zero-Inflated Tobit NLL):
- Detected: -log(1 - pi) - log N(y | mu, sigma)
- Censored: -log(pi + (1-pi) * Phi((DL-mu)/sigma))   (log-sum-exp for stability)
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
from torch.utils.data import DataLoader

from aquacontam.models._torch_utils import (
    EarlyStopping,
    get_device,
    tensor_to_numpy,
    to_array,
    to_numpy,
)
from aquacontam.models.base import BaseModel
from aquacontam.models.deep_tobit import _prepare_censoring_arrays, _TobitDataset

logger = logging.getLogger(__name__)


class _ZeroInflatedTobitNetwork(nn.Module):
    """MLP with three heads: gate (π), mu, and log_sigma."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int],
        dropout: float = 0.4,
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
        self.gate_head = nn.Linear(in_size, 1)  # P(structural_zero)
        self.mu_head = nn.Linear(in_size, 1)
        self.log_sigma_head = nn.Linear(in_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        pi = torch.sigmoid(self.gate_head(h).squeeze(-1))  # P(structural_zero)
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1).clamp(-5.0, 5.0)
        return pi, mu, log_sigma


def zero_inflated_tobit_nll(
    pi: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    y: torch.Tensor,
    censored: torch.Tensor,
    detection_limits: torch.Tensor,
) -> torch.Tensor:
    """Compute zero-inflated Tobit negative log-likelihood.

    Parameters
    ----------
    pi : torch.Tensor
        Gate probability P(structural_zero), shape (N,).
    mu : torch.Tensor
        Predicted mean of positive component, shape (N,).
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
    # Clamp pi away from 0 and 1 for numerical stability
    pi_safe = pi.clamp(1e-6, 1.0 - 1e-6)

    # --- Detected samples: -log(1 - pi) - log N(y | mu, sigma) ---
    z_obs = (y - mu) / sigma
    log_pdf = -0.5 * z_obs**2 - log_sigma - 0.5 * np.log(2 * np.pi)
    nll_detected = -torch.log(1.0 - pi_safe) - log_pdf

    # --- Censored samples: -log(pi + (1-pi) * Phi((DL-mu)/sigma)) ---
    z_dl = (detection_limits - mu) / sigma
    log_cdf = torch.log(0.5 * torch.erfc(-z_dl / np.sqrt(2)) + 1e-10)

    # Log-sum-exp for stability: log(pi + (1-pi)*Phi) = log(exp(log_pi) + exp(log(1-pi) + log_Phi))
    log_pi = torch.log(pi_safe)
    log_one_minus_pi = torch.log(1.0 - pi_safe)
    log_mixture = torch.logaddexp(log_pi, log_one_minus_pi + log_cdf)
    nll_censored = -log_mixture

    # Combine
    nll = (1.0 - censored) * nll_detected + censored * nll_censored
    result: torch.Tensor = nll.mean()
    return result


class ZeroInflatedTobitClassifier(BaseModel):
    """Zero-inflated Tobit MLP classifier.

    Predicts P(detected) = (1 - pi) * (1 - Phi((DL - mu)/sigma))

    Parameters
    ----------
    config : dict, optional
        Hyperparameters:
        - ``hidden_sizes``: list of hidden layer sizes (default ``[64, 32]``)
        - ``dropout``: dropout rate (default ``0.4``)
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
        return "zi_tobit_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        cfg = dict(self.config)
        hidden_sizes = cfg.get("hidden_sizes", [64, 32])
        dropout = cfg.get("dropout", 0.4)
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
        y_conc = kwargs.get("y_concentration")
        if y_conc is not None:
            y_np = np.asarray(y_conc, dtype=np.float64).ravel()
        else:
            y_np = to_array(y_train).astype(np.float64)

        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_: list[str] = list(X_train.columns)

        if self._log_transform:
            y_np = np.log1p(np.clip(y_np, 0.0, None))

        cens_np, dl_np = _prepare_censoring_arrays(
            y_np, len(y_np), kwargs.get("censored"), kwargs.get("detection_limits")
        )
        if self._log_transform:
            dl_np = np.log1p(np.clip(dl_np, 0.0, None))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)

        n_features = X_scaled.shape[1]
        self._model = _ZeroInflatedTobitNetwork(n_features, hidden_sizes, dropout).to(self._device)
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
                pi, mu, log_sigma = self._model(X_b)
                loss = zero_inflated_tobit_nll(pi, mu, log_sigma, y_b, c_b, dl_b)
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
        self._model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X_b, y_b, c_b, dl_b in val_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                c_b = c_b.to(self._device)
                dl_b = dl_b.to(self._device)
                pi, mu, log_sigma = self._model(X_b)
                total_loss += zero_inflated_tobit_nll(pi, mu, log_sigma, y_b, c_b, dl_b).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _predict_params(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get pi, mu, sigma predictions."""
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
        all_pi: list[np.ndarray] = []
        all_mu: list[np.ndarray] = []
        all_sigma: list[np.ndarray] = []
        with torch.no_grad():
            for X_b, _, _, _ in loader:
                X_b = X_b.to(self._device)
                pi, mu, log_sigma = self._model(X_b)
                all_pi.append(tensor_to_numpy(pi))
                all_mu.append(tensor_to_numpy(mu))
                all_sigma.append(tensor_to_numpy(torch.exp(log_sigma)))

        return np.concatenate(all_pi), np.concatenate(all_mu), np.concatenate(all_sigma)

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels (0/1)."""
        probs = self.predict_proba(X, **kwargs)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities.

        P(detected) = (1 - pi) * (1 - Phi((DL - mu)/sigma))
        """
        pi, mu, sigma = self._predict_params(X)
        dl = kwargs.get("detection_limits")
        if dl is not None:
            dl = np.asarray(dl, dtype=np.float64).ravel()
        else:
            dl = np.full(len(mu), self._default_dl)

        if self._log_transform:
            dl = np.log1p(np.clip(dl, 0.0, None))

        z = (dl - mu) / np.clip(sigma, 1e-6, None)
        p_above_dl = 1.0 - norm.cdf(z)
        p_detected = (1.0 - pi) * p_above_dl
        p_detected = np.clip(p_detected, 0.0, 1.0)
        return np.column_stack([1.0 - p_detected, p_detected])


class ZeroInflatedTobitRegressor(BaseModel):
    """Zero-inflated Tobit MLP regressor.

    Predicts E[Y] = (1 - pi) * E[Y | Y > DL, positive component]

    Parameters
    ----------
    config : dict, optional
        Same hyperparameters as ZeroInflatedTobitClassifier.
    """

    requires_censoring_metadata: bool = True

    @property
    def name(self) -> str:
        return "zi_tobit_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        self._clf = ZeroInflatedTobitClassifier(config=self.config)
        self._clf.fit(X_train, y_train, **kwargs)
        self._model = self._clf._model
        self._scaler = self._clf._scaler
        self._device = self._clf._device
        self._default_dl = self._clf._default_dl
        self._log_transform = self._clf._log_transform
        y_np = to_array(y_train).astype(np.float64)
        self._y_train_max = float(np.nanmax(y_np))
        if hasattr(self._clf, "feature_names_in_"):
            self.feature_names_in_ = self._clf.feature_names_in_

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict concentration values.

        Returns (1 - pi) * truncated_mean(mu, sigma, DL).
        """
        pi, mu, sigma = self._clf._predict_params(X)
        dl = kwargs.get("detection_limits")
        if dl is not None:
            dl = np.asarray(dl, dtype=np.float64).ravel()
        else:
            dl = np.full(len(mu), self._default_dl)

        if self._log_transform:
            dl = np.log1p(np.clip(dl, 0.0, None))

        safe_sigma = np.clip(sigma, 1e-6, None)
        z = (dl - mu) / safe_sigma

        # Truncated mean E[Y | Y > DL] = mu + sigma * phi(z) / (1 - Phi(z))
        phi_z = norm.pdf(z)
        survival = 1.0 - norm.cdf(z)
        safe_survival = np.clip(survival, 1e-30, None)
        standard_result = mu + safe_sigma * phi_z / safe_survival

        # Asymptotic for large z
        safe_z = np.clip(z, 1e-6, None)
        asymptotic_result = dl + safe_sigma / safe_z

        truncated_mean = np.where(z <= 6.0, standard_result, asymptotic_result)
        truncated_mean = np.where(z < -6.0, mu, truncated_mean)

        # Weight by probability of being a positive (non-structural-zero) system
        result = (1.0 - pi) * truncated_mean

        if self._log_transform:
            max_log = np.log1p(self._y_train_max * 2.0) if self._y_train_max > 0 else 10.0
            result = np.clip(result, 0.0, max_log)
            result = np.expm1(result)

        y_cap = self._y_train_max * 2.0 if self._y_train_max > 0 else np.inf
        result = np.clip(result, 0.0, y_cap)
        return np.asarray(result)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")
