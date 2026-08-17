"""Invariant Contamination Predictor (ICP).

Combines three complementary techniques for monitoring-invariant prediction:
1. Gradient Reversal Layer (GRL) — forces representations to be
   monitoring-independent by reversing gradients from a monitoring head
2. IPW Reweighting — corrects training distribution for monitoring
   selection bias via inverse propensity weights
3. Group DRO — optimizes worst-case performance across EPA regions
   for geographic robustness

The model includes monitoring features in its input but learns
representations that cannot predict them, removing both direct and
indirect monitoring signal (via proxy features like population).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from aquacontam.models._propensity import compute_ipw_weights, compute_propensity_scores
from aquacontam.models._torch_utils import (
    EarlyStopping,
    get_device,
    tensor_to_numpy,
    to_array,
    to_numpy,
)
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal for domain adaptation.

    Forward pass: identity.
    Backward pass: negate gradients and scale by lambda.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for gradient reversal."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = GradientReversalFunction.apply(x, self.lambda_)
        return result

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_


class _ICPDataset(Dataset):
    """Dataset for ICP: features, labels, IPW weights, confounder targets, group IDs."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        confounders: np.ndarray,
        groups: np.ndarray,
    ) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.confounders = torch.as_tensor(confounders, dtype=torch.float32)
        self.groups = torch.as_tensor(groups, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.weights[idx], self.confounders[idx], self.groups[idx]


class _ICPNetwork(nn.Module):
    """ICP architecture: shared encoder + contamination head + monitoring head.

    The monitoring head receives gradient-reversed representations to ensure
    the encoder cannot encode monitoring information.
    """

    def __init__(
        self,
        n_features: int,
        encoder_sizes: list[int] | None = None,
        head_size: int = 32,
        dropout: float = 0.3,
        n_confounder_targets: int = 2,
    ) -> None:
        super().__init__()
        if encoder_sizes is None:
            encoder_sizes = [128, 64]

        # Shared encoder
        encoder_layers: list[nn.Module] = []
        in_size = n_features
        for h in encoder_sizes:
            encoder_layers.append(nn.Linear(in_size, h))
            encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            in_size = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.repr_dim = in_size

        # Contamination head (task head)
        self.contamination_head = nn.Sequential(
            nn.Linear(in_size, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )

        # Monitoring head (adversary) — predicts monitoring confounders
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.monitoring_head = nn.Sequential(
            nn.Linear(in_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, n_confounder_targets),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(task_logits, confounder_preds, representation)``
        """
        z = self.encoder(x)
        task_out = self.contamination_head(z).squeeze(-1)
        z_reversed = self.grl(z)
        conf_out = self.monitoring_head(z_reversed)
        return task_out, conf_out, z


def _sigmoid_schedule(epoch: int, warmup: int, gamma: float = 10.0) -> float:
    """Sigmoid annealing schedule for lambda_adv.

    Returns 0 during warmup, then smoothly increases to ~1.
    """
    if epoch < warmup:
        return 0.0
    progress = (epoch - warmup) / max(warmup, 1)
    return float(2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0)


class ICPClassifier(BaseModel):
    """Invariant Contamination Predictor — classifier.

    Achieves monitoring-invariant contamination prediction by combining
    gradient reversal, IPW reweighting, and Group DRO.

    Parameters
    ----------
    config : dict, optional
        Hyperparameters:
        - ``encoder_sizes``: hidden layer sizes (default ``[128, 64]``)
        - ``head_size``: task/adversary head width (default ``32``)
        - ``dropout``: dropout rate (default ``0.3``)
        - ``learning_rate``: Adam LR for encoder+task head (default ``0.001``)
        - ``adv_learning_rate``: SGD LR for adversary (default ``0.01``)
        - ``epochs``: max epochs (default ``200``)
        - ``warmup_epochs``: epochs before adversarial training (default ``20``)
        - ``batch_size``: mini-batch size (default ``256``)
        - ``patience``: early stopping patience (default ``20``)
        - ``lambda_adv``: max adversarial loss weight (default ``1.0``)
        - ``lambda_dro``: Group DRO weight (default ``0.1``)
        - ``adv_steps``: adversary updates per encoder step (default ``5``)
        - ``random_state``: seed (default ``42``)
    """

    requires_confounder_targets: bool = True

    @property
    def name(self) -> str:
        return "icp_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the ICP classifier.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Binary training labels.
        **kwargs
            Required: ``confounder_targets`` (n_samples, mean_DL).
            Optional: ``groups`` (EPA region IDs), ``X_val``, ``y_val``,
            ``confounder_targets_val``, ``groups_val``.
        """
        cfg = dict(self.config)
        encoder_sizes = cfg.get("encoder_sizes", [128, 64])
        head_size = cfg.get("head_size", 32)
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.001)
        adv_lr = cfg.get("adv_learning_rate", 0.01)
        epochs = cfg.get("epochs", 200)
        warmup = cfg.get("warmup_epochs", 20)
        batch_size = cfg.get("batch_size", 256)
        patience = cfg.get("patience", 20)
        lambda_adv_max = cfg.get("lambda_adv", 1.0)
        lambda_dro = cfg.get("lambda_dro", 0.1)
        adv_steps = cfg.get("adv_steps", 5)
        seed = cfg.get("random_state", 42)

        torch.manual_seed(seed)
        self._device = get_device()

        X_np = to_numpy(X_train)
        y_np = to_array(y_train).astype(np.float64)

        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_: list[str] = list(X_train.columns)

        # Confounder targets: [log(n_samples), mean_detection_limit]
        confounders = kwargs.get("confounder_targets")
        if confounders is not None:
            conf_np = np.asarray(confounders, dtype=np.float64)
            if conf_np.ndim == 1:
                conf_np = conf_np[:, np.newaxis]
        else:
            # Fallback: no confounder targets, adversary predicts zeros
            conf_np = np.zeros((len(X_np), 2), dtype=np.float64)

        n_conf = conf_np.shape[1]

        # Group IDs for DRO
        groups = kwargs.get("groups")
        if groups is not None:
            groups_np = np.asarray(groups, dtype=np.int64)
        else:
            groups_np = np.zeros(len(X_np), dtype=np.int64)

        n_groups = int(groups_np.max()) + 1

        # IPW weights
        n_samples_col = None
        if isinstance(X_train, pd.DataFrame) and "n_samples" in X_train.columns:
            n_samples_col = X_train["n_samples"].to_numpy()
        elif confounders is not None and conf_np.shape[1] >= 1:
            # First confounder column is log(n_samples)
            n_samples_col = np.expm1(conf_np[:, 0])

        if n_samples_col is not None and len(np.unique(n_samples_col)) > 1:
            propensity = compute_propensity_scores(
                pd.DataFrame(X_np) if not isinstance(X_train, pd.DataFrame) else X_train,
                n_samples_col,
                random_state=seed,
            )
            ipw_weights = compute_ipw_weights(propensity, n_samples_col)
        else:
            ipw_weights = np.ones(len(X_np), dtype=np.float64)

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)

        # Standardize confounder targets
        self._conf_scaler = StandardScaler()
        conf_scaled = self._conf_scaler.fit_transform(conf_np)

        n_features = X_scaled.shape[1]
        self._model = _ICPNetwork(
            n_features,
            encoder_sizes=encoder_sizes,
            head_size=head_size,
            dropout=dropout,
            n_confounder_targets=n_conf,
        ).to(self._device)

        # Separate optimizers
        encoder_task_params = list(self._model.encoder.parameters()) + list(
            self._model.contamination_head.parameters()
        )
        adv_params = list(self._model.monitoring_head.parameters())
        optimizer_task = torch.optim.Adam(encoder_task_params, lr=lr)
        optimizer_adv = torch.optim.SGD(adv_params, lr=adv_lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_task, T_max=epochs)

        train_ds = _ICPDataset(X_scaled, y_np, ipw_weights, conf_scaled, groups_np)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Group DRO weights (exponentiated gradient)
        group_weights = torch.ones(n_groups, dtype=torch.float32, device=self._device) / n_groups

        # Validation setup
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        early_stop = None
        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_np = self._scaler.transform(to_numpy(X_val))
            y_val_np = to_array(y_val).astype(np.float64)
            conf_val = kwargs.get("confounder_targets_val")
            if conf_val is not None:
                conf_val_np = self._conf_scaler.transform(
                    np.asarray(conf_val, dtype=np.float64).reshape(-1, n_conf)
                )
            else:
                conf_val_np = np.zeros((len(X_val_np), n_conf), dtype=np.float64)
            groups_val = kwargs.get("groups_val")
            if groups_val is not None:
                groups_val_np = np.asarray(groups_val, dtype=np.int64)
            else:
                groups_val_np = np.zeros(len(X_val_np), dtype=np.int64)
            ipw_val = np.ones(len(X_val_np), dtype=np.float64)
            val_ds = _ICPDataset(X_val_np, y_val_np, ipw_val, conf_val_np, groups_val_np)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            early_stop = (
                EarlyStopping(patience=patience) if cfg.get("early_stopping", True) else None
            )

        bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        mse_loss = nn.MSELoss()

        self._history: list[dict[str, float]] = []

        for epoch in range(epochs):
            self._model.train()
            lambda_adv = lambda_adv_max * _sigmoid_schedule(epoch, warmup)
            self._model.grl.set_lambda(lambda_adv)

            epoch_task_loss = 0.0
            epoch_adv_loss = 0.0
            n_batches = 0

            for X_b, y_b, w_b, conf_b, g_b in train_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                w_b = w_b.to(self._device)
                conf_b = conf_b.to(self._device)
                g_b = g_b.to(self._device)

                # --- Adversary update steps ---
                if lambda_adv > 0:
                    for _ in range(adv_steps):
                        optimizer_adv.zero_grad()
                        with torch.no_grad():
                            z = self._model.encoder(X_b)
                        conf_pred = self._model.monitoring_head(z)
                        adv_loss = mse_loss(conf_pred, conf_b)
                        adv_loss.backward()
                        optimizer_adv.step()

                # --- Encoder + task head update ---
                optimizer_task.zero_grad()
                task_logits, conf_pred, z = self._model(X_b)

                # IPW-weighted BCE loss
                per_sample_loss = bce_loss(task_logits, y_b) * w_b

                # Group DRO: reweight by group
                if n_groups > 1 and lambda_dro > 0:
                    group_losses = torch.zeros(n_groups, device=self._device)
                    for g in range(n_groups):
                        mask = g_b == g
                        if mask.any():
                            group_losses[g] = per_sample_loss[mask].mean()
                    # Exponentiated gradient update
                    with torch.no_grad():
                        group_weights = group_weights * torch.exp(lambda_dro * group_losses)
                        group_weights = group_weights / group_weights.sum()
                    task_loss = (group_weights * group_losses).sum()
                else:
                    task_loss = per_sample_loss.mean()

                # Adversary loss; the GRL reverses its gradient into the encoder so the
                # encoder learns representations the monitoring head cannot decode. lambda
                # is applied inside the GRL, so the term enters total_loss at unit weight.
                adv_loss_enc = mse_loss(conf_pred, conf_b)
                total_loss = task_loss + adv_loss_enc
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer_task.step()

                epoch_task_loss += task_loss.item()
                epoch_adv_loss += adv_loss_enc.item()
                n_batches += 1

            scheduler.step()

            self._history.append(
                {
                    "epoch": epoch,
                    "task_loss": epoch_task_loss / max(n_batches, 1),
                    "adv_loss": epoch_adv_loss / max(n_batches, 1),
                    "lambda_adv": lambda_adv,
                }
            )

            if epoch % 20 == 0:
                avg_task = epoch_task_loss / max(n_batches, 1)
                avg_adv = epoch_adv_loss / max(n_batches, 1)
                logger.debug(
                    "Epoch %d: task_loss=%.4f, adv_loss=%.4f, λ_adv=%.3f",
                    epoch,
                    avg_task,
                    avg_adv,
                    lambda_adv,
                )

            if early_stop is not None and val_loader is not None:
                vl = self._compute_val_loss(val_loader)
                if early_stop.step(vl, self._model):
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if early_stop is not None:
            early_stop.restore(self._model)

        # Compute and store adversary R² on training data for diagnostics
        self._adversary_r2 = self._compute_adversary_r2(train_loader)
        logger.info("Final adversary R² on training data: %.4f", self._adversary_r2)

    def _compute_val_loss(self, val_loader: DataLoader) -> float:
        """Compute task BCE loss on validation data."""
        self._model.eval()
        total_loss = 0.0
        n_total = 0
        bce = nn.BCEWithLogitsLoss(reduction="sum")
        with torch.no_grad():
            for X_b, y_b, _, _, _ in val_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                logits, _, _ = self._model(X_b)
                total_loss += bce(logits, y_b).item()
                n_total += len(y_b)
        return total_loss / max(n_total, 1)

    def _compute_adversary_r2(self, data_loader: DataLoader) -> float:
        """Compute R² of the adversary predicting confounders from z."""
        self._model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        with torch.no_grad():
            for X_b, _, _, conf_b, _ in data_loader:
                X_b = X_b.to(self._device)
                z = self._model.encoder(X_b)
                conf_pred = self._model.monitoring_head(z)
                all_preds.append(tensor_to_numpy(conf_pred))
                all_targets.append(tensor_to_numpy(conf_b))

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels (0/1)."""
        probs = self.predict_proba(X, **kwargs)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities.

        Uses only the contamination head (monitoring head discarded).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)`` — [P(negative), P(positive)].
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(to_numpy(X))
        X_t = torch.as_tensor(X_scaled, dtype=torch.float32).to(self._device)

        self._model.eval()
        with torch.no_grad():
            z = self._model.encoder(X_t)
            logits = self._model.contamination_head(z).squeeze(-1)
            probs = torch.sigmoid(logits)

        p_pos = tensor_to_numpy(probs)
        p_pos = np.clip(p_pos, 0.0, 1.0)
        return np.column_stack([1.0 - p_pos, p_pos])

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the shared-encoder representation z for inputs X.

        Exposes the monitoring-invariant latent on which the adversary is
        trained, so an independent probe can measure how much monitoring
        information remains recoverable from it.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features (same schema as ``fit``).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, encoder_output_dim)`` representation.
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(to_numpy(X))
        X_t = torch.as_tensor(X_scaled, dtype=torch.float32).to(self._device)
        self._model.eval()
        with torch.no_grad():
            z = self._model.encoder(X_t)
        return tensor_to_numpy(z)

    def get_adversary_r2(self) -> float:
        """Return the adversary R² computed after training."""
        return getattr(self, "_adversary_r2", float("nan"))

    def get_training_history(self) -> list[dict[str, float]]:
        """Return per-epoch training history (task_loss, adv_loss, lambda_adv)."""
        return getattr(self, "_history", [])


class ICPRegressor(BaseModel):
    """Invariant Contamination Predictor — regressor.

    Uses the same architecture as ICPClassifier but with Tobit NLL
    for the task loss, supporting censored regression.

    Parameters
    ----------
    config : dict, optional
        Same hyperparameters as ICPClassifier, plus:
        - ``default_detection_limit``: DL for prediction (default ``0.004``)
        - ``log_transform``: apply log1p to targets (default ``False``)
    """

    requires_confounder_targets: bool = True
    requires_censoring_metadata: bool = True

    @property
    def name(self) -> str:
        return "icp_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the ICP regressor with Tobit loss."""
        from aquacontam.models.deep_tobit import _prepare_censoring_arrays, tobit_nll

        cfg = dict(self.config)
        encoder_sizes = cfg.get("encoder_sizes", [128, 64])
        head_size = cfg.get("head_size", 32)
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.001)
        adv_lr = cfg.get("adv_learning_rate", 0.01)
        epochs = cfg.get("epochs", 200)
        warmup = cfg.get("warmup_epochs", 20)
        batch_size = cfg.get("batch_size", 256)
        patience = cfg.get("patience", 20)
        lambda_adv_max = cfg.get("lambda_adv", 1.0)
        seed = cfg.get("random_state", 42)
        self._default_dl = cfg.get("default_detection_limit", 0.004)
        self._log_transform = cfg.get("log_transform", False)

        torch.manual_seed(seed)
        self._device = get_device()

        X_np = to_numpy(X_train)
        y_np = to_array(y_train).astype(np.float64)

        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_: list[str] = list(X_train.columns)

        # Censoring arrays
        cens_np, dl_np = _prepare_censoring_arrays(
            y_np, len(y_np), kwargs.get("censored"), kwargs.get("detection_limits")
        )

        if self._log_transform:
            y_np = np.log1p(np.clip(y_np, 0.0, None))
            dl_np = np.log1p(np.clip(dl_np, 0.0, None))

        # Store training range
        self._y_train_max = float(np.nanmax(y_np))

        # Confounder targets
        confounders = kwargs.get("confounder_targets")
        if confounders is not None:
            conf_np = np.asarray(confounders, dtype=np.float64)
            if conf_np.ndim == 1:
                conf_np = conf_np[:, np.newaxis]
        else:
            conf_np = np.zeros((len(X_np), 2), dtype=np.float64)

        n_conf = conf_np.shape[1]

        # Groups
        groups = kwargs.get("groups")
        groups_np = (
            np.asarray(groups, dtype=np.int64)
            if groups is not None
            else np.zeros(len(X_np), dtype=np.int64)
        )

        # IPW weights
        ipw_weights = np.ones(len(X_np), dtype=np.float64)

        # Standardize
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)
        self._conf_scaler = StandardScaler()
        conf_scaled = self._conf_scaler.fit_transform(conf_np)

        n_features = X_scaled.shape[1]
        # Regression: task head outputs mu and log_sigma
        self._model = _ICPRegressorNetwork(
            n_features,
            encoder_sizes=encoder_sizes,
            head_size=head_size,
            dropout=dropout,
            n_confounder_targets=n_conf,
        ).to(self._device)

        encoder_task_params = (
            list(self._model.encoder.parameters())
            + list(self._model.mu_head.parameters())
            + list(self._model.log_sigma_head.parameters())
        )
        adv_params = list(self._model.monitoring_head.parameters())
        optimizer_task = torch.optim.Adam(encoder_task_params, lr=lr)
        optimizer_adv = torch.optim.SGD(adv_params, lr=adv_lr, momentum=0.9)

        # Custom dataset with censoring info
        train_ds = _ICPRegressorDataset(
            X_scaled, y_np, ipw_weights, conf_scaled, groups_np, cens_np, dl_np
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Validation
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        early_stop = None
        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_scaled = self._scaler.transform(to_numpy(X_val))
            y_val_np = to_array(y_val).astype(np.float64)
            cens_val, dl_val = _prepare_censoring_arrays(
                y_val_np,
                len(y_val_np),
                kwargs.get("censored_val"),
                kwargs.get("detection_limits_val"),
            )
            if self._log_transform:
                y_val_np = np.log1p(np.clip(y_val_np, 0.0, None))
                dl_val = np.log1p(np.clip(dl_val, 0.0, None))
            conf_val = kwargs.get("confounder_targets_val")
            conf_val_np = (
                self._conf_scaler.transform(
                    np.asarray(conf_val, dtype=np.float64).reshape(-1, n_conf)
                )
                if conf_val is not None
                else np.zeros((len(X_val_scaled), n_conf), dtype=np.float64)
            )
            groups_val_np = (
                np.asarray(kwargs.get("groups_val"), dtype=np.int64)
                if kwargs.get("groups_val") is not None
                else np.zeros(len(X_val_scaled), dtype=np.int64)
            )
            ipw_val = np.ones(len(X_val_scaled), dtype=np.float64)
            val_ds = _ICPRegressorDataset(
                X_val_scaled, y_val_np, ipw_val, conf_val_np, groups_val_np, cens_val, dl_val
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            early_stop = (
                EarlyStopping(patience=patience) if cfg.get("early_stopping", True) else None
            )

        mse_loss = nn.MSELoss()

        self._history: list[dict[str, float]] = []

        for epoch in range(epochs):
            self._model.train()
            lambda_adv = lambda_adv_max * _sigmoid_schedule(epoch, warmup)
            self._model.grl.set_lambda(lambda_adv)

            epoch_task_loss = 0.0
            epoch_adv_loss = 0.0
            n_batches = 0

            for X_b, y_b, w_b, conf_b, _g_b, cens_b, dl_b in train_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                w_b = w_b.to(self._device)
                conf_b = conf_b.to(self._device)
                cens_b = cens_b.to(self._device)
                dl_b = dl_b.to(self._device)

                # Adversary steps
                if lambda_adv > 0:
                    for _ in range(cfg.get("adv_steps", 5)):
                        optimizer_adv.zero_grad()
                        with torch.no_grad():
                            z = self._model.encoder(X_b)
                        conf_pred = self._model.monitoring_head(z)
                        adv_loss = mse_loss(conf_pred, conf_b)
                        adv_loss.backward()
                        optimizer_adv.step()

                # Task update with Tobit NLL + GRL-reversed adversary (lambda applied
                # inside the GRL); the encoder learns to hide the monitoring confounders.
                optimizer_task.zero_grad()
                mu, log_sigma, conf_pred, z = self._model(X_b)
                task_loss = tobit_nll(mu, log_sigma, y_b, cens_b, dl_b)
                adv_loss_enc = mse_loss(conf_pred, conf_b)
                total_loss = task_loss + adv_loss_enc
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer_task.step()

                epoch_task_loss += task_loss.item()
                epoch_adv_loss += adv_loss_enc.item()
                n_batches += 1

            self._history.append(
                {
                    "epoch": epoch,
                    "task_loss": epoch_task_loss / max(n_batches, 1),
                    "adv_loss": epoch_adv_loss / max(n_batches, 1),
                    "lambda_adv": lambda_adv,
                }
            )

            if early_stop is not None and val_loader is not None:
                vl = self._compute_val_tobit_loss(val_loader, tobit_nll)
                if early_stop.step(vl, self._model):
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if early_stop is not None:
            early_stop.restore(self._model)

    def get_training_history(self) -> list[dict[str, float]]:
        """Return per-epoch training history (task_loss, adv_loss, lambda_adv)."""
        return getattr(self, "_history", [])

    def _compute_val_tobit_loss(self, val_loader: DataLoader, tobit_nll_fn: Any) -> float:
        self._model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X_b, y_b, _, _, _, cens_b, dl_b in val_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                cens_b = cens_b.to(self._device)
                dl_b = dl_b.to(self._device)
                mu, log_sigma, _, _ = self._model(X_b)
                total_loss += tobit_nll_fn(mu, log_sigma, y_b, cens_b, dl_b).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict concentration values."""
        self._check_fitted()
        from scipy.stats import norm as scipy_norm

        X_scaled = self._scaler.transform(to_numpy(X))
        X_t = torch.as_tensor(X_scaled, dtype=torch.float32).to(self._device)

        self._model.eval()
        with torch.no_grad():
            z = self._model.encoder(X_t)
            mu = self._model.mu_head(z).squeeze(-1)
            log_sigma = self._model.log_sigma_head(z).squeeze(-1).clamp(-5.0, 5.0)
            sigma = torch.exp(log_sigma)

        mu_np = tensor_to_numpy(mu)
        sigma_np = tensor_to_numpy(sigma)

        dl = kwargs.get("detection_limits")
        if dl is not None:
            dl = np.asarray(dl, dtype=np.float64).ravel()
        else:
            dl = np.full(len(mu_np), self._default_dl)
        if self._log_transform:
            dl = np.log1p(np.clip(dl, 0.0, None))

        safe_sigma = np.clip(sigma_np, 1e-6, None)
        z_val = (dl - mu_np) / safe_sigma

        phi_z = scipy_norm.pdf(z_val)
        survival = 1.0 - scipy_norm.cdf(z_val)
        safe_survival = np.clip(survival, 1e-30, None)
        standard_result = mu_np + safe_sigma * phi_z / safe_survival

        safe_z = np.clip(z_val, 1e-6, None)
        asymptotic_result = dl + safe_sigma / safe_z

        truncated_mean = np.where(z_val <= 6.0, standard_result, asymptotic_result)
        truncated_mean = np.where(z_val < -6.0, mu_np, truncated_mean)

        if self._log_transform:
            max_log = np.log1p(self._y_train_max * 2.0) if self._y_train_max > 0 else 10.0
            truncated_mean = np.clip(truncated_mean, 0.0, max_log)
            truncated_mean = np.expm1(truncated_mean)

        y_cap = self._y_train_max * 2.0 if self._y_train_max > 0 else np.inf
        truncated_mean = np.clip(truncated_mean, 0.0, y_cap)
        return np.asarray(truncated_mean)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError("Regression models do not support predict_proba")


class _ICPRegressorNetwork(nn.Module):
    """ICP regressor network with mu/log_sigma heads + monitoring adversary."""

    def __init__(
        self,
        n_features: int,
        encoder_sizes: list[int] | None = None,
        head_size: int = 32,
        dropout: float = 0.3,
        n_confounder_targets: int = 2,
    ) -> None:
        super().__init__()
        if encoder_sizes is None:
            encoder_sizes = [128, 64]

        encoder_layers: list[nn.Module] = []
        in_size = n_features
        for h in encoder_sizes:
            encoder_layers.append(nn.Linear(in_size, h))
            encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            in_size = h
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_head = nn.Sequential(
            nn.Linear(in_size, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(in_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )

        self.grl = GradientReversalLayer(lambda_=0.0)
        self.monitoring_head = nn.Sequential(
            nn.Linear(in_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, n_confounder_targets),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        mu = self.mu_head(z).squeeze(-1)
        log_sigma = self.log_sigma_head(z).squeeze(-1).clamp(-5.0, 5.0)
        z_rev = self.grl(z)
        conf_pred = self.monitoring_head(z_rev)
        return mu, log_sigma, conf_pred, z


class _ICPRegressorDataset(Dataset):
    """Dataset for ICP regressor with censoring info."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        confounders: np.ndarray,
        groups: np.ndarray,
        censored: np.ndarray,
        detection_limits: np.ndarray,
    ) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.confounders = torch.as_tensor(confounders, dtype=torch.float32)
        self.groups = torch.as_tensor(groups, dtype=torch.long)
        self.censored = torch.as_tensor(censored, dtype=torch.float32)
        self.detection_limits = torch.as_tensor(detection_limits, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.X[idx],
            self.y[idx],
            self.weights[idx],
            self.confounders[idx],
            self.groups[idx],
            self.censored[idx],
            self.detection_limits[idx],
        )
