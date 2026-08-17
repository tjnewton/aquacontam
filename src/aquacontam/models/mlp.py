"""MLP (Multi-Layer Perceptron) models for classification and regression."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from aquacontam.models._torch_utils import (
    EarlyStopping,
    TabularDataset,
    compute_validation_loss,
    get_device,
    tensor_to_numpy,
    to_array,
    to_numpy,
)
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class _MLPNetwork(nn.Module):
    """Feedforward MLP with BatchNorm and dropout."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int],
        n_outputs: int,
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
        layers.append(nn.Linear(in_size, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[no-any-return]


class MLPClassifier(BaseModel):
    """MLP binary classifier with StandardScaler, BatchNorm, and dropout.

    Parameters
    ----------
    config : dict, optional
        Hyperparameters:
        - ``hidden_sizes``: list of hidden layer sizes (default ``[64, 32]``)
        - ``dropout``: dropout rate (default ``0.3``)
        - ``learning_rate``: Adam LR (default ``0.001``)
        - ``epochs``: max epochs (default ``100``)
        - ``batch_size``: mini-batch size (default ``256``)
        - ``patience``: early stopping patience (default ``10``)
        - ``random_state``: seed (default ``42``)
    """

    @property
    def name(self) -> str:
        return "mlp_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the MLP classifier.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training labels (0/1).
        **kwargs
            Optional: ``X_val``, ``y_val`` for early stopping.
        """
        cfg = dict(self.config)
        hidden_sizes = cfg.get("hidden_sizes", [64, 32])
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.001)
        epochs = cfg.get("epochs", 100)
        batch_size = cfg.get("batch_size", 256)
        patience = cfg.get("patience", 10)
        seed = cfg.get("random_state", 42)

        torch.manual_seed(seed)
        self._device = get_device()

        X_np = to_numpy(X_train)
        y_np = to_array(y_train).astype(np.float64)

        # Store feature names for prediction-time alignment
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_: list[str] = list(X_train.columns)

        # Fit scaler on training data
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)

        n_features = X_scaled.shape[1]
        self._model = _MLPNetwork(n_features, hidden_sizes, 1, dropout).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        # Class weighting for imbalanced data
        n_pos = float(y_np.sum())
        n_neg = float(len(y_np) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TabularDataset(X_scaled, y_np)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Validation setup
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        early_stop = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_np = self._scaler.transform(to_numpy(X_val))
            y_val_np = to_array(y_val).astype(np.float64)
            val_ds = TabularDataset(X_val_np, y_val_np)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            early_stop = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            self._model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                logits = self._model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            if early_stop is not None:
                vl = compute_validation_loss(self._model, val_loader, criterion, self._device)
                if early_stop.step(vl, self._model):
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if early_stop is not None:
            early_stop.restore(self._model)

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels (0/1)."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)`` — probabilities for each class.
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(to_numpy(X))
        ds = TabularDataset(X_scaled, np.zeros(len(X_scaled)))
        loader = DataLoader(ds, batch_size=512, shuffle=False)

        self._model.eval()
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self._device)
                logits = self._model(X_batch).squeeze(-1)
                probs = tensor_to_numpy(torch.sigmoid(logits))
                all_probs.append(probs)

        pos_prob = np.concatenate(all_probs)
        return np.column_stack([1 - pos_prob, pos_prob])


class MLPRegressor(BaseModel):
    """MLP regressor for concentration prediction.

    Parameters
    ----------
    config : dict, optional
        Same hyperparameters as MLPClassifier.
    """

    @property
    def name(self) -> str:
        return "mlp_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the MLP regressor."""
        cfg = dict(self.config)
        hidden_sizes = cfg.get("hidden_sizes", [64, 32])
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.001)
        epochs = cfg.get("epochs", 100)
        batch_size = cfg.get("batch_size", 256)
        patience = cfg.get("patience", 10)
        seed = cfg.get("random_state", 42)

        torch.manual_seed(seed)
        self._device = get_device()

        X_np = to_numpy(X_train)
        y_np = to_array(y_train).astype(np.float64)

        # Store feature names for prediction-time alignment
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_: list[str] = list(X_train.columns)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)

        n_features = X_scaled.shape[1]
        self._model = _MLPNetwork(n_features, hidden_sizes, 1, dropout).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_ds = TabularDataset(X_scaled, y_np)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        early_stop = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_np = self._scaler.transform(to_numpy(X_val))
            y_val_np = to_array(y_val).astype(np.float64)
            val_ds = TabularDataset(X_val_np, y_val_np)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            early_stop = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            self._model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                preds = self._model(X_batch).squeeze(-1)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

            if early_stop is not None:
                vl = compute_validation_loss(self._model, val_loader, criterion, self._device)
                if early_stop.step(vl, self._model):
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if early_stop is not None:
            early_stop.restore(self._model)

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict concentration values."""
        self._check_fitted()
        X_scaled = self._scaler.transform(to_numpy(X))
        ds = TabularDataset(X_scaled, np.zeros(len(X_scaled)))
        loader = DataLoader(ds, batch_size=512, shuffle=False)

        self._model.eval()
        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self._device)
                preds = tensor_to_numpy(self._model(X_batch).squeeze(-1))
                all_preds.append(preds)

        return np.concatenate(all_preds)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")
