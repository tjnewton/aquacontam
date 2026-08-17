"""1D-CNN models treating feature vectors as 1D signals."""

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


def _domain_order_columns(columns: list[str]) -> list[str]:
    """Order columns by domain group (proximity → land use → hydro → demographics).

    Columns not matching any domain prefix are appended at the end in
    alphabetical order.
    """
    from aquacontam._constants import DOMAIN_FEATURE_ORDER_PREFIXES

    ordered: list[str] = []
    remaining = set(columns)

    for prefix_group in DOMAIN_FEATURE_ORDER_PREFIXES:
        group_cols = sorted(c for c in remaining if any(c.startswith(p) for p in prefix_group))
        ordered.extend(group_cols)
        remaining -= set(group_cols)

    # Append unmatched columns alphabetically
    ordered.extend(sorted(remaining))
    return ordered


class _CNN1DNetwork(nn.Module):
    """1D convolutional network for tabular data.

    Treats each sample's feature vector as a 1-channel 1D signal.
    """

    def __init__(
        self,
        n_features: int,
        n_filters: list[int],
        kernel_size: int,
        n_outputs: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_channels = 1
        for f in n_filters:
            conv_layers.append(nn.Conv1d(in_channels, f, kernel_size, padding="same"))
            conv_layers.append(nn.BatchNorm1d(f))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = f
        self.conv = nn.Sequential(*conv_layers)

        # Adaptive pooling collapses spatial dimension to 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)  # type: ignore[no-any-return]


class CNN1DClassifier(BaseModel):
    """1D-CNN binary classifier.

    Parameters
    ----------
    config : dict, optional
        Hyperparameters:
        - ``n_filters``: list of filter counts per conv layer (default ``[32, 64]``)
        - ``kernel_size``: convolution kernel size (default ``3``)
        - ``dropout``: dropout rate (default ``0.3``)
        - ``learning_rate``: Adam LR (default ``0.001``)
        - ``epochs``: max epochs (default ``100``)
        - ``batch_size``: mini-batch size (default ``256``)
        - ``patience``: early stopping patience (default ``10``)
        - ``random_state``: seed (default ``42``)
        - ``feature_order``: ``"alphabetical"`` (default), ``"random"``, or
          ``"domain"`` — controls column ordering before convolution.
          When ``"random"``, columns are shuffled using ``random_state``
          as seed to test sensitivity to arbitrary feature adjacency.
          When ``"domain"``, columns are grouped by feature domain
          (proximity → land use → hydrogeology → demographics).
    """

    @property
    def name(self) -> str:
        return "cnn1d_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the 1D-CNN classifier.

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
        n_filters = cfg.get("n_filters", [32, 64])
        kernel_size = cfg.get("kernel_size", 3)
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.001)
        epochs = cfg.get("epochs", 100)
        batch_size = cfg.get("batch_size", 256)
        patience = cfg.get("patience", 10)
        seed = cfg.get("random_state", 42)

        feature_order = cfg.get("feature_order", "alphabetical")

        torch.manual_seed(seed)
        self._device = get_device()

        # Apply feature ordering (alphabetical, random, or domain)
        X_arr: np.ndarray
        if isinstance(X_train, pd.DataFrame):
            df_train = X_train
            if feature_order == "random":
                rng = np.random.RandomState(seed)
                self._col_order: list[str] = list(rng.permutation(df_train.columns))
                df_train = df_train[self._col_order]
            elif feature_order == "domain":
                self._col_order = _domain_order_columns(list(df_train.columns))
                df_train = df_train[self._col_order]
            else:
                self._col_order = sorted(df_train.columns)
                df_train = df_train[self._col_order]
            self.feature_names_in_: list[str] = list(df_train.columns)
            X_arr = to_numpy(df_train)
        else:
            if feature_order == "random":
                rng = np.random.RandomState(seed)
                perm = rng.permutation(X_train.shape[1])
                self._col_perm: np.ndarray = perm
                X_train = X_train[:, perm]
            else:
                self._col_perm = np.arange(X_train.shape[1])
            X_arr = to_numpy(X_train)

        y_np = to_array(y_train).astype(np.float64)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)

        n_features = X_scaled.shape[1]
        self._model = _CNN1DNetwork(n_features, n_filters, kernel_size, 1, dropout).to(
            self._device
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        n_pos = float(y_np.sum())
        n_neg = float(len(y_np) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TabularDataset(X_scaled, y_np)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        early_stop = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            if isinstance(X_val, pd.DataFrame) and hasattr(self, "_col_order"):
                X_val = X_val[self._col_order]
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
        # Apply same column ordering used during fit
        if isinstance(X, pd.DataFrame) and hasattr(self, "_col_order"):
            X = X.reindex(columns=self._col_order, fill_value=0.0)
        elif isinstance(X, np.ndarray) and hasattr(self, "_col_perm"):
            X = X[:, self._col_perm]
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


class CNN1DRegressor(BaseModel):
    """1D-CNN regressor for concentration prediction.

    Parameters
    ----------
    config : dict, optional
        Same hyperparameters as CNN1DClassifier, including ``feature_order``
        (``"alphabetical"``, ``"random"``, or ``"domain"``).
    """

    @property
    def name(self) -> str:
        return "cnn1d_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the 1D-CNN regressor."""
        cfg = dict(self.config)
        n_filters = cfg.get("n_filters", [32, 64])
        kernel_size = cfg.get("kernel_size", 3)
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.001)
        epochs = cfg.get("epochs", 100)
        batch_size = cfg.get("batch_size", 256)
        patience = cfg.get("patience", 10)
        seed = cfg.get("random_state", 42)

        feature_order = cfg.get("feature_order", "alphabetical")

        torch.manual_seed(seed)
        self._device = get_device()

        # Apply feature ordering (alphabetical, random, or domain)
        X_arr: np.ndarray
        if isinstance(X_train, pd.DataFrame):
            df_train = X_train
            if feature_order == "random":
                rng = np.random.RandomState(seed)
                self._col_order: list[str] = list(rng.permutation(df_train.columns))
                df_train = df_train[self._col_order]
            elif feature_order == "domain":
                self._col_order = _domain_order_columns(list(df_train.columns))
                df_train = df_train[self._col_order]
            else:
                self._col_order = sorted(df_train.columns)
                df_train = df_train[self._col_order]
            self.feature_names_in_: list[str] = list(df_train.columns)
            X_arr = to_numpy(df_train)
        else:
            if feature_order == "random":
                rng = np.random.RandomState(seed)
                perm = rng.permutation(X_train.shape[1])
                self._col_perm: np.ndarray = perm
                X_train = X_train[:, perm]
            else:
                self._col_perm = np.arange(X_train.shape[1])
            X_arr = to_numpy(X_train)

        y_np = to_array(y_train).astype(np.float64)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)

        n_features = X_scaled.shape[1]
        self._model = _CNN1DNetwork(n_features, n_filters, kernel_size, 1, dropout).to(
            self._device
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_ds = TabularDataset(X_scaled, y_np)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        early_stop = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            if isinstance(X_val, pd.DataFrame) and hasattr(self, "_col_order"):
                X_val = X_val[self._col_order]
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
        # Apply same column ordering used during fit
        if isinstance(X, pd.DataFrame) and hasattr(self, "_col_order"):
            X = X.reindex(columns=self._col_order, fill_value=0.0)
        elif isinstance(X, np.ndarray) and hasattr(self, "_col_perm"):
            X = X[:, self._col_perm]
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
