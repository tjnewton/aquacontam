"""Graph Neural Network classifiers for spatial contamination prediction.

Builds a KNN spatial graph from system coordinates and applies GCN or
GraphSAGE convolutions. Operates in transductive mode with train/val/test
node masks from geographic splits.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — PyTorch convention
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

from aquacontam.models._torch_utils import get_device, tensor_to_numpy, to_array, to_numpy
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class _GCNNetwork(nn.Module):
    """Two-layer GCN with dropout and ReLU activation.

    Parameters
    ----------
    n_features : int
        Number of input features per node.
    n_hidden : int
        Hidden layer width.
    n_outputs : int
        Number of output classes.
    dropout : float
        Dropout probability applied after the first convolution.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_outputs: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_outputs)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through two GCN layers.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape ``(n_nodes, n_features)``.
        edge_index : torch.Tensor
            COO edge index, shape ``(2, n_edges)``.

        Returns
        -------
        torch.Tensor
            Logits, shape ``(n_nodes, n_outputs)``.
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h  # type: ignore[no-any-return]


class _SAGENetwork(nn.Module):
    """Two-layer GraphSAGE with dropout and ReLU activation.

    Parameters
    ----------
    n_features : int
        Number of input features per node.
    n_hidden : int
        Hidden layer width.
    n_outputs : int
        Number of output classes.
    dropout : float
        Dropout probability applied after the first convolution.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_outputs: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(n_features, n_hidden)
        self.conv2 = SAGEConv(n_hidden, n_outputs)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through two GraphSAGE layers.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape ``(n_nodes, n_features)``.
        edge_index : torch.Tensor
            COO edge index, shape ``(2, n_edges)``.

        Returns
        -------
        torch.Tensor
            Logits, shape ``(n_nodes, n_outputs)``.
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h  # type: ignore[no-any-return]


def _build_graph(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    k: int = 10,
) -> Data:
    """Build a KNN spatial graph from node features and coordinates.

    Parameters
    ----------
    X : np.ndarray
        Node feature matrix, shape ``(n_nodes, n_features)``, float32.
    y : np.ndarray
        Node labels, shape ``(n_nodes,)``, int64.
    coords : np.ndarray
        Spatial coordinates in EPSG:5070, shape ``(n_nodes, 2)``, float32.
    k : int
        Number of nearest neighbours for KNN graph construction.

    Returns
    -------
    torch_geometric.data.Data
        PyG Data object with ``x``, ``edge_index``, and ``y`` attributes.
    """
    # Clamp k to at most n-1 to avoid requesting more neighbours than exist
    n = len(X)
    k_actual = min(k, n - 1)
    k_actual = max(k_actual, 1)

    # Build KNN using scipy cKDTree (avoids torch-cluster dependency)
    tree = cKDTree(coords.astype(np.float64))
    # Query k+1 neighbours (includes self), then drop self-loops
    _, indices = tree.query(coords.astype(np.float64), k=k_actual + 1)

    # Build edge list: each node i connects to its k nearest neighbours
    src_list: list[int] = []
    tgt_list: list[int] = []
    for i in range(n):
        for j_idx in range(k_actual + 1):
            j = indices[i, j_idx]
            if j != i:  # skip self-loop
                src_list.append(i)
                tgt_list.append(j)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    return Data(
        x=torch.from_numpy(X),
        edge_index=edge_index,
        y=torch.from_numpy(y),
    )


class GNNClassifier(BaseModel):
    """Graph neural network binary classifier.

    Builds a KNN spatial graph from projected coordinates (EPSG:5070) and
    trains a GCN or GraphSAGE model in transductive mode. For test nodes
    not present in the training graph, prediction falls back to the learned
    linear transformations without message passing (MLP-like inference).

    Parameters
    ----------
    config : dict, optional
        Hyperparameters:

        - ``arch``: ``"gcn"`` or ``"sage"`` (default ``"gcn"``)
        - ``n_hidden``: hidden layer width (default ``64``)
        - ``dropout``: dropout rate (default ``0.3``)
        - ``learning_rate``: Adam LR (default ``0.01``)
        - ``epochs``: max training epochs (default ``200``)
        - ``patience``: early stopping patience (default ``20``)
        - ``k``: KNN neighbours for graph construction (default ``10``)
        - ``random_state``: seed (default ``42``)
    """

    #: GNN builds a spatial graph and requires per-sample coordinates at fit time.
    requires_coords: bool = True

    @property
    def name(self) -> str:
        arch = self.config.get("arch", "gcn")
        return f"gnn_{arch}_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the GNN classifier.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training labels (0/1).
        **kwargs
            Required: ``coords`` — (n_train, 2) array in EPSG:5070.
            Optional: ``X_val``, ``y_val``, ``coords_val`` for early stopping.
        """
        cfg = dict(self.config)
        arch = cfg.get("arch", "gcn")
        n_hidden = cfg.get("n_hidden", 64)
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.01)
        epochs = cfg.get("epochs", 200)
        patience = cfg.get("patience", 20)
        k = cfg.get("k", 10)
        seed = cfg.get("random_state", 42)

        torch.manual_seed(seed)
        self._device = get_device()
        self._dropout = dropout

        coords = kwargs.get("coords")
        if coords is None:
            raise ValueError(
                "GNN requires coords kwarg — pass (n, 2) array of EPSG:5070 coordinates"
            )

        X_np = to_numpy(X_train).astype(np.float32)
        y_np = to_array(y_train).astype(np.int64)
        coords_np = np.asarray(coords, dtype=np.float32)

        # Store feature names for prediction-time alignment
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_: list[str] = list(X_train.columns)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np).astype(np.float32)

        # Combine train + val for the transductive graph
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        coords_val = kwargs.get("coords_val")

        has_val = (
            X_val is not None and y_val is not None and coords_val is not None and len(X_val) > 0
        )

        n_train = len(X_scaled)
        if has_val:
            X_val_np = self._scaler.transform(to_numpy(X_val).astype(np.float32)).astype(  # type: ignore[arg-type]
                np.float32
            )
            y_val_np = to_array(y_val).astype(np.int64)  # type: ignore[arg-type]
            coords_val_np = np.asarray(coords_val, dtype=np.float32)

            X_all = np.vstack([X_scaled, X_val_np])
            y_all = np.concatenate([y_np, y_val_np])
            coords_all = np.vstack([coords_np, coords_val_np])
            n_val = len(X_val_np)
        else:
            X_all = X_scaled
            y_all = y_np
            coords_all = coords_np
            n_val = 0

        # Build KNN graph
        self._data = _build_graph(X_all, y_all, coords_all, k=k)
        self._data = self._data.to(self._device)

        # Node masks
        n_total = len(X_all)
        train_mask = torch.zeros(n_total, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask = torch.zeros(n_total, dtype=torch.bool)
        if has_val:
            val_mask[n_train : n_train + n_val] = True

        train_mask = train_mask.to(self._device)
        val_mask = val_mask.to(self._device)

        # Determine number of classes
        n_classes = int(y_all.max()) + 1
        if n_classes < 2:
            n_classes = 2

        # Build network
        n_features = X_all.shape[1]
        if arch == "sage":
            self._net: _SAGENetwork | _GCNNetwork = _SAGENetwork(
                n_features, n_hidden, n_classes, dropout
            ).to(self._device)
        else:
            self._net = _GCNNetwork(n_features, n_hidden, n_classes, dropout).to(self._device)

        # Class weighting for imbalanced data
        n_pos = float((y_np == 1).sum())
        n_neg = float((y_np == 0).sum())
        weight = torch.tensor(
            [1.0, n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=self._device
        )
        criterion = nn.CrossEntropyLoss(weight=weight)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)

        # Training loop with early stopping
        best_val_loss: float | None = None
        patience_counter = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(epochs):
            self._net.train()
            optimizer.zero_grad()
            logits = self._net(self._data.x, self._data.edge_index)
            loss = criterion(logits[train_mask], self._data.y[train_mask])
            loss.backward()
            optimizer.step()

            # Early stopping on validation loss
            if has_val and val_mask.any():
                self._net.eval()
                with torch.no_grad():
                    val_logits = self._net(self._data.x, self._data.edge_index)
                    vl = criterion(val_logits[val_mask], self._data.y[val_mask]).item()

                if best_val_loss is None or vl < best_val_loss:
                    best_val_loss = vl
                    patience_counter = 0
                    best_state = copy.deepcopy(self._net.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

        # Restore best weights
        if best_state is not None:
            self._net.load_state_dict(best_state)

        # Mark as fitted (BaseModel uses _model for _check_fitted)
        self._model = self._net

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels (0/1).

        For nodes present in the training graph, runs a full GNN forward
        pass with message passing. For new test nodes, falls back to the
        learned linear transformations without graph structure (MLP mode).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        proba = self.predict_proba(X, **kwargs)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)`` -- probabilities for each class.
        """
        self._check_fitted()
        X_np = self._scaler.transform(to_numpy(X).astype(np.float32)).astype(np.float32)
        x = torch.from_numpy(X_np).to(self._device)

        self._net.eval()
        with torch.no_grad():
            # IMPORTANT — Inductive fallback (linear-only / MLP-equivalent):
            # Test nodes are NOT part of the training graph, so we cannot run
            # message passing at inference time.  Instead we extract the
            # learned linear transforms from each conv layer and apply them
            # sequentially — equivalent to a two-layer MLP.  This means GNN
            # test-set metrics reflect linear-transform performance only, NOT
            # full graph convolution.  Results should be reported as
            # "GCN (linear fallback)" / "GraphSAGE (linear fallback)" and
            # interpreted accordingly.
            # GCNConv stores its linear transform as .lin while SAGEConv
            # uses .lin_l (self-node) and .lin_r (neighbour).
            lin1 = getattr(self._net.conv1, "lin", None) or self._net.conv1.lin_l
            lin2 = getattr(self._net.conv2, "lin", None) or self._net.conv2.lin_l
            h = lin1(x)
            h = F.relu(h)
            h = F.dropout(h, p=self._dropout, training=False)
            logits = lin2(h)
            proba = tensor_to_numpy(F.softmax(logits, dim=1))

        return proba
