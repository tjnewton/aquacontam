"""Tests for GNN graph neural network models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

from aquacontam.models.gnn import GNNClassifier, _build_graph

# Small config for fast tests
FAST_CFG: dict = {
    "arch": "gcn",
    "n_hidden": 8,
    "epochs": 5,
    "k": 3,
    "random_state": 42,
}


class TestBuildGraph:
    """Tests for _build_graph helper."""

    def test_returns_data_object(self) -> None:
        from torch_geometric.data import Data

        rng = np.random.RandomState(42)
        X = rng.randn(10, 3).astype(np.float32)
        y = np.zeros(10, dtype=np.int64)
        coords = rng.randn(10, 2).astype(np.float32) * 100_000
        data = _build_graph(X, y, coords, k=3)
        assert isinstance(data, Data)
        assert data.x.shape == (10, 3)
        assert data.y.shape == (10,)
        assert data.edge_index.shape[0] == 2

    def test_k_clamped_for_small_graphs(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(3, 2).astype(np.float32)
        y = np.zeros(3, dtype=np.int64)
        coords = rng.randn(3, 2).astype(np.float32) * 100_000
        # k=10 but only 3 nodes => clamped to 2
        data = _build_graph(X, y, coords, k=10)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0


class TestGNNClassifier:
    """Tests for GNNClassifier."""

    def test_name_gcn(self) -> None:
        model = GNNClassifier(config={"arch": "gcn"})
        assert model.name == "gnn_gcn_classifier"

    def test_name_sage(self) -> None:
        model = GNNClassifier(config={"arch": "sage"})
        assert model.name == "gnn_sage_classifier"

    def test_name_default(self) -> None:
        model = GNNClassifier()
        assert model.name == "gnn_gcn_classifier"

    def test_fit_requires_coords(self) -> None:
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(10, 3), columns=["f0", "f1", "f2"])
        y = pd.Series(np.ones(10, dtype=int))
        model = GNNClassifier(config=FAST_CFG)
        with pytest.raises(ValueError, match="coords"):
            model.fit(X, y)

    def test_fit_predict_dataframe(self, classification_data) -> None:
        X, y = classification_data
        rng = np.random.RandomState(42)
        coords = rng.randn(len(X), 2).astype(np.float32) * 100_000
        model = GNNClassifier(config=FAST_CFG)
        model.fit(X, y, coords=coords)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_fit_predict_numpy(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5).astype(np.float32)
        y = (rng.random(20) > 0.5).astype(int)
        coords = rng.randn(20, 2) * 100_000
        model = GNNClassifier(config=FAST_CFG)
        model.fit(X, y, coords=coords)
        preds = model.predict(X)
        assert len(preds) == 20
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, classification_data) -> None:
        X, y = classification_data
        rng = np.random.RandomState(42)
        coords = rng.randn(len(X), 2).astype(np.float32) * 100_000
        model = GNNClassifier(config=FAST_CFG)
        model.fit(X, y, coords=coords)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_predict_proba_between_0_and_1(self, classification_data) -> None:
        X, y = classification_data
        rng = np.random.RandomState(42)
        coords = rng.randn(len(X), 2).astype(np.float32) * 100_000
        model = GNNClassifier(config=FAST_CFG)
        model.fit(X, y, coords=coords)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_sage_fit_predict(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5).astype(np.float32)
        y = (rng.random(20) > 0.5).astype(int)
        coords = rng.randn(20, 2) * 100_000
        cfg = {**FAST_CFG, "arch": "sage"}
        model = GNNClassifier(config=cfg)
        model.fit(X, y, coords=coords)
        preds = model.predict(X)
        assert len(preds) == 20
        assert set(np.unique(preds)).issubset({0, 1})

    def test_sage_predict_proba(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5).astype(np.float32)
        y = (rng.random(20) > 0.5).astype(int)
        coords = rng.randn(20, 2) * 100_000
        cfg = {**FAST_CFG, "arch": "sage"}
        model = GNNClassifier(config=cfg)
        model.fit(X, y, coords=coords)
        proba = model.predict_proba(X)
        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_early_stopping_with_val(self, classification_data) -> None:
        X, y = classification_data
        rng = np.random.RandomState(42)
        coords = rng.randn(len(X), 2).astype(np.float32) * 100_000
        # Split into train/val
        X_train, X_val = X.iloc[:80], X.iloc[80:]
        y_train, y_val = y.iloc[:80], y.iloc[80:]
        coords_train, coords_val = coords[:80], coords[80:]
        cfg = {**FAST_CFG, "epochs": 50, "patience": 2}
        model = GNNClassifier(config=cfg)
        model.fit(
            X_train,
            y_train,
            coords=coords_train,
            X_val=X_val,
            y_val=y_val,
            coords_val=coords_val,
        )
        preds = model.predict(X_val)
        assert len(preds) == 20

    def test_predict_unfitted_raises(self) -> None:
        model = GNNClassifier(config=FAST_CFG)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_predict_proba_unfitted_raises(self) -> None:
        model = GNNClassifier(config=FAST_CFG)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_predict_new_data(self, classification_data) -> None:
        """Test prediction on data not seen during training (inductive fallback)."""
        X, y = classification_data
        rng = np.random.RandomState(42)
        coords = rng.randn(len(X), 2).astype(np.float32) * 100_000
        model = GNNClassifier(config=FAST_CFG)
        model.fit(X, y, coords=coords)
        # Predict on completely new data
        X_new = pd.DataFrame(rng.randn(15, 5), columns=[f"f{i}" for i in range(5)])
        preds = model.predict(X_new)
        assert len(preds) == 15
        assert set(np.unique(preds)).issubset({0, 1})

    def test_all_positive_class(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3)
        y = np.ones(20, dtype=int)
        coords = rng.randn(20, 2) * 100_000
        model = GNNClassifier(config=FAST_CFG)
        model.fit(X, y, coords=coords)
        preds = model.predict(X)
        assert len(preds) == 20

    def test_config_defaults(self) -> None:
        model = GNNClassifier()
        assert model.config == {}
        assert model.name == "gnn_gcn_classifier"
