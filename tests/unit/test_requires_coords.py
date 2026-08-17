"""Regression tests for the ``requires_coords`` model flag.

The benchmark task runner only applies coordinate alignment/subsetting to models
that declare ``requires_coords = True`` (GNNs). All other models must train on the
full split — a prior bug subset every model's training data to coordinate-covered
systems, which silently depressed metrics.
"""

from __future__ import annotations

import pytest

from aquacontam.models.base import BaseModel


def test_base_default_does_not_require_coords() -> None:
    assert BaseModel.requires_coords is False


def test_xgboost_does_not_require_coords() -> None:
    from aquacontam.models import XGBoostClassifier

    assert XGBoostClassifier().requires_coords is False


def test_gnn_requires_coords() -> None:
    pytest.importorskip("torch")  # GNN needs torch, not in CI's [test] extras
    from aquacontam.models.gnn import GNNClassifier

    assert GNNClassifier().requires_coords is True
