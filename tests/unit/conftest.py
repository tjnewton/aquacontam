"""Shared test fixtures for model unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def classification_data():
    """Synthetic binary classification dataset."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.random(100) > 0.7).astype(int), name="target")
    return X, y


@pytest.fixture()
def regression_data():
    """Synthetic regression dataset."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.random(100) * 10, name="target")
    return X, y
