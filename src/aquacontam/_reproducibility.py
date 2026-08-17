"""Centralized random seed management for reproducibility."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for numpy, Python stdlib, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.

    Notes
    -----
    Hash randomization and cuBLAS determinism are governed by environment
    variables. ``PYTHONHASHSEED`` only takes effect in *child* processes when set
    here (the parent interpreter reads it at startup), but it makes spawned
    workers reproducible. ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first
    CUDA context is created for torch's deterministic GEMM path; we set it here
    and rely on ``set_seed`` being called early in each run. Both use
    ``setdefault`` so a deliberate value already present in the environment is
    respected.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  # global seed for reproducibility

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
