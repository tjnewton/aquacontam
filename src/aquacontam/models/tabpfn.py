"""TabPFN v2 foundation model wrapper for tabular classification.

TabPFN v2 (Hollmann et al., Nature 2025) is a pre-trained transformer that
performs in-context learning on tabular data. It conditions on the entire
training set at inference time — no gradient-based training is needed.

Optimized for datasets with ≤10,000 samples, making it ideal for
environmental monitoring benchmark tasks (T1 ~7,400 train systems).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)

# Maximum training set size TabPFN can handle efficiently
_MAX_TRAIN_SIZE = 10_000

# CPU-practical subsample limit. TabPFN inference scales O(n_train * n_test)
# so >3000 train samples on CPU becomes impractical (>30 min per task).
_CPU_SUBSAMPLE_SIZE = 3_000

# Default checkpoint that can be downloaded without HuggingFace gated-repo auth.
# Falls back to the S3 mirror automatically. The explicit v2 pin matters for
# licensing as well as reproducibility: newer ``tabpfn`` releases default to
# TabPFN-3 weights (non-commercial license), while the v2 weights carry the
# Prior Labs License (Apache-2.0-derived, no commercial restriction).
_DEFAULT_MODEL_FILENAME = "tabpfn-v2-classifier-v2_default.ckpt"


def _default_model_path() -> str:
    """Resolve the pinned v2 checkpoint path independent of the working directory.

    Prefers an existing checkpoint in the current directory (back-compat with
    repo-root runs), otherwise anchors to a stable per-user cache directory so
    the upstream downloader writes somewhere deterministic.
    """
    cwd_candidate = Path.cwd() / _DEFAULT_MODEL_FILENAME
    if cwd_candidate.exists():
        return str(cwd_candidate)
    cache_dir = Path(
        os.environ.get("AQUACONTAM_CACHE_DIR", str(Path.home() / ".cache" / "aquacontam"))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / _DEFAULT_MODEL_FILENAME)


def _stratified_subsample(y: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Stratified subsample preserving class ratio."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    indices: list[np.ndarray] = []
    for cls, count in zip(classes, counts, strict=True):
        cls_idx = np.where(y == cls)[0]
        n_cls = max(1, round(n * count / total))
        n_cls = min(n_cls, len(cls_idx))
        indices.append(rng.choice(cls_idx, size=n_cls, replace=False))
    return np.concatenate(indices)


def _to_numpy(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert feature input to float64 numpy array (torch-free)."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


class TabPFNClassifier(BaseModel):
    """TabPFN v2 foundation model classifier.

    Wraps the pre-trained TabPFN model for binary classification.
    ``fit()`` stores the training data (no gradient updates);
    ``predict_proba()`` runs in-context learning at inference time.

    Parameters
    ----------
    config : dict, optional
        Configuration:
        - ``max_train_size``: skip if training set exceeds this (default 10000)
        - ``random_state``: seed (default ``42``)
        - ``device``: ``"cpu"`` or ``"cuda"`` (default ``"cpu"``)
        - ``n_estimators``: number of ensemble estimators (default ``4``)
    """

    @property
    def name(self) -> str:
        return "tabpfn_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Store training data for in-context learning.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training labels (0/1).
        **kwargs
            Ignored (TabPFN does not train).

        Raises
        ------
        ValueError
            If training set exceeds ``max_train_size``.
        """
        from tabpfn import TabPFNClassifier as _TabPFN

        cfg = dict(self.config)
        max_size = cfg.get("max_train_size", _MAX_TRAIN_SIZE)
        seed = cfg.get("random_state", 42)
        device = cfg.get("device", "cpu")
        n_estimators = cfg.get("n_estimators", cfg.get("n_ensemble_configurations", 4))

        X_np = (
            X_train.to_numpy().astype(np.float64)
            if isinstance(X_train, pd.DataFrame)
            else np.asarray(X_train, dtype=np.float64)
        )
        y_np = (
            y_train.to_numpy() if isinstance(y_train, pd.Series) else np.asarray(y_train)
        ).astype(int)

        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_: list[str] = list(X_train.columns)

        if len(X_np) > max_size:
            raise ValueError(
                f"TabPFN training set ({len(X_np)}) exceeds max_train_size ({max_size}). "
                f"TabPFN is optimized for ≤{max_size} samples."
            )

        # Subsample on CPU to keep inference tractable
        cpu_limit = cfg.get("cpu_subsample_size", _CPU_SUBSAMPLE_SIZE)
        if device == "cpu" and len(X_np) > cpu_limit:
            rng = np.random.RandomState(seed)
            idx = _stratified_subsample(y_np, cpu_limit, rng)
            logger.info(
                "TabPFN CPU subsampling: %d -> %d (stratified, preserving class ratio)",
                len(X_np),
                len(idx),
            )
            X_np = X_np[idx]
            y_np = y_np[idx]

        model_path = cfg.get("model_path") or _default_model_path()
        fit_mode = cfg.get("fit_mode", "low_memory")
        self._model = _TabPFN(
            device=device,
            random_state=seed,
            n_estimators=n_estimators,
            model_path=model_path,
            ignore_pretraining_limits=True,
            fit_mode=fit_mode,
        )
        self._model.fit(X_np, y_np)
        logger.info("TabPFN fitted with %d samples, %d features", len(X_np), X_np.shape[1])

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels (0/1)."""
        self._check_fitted()
        X_np = _to_numpy(X)
        return np.asarray(self._model.predict(X_np))

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)`` — probabilities for each class.
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        return np.asarray(self._model.predict_proba(X_np))
