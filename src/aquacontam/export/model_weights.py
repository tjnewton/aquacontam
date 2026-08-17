"""Export and load pretrained model weights for Zenodo release.

Saves fitted sklearn models as joblib files alongside their metadata
(feature names, training config, performance metrics). Includes SHA-256
integrity verification on save and load.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def export_model_weights(
    model: BaseModel,
    task_name: str,
    output_dir: Path,
    *,
    metrics: dict[str, float] | None = None,
    feature_names: list[str] | None = None,
) -> Path:
    """Export a fitted model to a joblib file with metadata.

    Parameters
    ----------
    model : BaseModel
        Fitted model to export.
    task_name : str
        Task identifier (e.g., ``"T1"``).
    output_dir : Path
        Directory to save the model.
    metrics : dict[str, float] | None
        Test-set metrics to record in metadata.
    feature_names : list[str] | None
        Feature names used during training.

    Returns
    -------
    Path
        Path to saved model file.
    """
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model.name.replace(" ", "_").lower()
    filename = f"{task_name}_{model_name}.joblib"
    model_path = output_dir / filename

    # Save the underlying sklearn model and any auxiliary objects (e.g. scaler)
    payload: dict[str, Any] = {"model": model._model}
    if hasattr(model, "_scaler") and model._scaler is not None:
        payload["scaler"] = model._scaler
    joblib.dump(payload, model_path)

    # Compute integrity hash of the saved model file
    sha256 = _compute_file_sha256(model_path)

    # Save metadata (including SHA-256 for integrity verification on load)
    metadata: dict[str, Any] = {
        "task": task_name,
        "model_name": model.name,
        "config": model.config,
        "sha256": sha256,
    }
    if metrics:
        metadata["test_metrics"] = metrics
    if feature_names:
        metadata["feature_names"] = feature_names
    elif hasattr(model, "feature_names_in_"):
        metadata["feature_names"] = model.feature_names_in_

    meta_path = output_dir / f"{task_name}_{model_name}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    logger.info("Exported %s/%s to %s (sha256=%s)", task_name, model.name, model_path, sha256[:16])
    return model_path


def load_model_weights(
    model_cls: type[BaseModel],
    model_path: Path,
    config: dict[str, Any] | None = None,
) -> BaseModel:
    """Load a pretrained model from a joblib file.

    Verifies the file's SHA-256 hash against the metadata before loading
    when a hash is available. Legacy model files without hashes are loaded
    with a warning.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Model class to instantiate.
    model_path : Path
        Path to the joblib file.
    config : dict[str, Any] | None
        Config to pass to the model constructor.

    Returns
    -------
    BaseModel
        Model with loaded weights.

    Raises
    ------
    ValueError
        If integrity check fails in strict mode.
    """
    import joblib

    # Verify integrity BEFORE loading (joblib.load can execute arbitrary code)
    meta_path = model_path.with_name(model_path.stem + "_metadata.json")
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        expected_hash = metadata.get("sha256")
        if expected_hash:
            actual_hash = _compute_file_sha256(model_path)
            if actual_hash != expected_hash:
                msg = (
                    f"Model integrity check FAILED for {model_path.name}: "
                    f"expected {expected_hash[:16]}…, got {actual_hash[:16]}…"
                )
                logger.warning(msg)
                try:
                    from aquacontam.pipeline._strict import is_strict

                    if is_strict():
                        raise ValueError(msg)
                except ImportError:
                    pass
            else:
                logger.debug("Integrity check OK for %s", model_path.name)
        else:
            logger.debug(
                "No sha256 in metadata for %s — skipping integrity check", model_path.name
            )

    model = model_cls(config=config)
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        model._model = payload["model"]
        if "scaler" in payload and hasattr(model, "_scaler"):
            model._scaler = payload["scaler"]
    else:
        # Backwards compatibility: legacy exports saved the raw sklearn model
        model._model = payload

    # Apply metadata
    if "feature_names" in metadata:
        model.feature_names_in_ = metadata["feature_names"]  # type: ignore[attr-defined]

    logger.info("Loaded model from %s", model_path)
    return model


def verify_model_integrity(model_dir: Path) -> dict[str, bool]:
    """Verify all ``.joblib`` files in a directory against their metadata hashes.

    Parameters
    ----------
    model_dir : Path
        Directory containing ``.joblib`` model files and their
        ``_metadata.json`` companions.

    Returns
    -------
    dict[str, bool]
        Mapping of model filename to verification result (``True`` = OK,
        ``False`` = mismatch or missing hash).
    """
    results: dict[str, bool] = {}
    for model_path in sorted(model_dir.glob("*.joblib")):
        meta_path = model_path.with_name(model_path.stem + "_metadata.json")
        if not meta_path.exists():
            logger.warning("No metadata for %s — cannot verify", model_path.name)
            results[model_path.name] = False
            continue

        metadata = json.loads(meta_path.read_text())
        expected_hash = metadata.get("sha256")
        if not expected_hash:
            logger.warning("No sha256 in metadata for %s", model_path.name)
            results[model_path.name] = False
            continue

        actual_hash = _compute_file_sha256(model_path)
        ok = actual_hash == expected_hash
        if ok:
            logger.info("OK: %s", model_path.name)
        else:
            logger.error(
                "MISMATCH: %s (expected %s, got %s)",
                model_path.name,
                expected_hash[:16],
                actual_hash[:16],
            )
        results[model_path.name] = ok

    return results
