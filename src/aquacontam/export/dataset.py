"""Export integrated dataset as a self-contained directory.

Produces a directory suitable for upload to Zenodo or Figshare with
split Parquet files, feature files, metadata JSON, and a manifest.

Typical usage::

    from aquacontam.export.dataset import export_dataset

    export_dataset(
        data=wq_df,
        feature_dfs={"proximity": prox_df, "land_use": lu_df},
        output_dir=Path("release/aquacontam-dataset-v1.0"),
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aquacontam._constants import DATASET_VERSION, DOWNLOAD_CHUNK_SIZE
from aquacontam.preprocessing.splits import assign_epa_region, geographic_split

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    output_dir: Path,
    *,
    version: str = DATASET_VERSION,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write manifest.json with checksums and provenance.

    Parameters
    ----------
    output_dir : Path
        Dataset root directory.
    version : str
        Dataset version string.
    extra_metadata : dict, optional
        Additional key-value pairs to include.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    files: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            # POSIX separators so the manifest verifies on any platform.
            rel = path.relative_to(output_dir).as_posix()
            files[rel] = _sha256_file(path)

    manifest: dict[str, Any] = {
        "version": version,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "generator": "aquacontam.export.dataset",
        "n_files": len(files),
        "checksums": files,
    }
    if extra_metadata:
        manifest.update(extra_metadata)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest with %d file checksums", len(files))
    return manifest_path


def _write_split_assignments(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write split assignments CSV.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Split DataFrames with ``pwsid`` and ``epa_region`` columns.
    output_dir : Path
        Directory to write to.

    Returns
    -------
    Path
        Path to CSV file.
    """
    rows = []
    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        if df.empty:
            continue
        sub = df[["pwsid"]].copy()
        if "epa_region" in df.columns:
            sub["epa_region"] = df["epa_region"].to_numpy()
        else:
            sub["epa_region"] = pd.NA
        sub["split"] = split_name
        rows.append(sub)

    if not rows:
        assignments = pd.DataFrame(columns=["pwsid", "epa_region", "split"])
    else:
        assignments = pd.concat(rows, ignore_index=True)

    out_path = output_dir / "split_assignments.csv"
    assignments.to_csv(out_path, index=False)
    return out_path


def export_dataset(
    data: pd.DataFrame,
    feature_dfs: dict[str, pd.DataFrame] | None = None,
    output_dir: str | Path = "aquacontam-dataset",
    *,
    baseline_predictions: pd.DataFrame | None = None,
    exclude_sources: Sequence[str] | None = None,
    predictions_dir: str | Path | None = None,
    readme_text: str | None = None,
    version: str = DATASET_VERSION,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Export the integrated dataset for distribution.

    Parameters
    ----------
    data : pd.DataFrame
        Water quality DataFrame with standard schema columns including
        ``pwsid``.
    feature_dfs : dict[str, pd.DataFrame], optional
        Named feature DataFrames indexed by ``pwsid``.
    output_dir : str | Path
        Root output directory.
    baseline_predictions : pd.DataFrame, optional
        Baseline model predictions to include.
    exclude_sources : Sequence[str], optional
        ``source`` column values whose rows are dropped from the export
        (e.g. ``("mn_mdh",)`` while the agency's redistribution
        confirmation is pending). Ignored when ``data`` has no ``source``
        column.
    predictions_dir : str | Path, optional
        Existing per-system prediction export (Hive-partitioned Parquet
        tree) to copy into the archive's ``predictions/`` directory.
    readme_text : str, optional
        Markdown text written as the archive's top-level ``README.md``.
    version : str
        Dataset version string.
    extra_metadata : dict, optional
        Additional metadata for the manifest.

    Returns
    -------
    Path
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if feature_dfs is None:
        feature_dfs = {}

    # ---- 0. Source exclusions (redistribution terms) ----
    if exclude_sources and "source" in data.columns:
        before = len(data)
        data = data[~data["source"].isin(list(exclude_sources))]
        logger.info(
            "Excluded sources %s from the export: %d -> %d rows",
            list(exclude_sources),
            before,
            len(data),
        )

    # ---- 1. Geographic split ----
    df_with_region = assign_epa_region(data)

    # Need pwsid as column for geographic_split
    if "pwsid" in df_with_region.index.names:
        df_with_region = df_with_region.reset_index()

    train, val, test = geographic_split(df_with_region)

    # ---- 2. Write split Parquet files ----
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = data_dir / f"{name}.parquet"
        split_df.to_parquet(out, index=False)
        logger.info("Wrote %s: %d rows", out, len(split_df))

    # ---- 3. Write feature files ----
    if feature_dfs:
        feat_dir = output_dir / "features"
        feat_dir.mkdir(exist_ok=True)
        for feat_name, fdf in feature_dfs.items():
            out = feat_dir / f"{feat_name}.parquet"
            fdf.to_parquet(out, index=False)
            logger.info("Wrote feature file %s: %d rows, %d cols", out, len(fdf), len(fdf.columns))

    # ---- 4. Write metadata ----
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    _write_split_assignments(train, val, test, meta_dir)

    # ---- 5. Baseline predictions ----
    if baseline_predictions is not None:
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        out = pred_dir / "baselines.parquet"
        baseline_predictions.to_parquet(out, index=False)
        logger.info("Wrote baseline predictions: %d rows", len(baseline_predictions))

    # ---- 5b. Per-system prediction export (copied tree) ----
    if predictions_dir is not None:
        src = Path(predictions_dir)
        if src.is_dir():
            dst = output_dir / "predictions"
            shutil.copytree(src, dst, dirs_exist_ok=True)
            n_pred = sum(1 for p in dst.rglob("*") if p.is_file())
            logger.info("Copied per-system prediction export: %d files", n_pred)
        else:
            logger.warning("predictions_dir %s does not exist — skipping", src)

    # ---- 5c. Archive README ----
    if readme_text is not None:
        (output_dir / "README.md").write_text(readme_text, encoding="utf-8")

    # ---- 6. Manifest ----
    _write_manifest(output_dir, version=version, extra_metadata=extra_metadata)

    logger.info("Dataset export complete: %s", output_dir)
    return output_dir
