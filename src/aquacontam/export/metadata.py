"""Dataset metadata generation for feature documentation and dataset cards.

Provides structured metadata about features, analytes, and the dataset
itself for inclusion in exported dataset packages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from aquacontam._constants import (
    CLASSIFICATION_METRICS,
    DATASET_VERSION,
    HEAVY_METAL_ANALYTES,
    REGRESSION_METRICS,
    T1_DEFAULT_ANALYTES,
    T2_DEFAULT_ANALYTES,
    T4_DEFAULT_ANALYTES,
    T6_DEFAULT_ANALYTES,
    UCMR3_ANALYTES,
    UCMR5_ANALYTES,
)

logger = logging.getLogger(__name__)


def generate_feature_metadata(
    feature_dfs: dict[str, pd.DataFrame],
    *,
    descriptions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Generate per-column metadata for feature DataFrames.

    Parameters
    ----------
    feature_dfs : dict[str, pd.DataFrame]
        Named feature DataFrames.
    descriptions : dict[str, str], optional
        Manual column descriptions keyed by column name.

    Returns
    -------
    list[dict[str, Any]]
        List of dicts with keys: ``name``, ``dtype``, ``source``,
        ``n_non_null``, ``description``.
    """
    if descriptions is None:
        descriptions = {}

    columns: list[dict[str, Any]] = []
    for source_name, df in feature_dfs.items():
        for col in df.columns:
            col_info: dict[str, Any] = {
                "name": col,
                "dtype": str(df[col].dtype),
                "source": source_name,
                "n_non_null": int(df[col].notna().sum()),
                "n_total": len(df),
                "description": descriptions.get(col, ""),
            }
            columns.append(col_info)

    return columns


def _analyte_info() -> list[dict[str, str | None]]:
    """Build analyte information list from constants."""
    analytes: list[dict[str, str | None]] = []
    seen: set[str] = set()

    for name in UCMR5_ANALYTES:
        if name not in seen:
            analytes.append(
                {
                    "name": name,
                    "source": "UCMR5",
                    "type": "non-PFAS" if name == "lithium" else "PFAS",
                }
            )
            seen.add(name)

    for name in UCMR3_ANALYTES:
        if name not in seen:
            analytes.append({"name": name, "source": "UCMR3", "type": "PFAS"})
            seen.add(name)

    for name in HEAVY_METAL_ANALYTES:
        if name not in seen:
            analytes.append({"name": name, "source": "SDWIS", "type": "heavy_metal"})
            seen.add(name)

    return analytes


def generate_dataset_card(
    *,
    version: str = DATASET_VERSION,
    n_systems: int | None = None,
    n_samples: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a dataset card (description, citation, tasks, etc.).

    Parameters
    ----------
    version : str
        Dataset version.
    n_systems : int, optional
        Number of water systems in the dataset.
    n_samples : int, optional
        Total number of samples.
    extra : dict, optional
        Additional metadata to include.

    Returns
    -------
    dict[str, Any]
        Dataset card as a nested dictionary.
    """
    card: dict[str, Any] = {
        "name": "AquaContam Benchmark Dataset",
        "version": version,
        "description": (
            "ML-ready benchmark for predicting PFAS and heavy metal "
            "contamination in U.S. public drinking water systems."
        ),
        "license": (
            "CC BY 4.0 where source terms permit (compiled data); code: Apache License 2.0"
        ),
        "citation": (
            "Newton, T. J. (2026). AquaContam: machine-learning models of "
            "drinking-water contamination learn who is monitored as much as "
            "where contamination occurs. Nature Water."
        ),
        "homepage": "https://github.com/tjnewton/aquacontam",
        "tasks": {
            "T1": {
                "name": "PFAS Binary Detection",
                "type": "classification",
                "primary_metric": "auprc",
                "analytes": list(T1_DEFAULT_ANALYTES),
            },
            "T2": {
                "name": "PFAS Concentration Regression",
                "type": "regression",
                "primary_metric": "rmse",
                "analytes": list(T2_DEFAULT_ANALYTES),
            },
            "T4": {
                "name": "Heavy Metal Prediction",
                "type": "classification",
                "primary_metric": "auprc",
                "analytes": list(T4_DEFAULT_ANALYTES),
            },
            "T5": {
                "name": "Cross-Contaminant Transfer",
                "type": "classification",
                "primary_metric": "auprc",
            },
            "T6": {
                "name": "Private Well Risk Extrapolation",
                "type": "classification",
                "primary_metric": "auprc",
                "analytes": list(T6_DEFAULT_ANALYTES),
            },
        },
        "metrics": {
            "classification": list(CLASSIFICATION_METRICS),
            "regression": list(REGRESSION_METRICS),
        },
        "analytes": _analyte_info(),
    }

    if n_systems is not None:
        card["n_systems"] = n_systems
    if n_samples is not None:
        card["n_samples"] = n_samples
    if extra:
        card.update(extra)

    return card


def write_metadata_files(
    output_dir: str | Path,
    feature_dfs: dict[str, pd.DataFrame] | None = None,
    *,
    version: str = DATASET_VERSION,
    n_systems: int | None = None,
    n_samples: int | None = None,
    descriptions: dict[str, str] | None = None,
) -> list[Path]:
    """Write all metadata files to the output directory.

    Parameters
    ----------
    output_dir : str | Path
        Metadata directory.
    feature_dfs : dict[str, pd.DataFrame], optional
        Feature DataFrames for column metadata.
    version : str
        Dataset version.
    n_systems : int, optional
        Number of water systems.
    n_samples : int, optional
        Total samples.
    descriptions : dict[str, str], optional
        Manual column descriptions.

    Returns
    -------
    list[Path]
        Paths to written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Feature columns
    if feature_dfs:
        feat_meta = generate_feature_metadata(feature_dfs, descriptions=descriptions)
        path = output_dir / "feature_columns.json"
        with open(path, "w") as f:
            json.dump(feat_meta, f, indent=2)
        written.append(path)

    # Analyte info
    path = output_dir / "analyte_info.json"
    with open(path, "w") as f:
        json.dump(_analyte_info(), f, indent=2)
    written.append(path)

    # Dataset card
    card = generate_dataset_card(version=version, n_systems=n_systems, n_samples=n_samples)
    path = output_dir / "dataset_card.json"
    with open(path, "w") as f:
        json.dump(card, f, indent=2)
    written.append(path)

    logger.info("Wrote %d metadata files to %s", len(written), output_dir)
    return written
