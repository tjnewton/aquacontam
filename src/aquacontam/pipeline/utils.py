"""Pipeline utility functions: cache cleanup, versioning, and result summaries."""

from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def clean_caches(data_dir: Path) -> None:
    """Delete intermediate caches so the pipeline rebuilds from raw data.

    Removes ``data/interim/merged_wq.parquet`` and all feature caches in
    ``data/interim/features/``. Does NOT touch raw or processed data.
    """
    merged = data_dir / "interim" / "merged_wq.parquet"
    if merged.exists():
        merged.unlink()
        logger.info("Deleted cached merged WQ: %s", merged)

    features_dir = data_dir / "interim" / "features"
    if features_dir.exists():
        for p in features_dir.glob("*.parquet"):
            p.unlink()
            logger.info("Deleted cached feature: %s", p)


def collect_software_versions() -> dict[str, str]:
    """Collect versions of key packages used in the pipeline.

    Returns a dict mapping package name to version string.
    """
    from aquacontam.pipeline._strict import is_strict

    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in (
        "aquacontam",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "torch",
        "geopandas",
        "scipy",
        "rasterio",
        "shapely",
        "lightgbm",
        "catboost",
        "lifelines",
        "shap",
        "tabpfn",
        "torch_geometric",
        "optuna",
    ):
        try:
            from importlib.metadata import version

            versions[pkg] = version(pkg)
        except (ImportError, ModuleNotFoundError):
            if is_strict():
                raise
    return versions


def save_software_versions(output_dir: Path) -> dict[str, str]:
    """Collect and save software versions to output directory."""
    versions = collect_software_versions()
    versions_path = output_dir / "software_versions.json"
    versions_path.parent.mkdir(parents=True, exist_ok=True)
    versions_path.write_text(json.dumps(versions, indent=2))
    logger.info("Software versions saved to %s", versions_path)
    return versions


def print_results_summary(results: list[dict[str, Any]]) -> None:
    """Print a compact summary table of benchmark results."""
    import click

    if not results:
        click.echo("\nNo results to summarize.")
        return

    click.echo("\n" + "=" * 60)
    click.echo("RESULTS SUMMARY")
    click.echo("=" * 60)
    click.echo(f"{'Task':<6} {'Model':<20} {'Primary Metric':<15} {'Value':>8}")
    click.echo("-" * 60)

    for r in results:
        task = r["task"]
        model = r["model"]
        metrics = r.get("metrics", {})
        # Find primary metric value (first metric reported)
        if metrics:
            metric_name = next(iter(metrics))
            metric_val = metrics[metric_name]
            click.echo(f"{task:<6} {model:<20} {metric_name:<15} {metric_val:>8.4f}")
        else:
            click.echo(f"{task:<6} {model:<20} {'(no metrics)':<15} {'N/A':>8}")

    click.echo("=" * 60)


def _grid_size(param_grid: dict[str, list[Any]]) -> int:
    """Compute total number of combinations in a parameter grid."""
    import itertools

    if not param_grid:
        return 0
    return len(list(itertools.product(*param_grid.values())))


def _is_nan(x: Any) -> bool:
    """Check if a value is NaN (works with floats)."""
    try:
        import math

        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False
