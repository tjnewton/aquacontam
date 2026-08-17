#!/usr/bin/env python
"""AquaContam full reproducibility pipeline.

Thin CLI wrapper that delegates to ``aquacontam.pipeline`` modules.

Usage::

    python scripts/reproduce.py --all
    python scripts/reproduce.py --download-only
    python scripts/reproduce.py --train-only --tasks T1,T3 --models xgboost,random_forest
"""

from __future__ import annotations

# GPU env-var setup MUST happen before any heavy import (torch, xgboost, lightgbm,
# catboost, tabpfn) so CUDA_VISIBLE_DEVICES is honored. Click parses argv after
# imports finish, so we sniff argv ourselves here.
import os as _os
import sys as _sys


def _early_gpu_env_setup() -> None:
    argv = _sys.argv
    gpu_id: str | None = None
    for i, tok in enumerate(argv):
        if tok == "--gpu-id" and i + 1 < len(argv):
            gpu_id = argv[i + 1]
            break
        if tok.startswith("--gpu-id="):
            gpu_id = tok.split("=", 1)[1]
            break
    if gpu_id is not None:
        # On Windows, torch defaults to FASTEST_FIRST ordering; force PCI bus
        # order so torch indices match nvidia-smi.
        _os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        _os.environ["AQUACONTAM_GPU_ID"] = "0"  # always 0 after CVD remap


_early_gpu_env_setup()

# Pre-import pandas + pyarrow before any aquacontam module pulls torch.
# Why: torch 2.5.1+cu124 on Windows ships CUDA DLLs that, when loaded
# first, trigger an access violation in pyarrow.dataset on parquet read.
# Pre-importing ensures pyarrow's runtime is bound before torch's.
import json
import logging
from pathlib import Path
from typing import Any

import click
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.dataset as _pa_ds  # noqa: F401
import pyarrow.parquet as _pa_pq  # noqa: F401

from aquacontam.features.assembly import PROVENANCE_FREE_EXCLUDE
from aquacontam.pipeline._strict import is_strict, set_strict
from aquacontam.pipeline.analysis import (
    _run_bootstrap_cis,
    _run_buffered_loro,
    _run_calibration_analysis,
    _run_causal_deconfounding,
    _run_cnn1d_feature_ordering,
    _run_conformal_analysis,
    _run_coordinate_sensitivity,
    _run_dedup_comparison,
    _run_detection_only_ablation,
    _run_equity_analysis,
    _run_external_validation,
    _run_fair_tuning,
    _run_feature_ablation,
    _run_group_conformal_analysis,
    _run_hyperparam_sensitivity,
    _run_icp_recoverability,
    _run_lift_analysis,
    _run_loro_cv,
    _run_loro_equity_analysis,
    _run_mcl_exceedance_analysis,
    _run_model_comparison,
    _run_monitoring_inequity,
    _run_multi_seed_analysis,
    _run_optuna_tuning,
    _run_power_analysis,
    _run_region_heterogeneity,
    _run_seed_inflation_check,
    _run_sensitivity_analysis,
    _run_sensitivity_bounds,
    _run_shap_analysis,
    _run_shap_interaction_analysis,
    _run_spatial_autocorrelation,
    _run_spatial_block_bootstrap,
    _run_split_comparison,
    _run_transfer_ablation,
    _run_val_reuse_bias,
)
from aquacontam.pipeline.assembly import (  # noqa: F401
    assemble_with_split_imputation as _assemble_with_split_imputation,
)
from aquacontam.pipeline.checksum import (
    generate_results_checksums as _generate_results_checksums,
)
from aquacontam.pipeline.checksum import (
    verify_results_checksums as _verify_results_checksums,
)
from aquacontam.pipeline.download import download_data as _download_data
from aquacontam.pipeline.export import (
    export_feature_importances as _export_feature_importances,
)
from aquacontam.pipeline.export import (
    export_icp_diagnostics as _export_icp_diagnostics,
)
from aquacontam.pipeline.export import (
    export_results as _export_results,
)
from aquacontam.pipeline.export import (
    export_zenodo_dataset as _export_zenodo_dataset,
)
from aquacontam.pipeline.export import (
    generate_paper_assets as _generate_paper_assets,
)
from aquacontam.pipeline.export import (
    generate_webapp_predictions as _generate_webapp_predictions,
)
from aquacontam.pipeline.features import extract_features as _extract_features
from aquacontam.pipeline.models import (  # noqa: F401
    get_model_instances as _get_model_instances,
)
from aquacontam.pipeline.preprocess import preprocess_data as _preprocess_data
from aquacontam.pipeline.training import (  # noqa: F401
    build_task_kwargs as _build_task_kwargs,
)
from aquacontam.pipeline.training import (
    train_and_evaluate as _train_and_evaluate,
)
from aquacontam.pipeline.utils import (  # noqa: F401
    _grid_size,
    _is_nan,
)
from aquacontam.pipeline.utils import (
    clean_caches as _clean_caches,
)
from aquacontam.pipeline.utils import (
    print_results_summary as _print_results_summary,
)
from aquacontam.pipeline.utils import (
    save_software_versions as _save_software_versions,
)

logger = logging.getLogger(__name__)

# Keep the legacy global for any third-party code that reads it directly.
_STRICT: bool = False


def _ensure_data(
    wq_df: Any,
    feature_dfs: list[Any],
    data_path: Path,
) -> tuple[Any, list[Any]]:
    """Load cached WQ data and feature DataFrames if not already loaded."""
    import pandas as pd

    if wq_df is None:
        cached = data_path / "interim" / "merged_wq.parquet"
        if cached.exists():
            wq_df = pd.read_parquet(cached)
            click.echo(f"Loaded cached WQ data: {len(wq_df)} rows")
        else:
            click.echo("No preprocessed data found. Run --preprocess-only first.")

    if not feature_dfs:
        features_dir = data_path / "interim" / "features"
        if features_dir.exists():
            for p in sorted(features_dir.glob("*.parquet")):
                import pandas as pd

                feature_dfs.append(pd.read_parquet(p))
            click.echo(f"Loaded {len(feature_dfs)} cached feature sets")

    return wq_df, feature_dfs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--all", "run_all", is_flag=True, help="Run the full pipeline.")
@click.option("--download-only", is_flag=True, help="Only download data.")
@click.option("--preprocess-only", is_flag=True, help="Only preprocess data.")
@click.option("--train-only", is_flag=True, help="Only train and evaluate.")
@click.option("--evaluate-only", is_flag=True, help="Only evaluate (alias for train-only).")
@click.option("--skip-large", is_flag=True, help="Skip large files (NLCD raster).")
@click.option(
    "--clean", is_flag=True, help="Delete interim caches before running (merged WQ + features)."
)
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option(
    "--random-split",
    is_flag=True,
    help="Use random (non-geographic) splits for ablation comparison.",
)
@click.option("--tasks", default=None, help="Comma-separated task names (e.g. T1,T3).")
@click.option(
    "--models", default=None, help="Comma-separated model names (e.g. xgboost,random_forest)."
)
@click.option("--data-dir", default="data", help="Root data directory.")
@click.option("--output-dir", default="results", help="Results output directory.")
@click.option("--paper", is_flag=True, help="Generate paper figures and tables.")
@click.option("--export-dataset", "export_ds", is_flag=True, help="Export Zenodo dataset.")
@click.option(
    "--bootstrap", is_flag=True, help="Compute bootstrap CIs for all classification results."
)
@click.option("--ablation", is_flag=True, help="Run feature category ablation study on T1 and T4.")
@click.option(
    "--fair-tuning",
    is_flag=True,
    help="Run symmetric HPO grid search for all model families on T1.",
)
@click.option("--sensitivity", is_flag=True, help="Run sensitivity analysis.")
@click.option("--multi-seed", is_flag=True, help="Run multi-seed stability analysis on T1 and T4.")
@click.option(
    "--region-heterogeneity",
    is_flag=True,
    help="Run per-region heterogeneity analysis on T1.",
)
@click.option(
    "--feature-ordering",
    is_flag=True,
    help="Run CNN1D feature ordering sensitivity analysis.",
)
@click.option(
    "--optuna",
    "optuna_hpo",
    is_flag=True,
    help="Run Optuna Bayesian HPO for all model families on T1.",
)
@click.option(
    "--no-monitoring-features",
    is_flag=True,
    help="Exclude monitoring intensity features (n_samples, mean_detection_limit) from models.",
)
@click.option(
    "--provenance-free",
    is_flag=True,
    help=(
        "Exclude data-source PROVENANCE features (monitoring intensity, system size, "
        "and all *_nan one-hot missingness indicators) for the honest environmental-signal "
        "model. Keeps hydrology (source_water_type GW/SW) and system_type. Combine with "
        "--output-dir results_provenance_free to keep this variant separate."
    ),
)
@click.option("--mcl-exceedance", is_flag=True, help="Run MCL exceedance comparison (S18).")
@click.option(
    "--val-reuse-bias",
    is_flag=True,
    help="Quantify validation set reuse bias in holdout-LORO gap.",
)
@click.option("--power-analysis", is_flag=True, help="Run expanded power analysis (S19).")
@click.option(
    "--sensitivity-bounds", is_flag=True, help="Run Rosenbaum/E-value sensitivity bounds (S20)."
)
@click.option(
    "--detection-only-ablation",
    is_flag=True,
    help="Run detection-only source ablation (S21).",
)
@click.option(
    "--t6-arsenic",
    is_flag=True,
    help="Run T6 arsenic transfer (public-supply to domestic, USGS NGA); writes t6_arsenic.json.",
)
@click.option("--shap-interactions", is_flag=True, help="Run SHAP interaction analysis (S22).")
@click.option(
    "--shap",
    "shap_analysis",
    is_flag=True,
    help=(
        "Run SHAP analysis standalone (writes shap_T1.json/shap_T4.json). "
        "Refits the T1/T4 XGBoost reference model when no fitted models are in "
        "memory; honors --provenance-free / --no-monitoring-features."
    ),
)
@click.option("--seed-inflation", is_flag=True, help="Run multi-seed inflation check.")
@click.option(
    "--split-comparison",
    "split_comparison",
    is_flag=True,
    help="Run split strategy comparison standalone (writes split_comparison.json).",
)
@click.option(
    "--split-size-matched",
    "split_size_matched",
    is_flag=True,
    help=(
        "Add a random_sizematched arm to --split-comparison: random CV with the "
        "training set subsampled to the geographic training-set size."
    ),
)
@click.option(
    "--spatial-block-bootstrap",
    "spatial_block_bootstrap",
    is_flag=True,
    help="Region-block bootstrap of the LORO headline CI (reads loro_cv.json).",
)
@click.option(
    "--buffered-loro",
    "buffered_loro",
    is_flag=True,
    help="Buffered-boundary LORO for T1 (drops training systems near the test region).",
)
@click.option(
    "--icp-recoverability",
    "icp_recoverability",
    is_flag=True,
    help="Probe monitoring-source recoverability from the ICP representation.",
)
@click.option("--loro-cv", "loro_cv", is_flag=True, help="Run LORO cross-validation on T1 and T4.")
@click.option("--loro-equity", "loro_equity", is_flag=True, help="Run LORO regional EJ analysis.")
@click.option(
    "--with-uncertainty",
    is_flag=True,
    help="Enable LORO CV, LORO equity, and bootstrap CIs (included in --all).",
)
@click.option(
    "--strict", is_flag=True, help="Re-raise exceptions instead of logging and continuing."
)
@click.option(
    "--verify", is_flag=True, help="Verify results checksums and exit (no pipeline execution)."
)
@click.option(
    "--gpu-id",
    type=int,
    default=None,
    help=(
        "CUDA GPU index in PCI bus order (matches nvidia-smi). "
        "Sets CUDA_VISIBLE_DEVICES + CUDA_DEVICE_ORDER=PCI_BUS_ID early, before "
        "torch/xgboost/lightgbm/catboost imports. Use --gpu-id 2 to select the RTX 6000."
    ),
)
@click.option(
    "--n-trials",
    type=int,
    default=None,
    help="Override default per-model trial count for --optuna (default: 300).",
)
@click.option(
    "--hpo-tasks",
    default=None,
    help="Comma-sep tasks to tune in --optuna (default: T1,T2,T3,T4,T5,T6,T7).",
)
@click.option(
    "--hpo-models",
    default=None,
    help="Comma-sep model registry names to restrict --optuna tuning.",
)
@click.option(
    "--hpo-sampler",
    default="tpe_multi",
    type=click.Choice(["tpe", "tpe_multi", "cmaes"]),
    help="Optuna sampler (default: tpe_multi).",
)
@click.option(
    "--hpo-pruner",
    default="hyperband",
    type=click.Choice(["hyperband", "median", "none"]),
    help="Optuna pruner (default: hyperband).",
)
@click.option(
    "--use-tuned-params",
    is_flag=True,
    help="Load best_config from results/optuna_tuning.json for non-HPO runs.",
)
@click.option(
    "--no-trial-isolation",
    is_flag=True,
    help=(
        "Debugging only. Disable per-trial subprocess isolation for Optuna HPO; "
        "run trials in-process (legacy behavior). A wedged CUDA kernel will "
        "then hang the whole sweep."
    ),
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def main(
    run_all: bool,
    download_only: bool,
    preprocess_only: bool,
    train_only: bool,
    evaluate_only: bool,
    skip_large: bool,
    clean: bool,
    seed: int,
    random_split: bool,
    tasks: str | None,
    models: str | None,
    data_dir: str,
    output_dir: str,
    paper: bool,
    export_ds: bool,
    bootstrap: bool,
    ablation: bool,
    fair_tuning: bool,
    sensitivity: bool,
    multi_seed: bool,
    region_heterogeneity: bool,
    feature_ordering: bool,
    optuna_hpo: bool,
    no_monitoring_features: bool,
    provenance_free: bool,
    mcl_exceedance: bool,
    val_reuse_bias: bool,
    power_analysis: bool,
    sensitivity_bounds: bool,
    detection_only_ablation: bool,
    t6_arsenic: bool,
    shap_interactions: bool,
    shap_analysis: bool,
    seed_inflation: bool,
    split_comparison: bool,
    split_size_matched: bool,
    spatial_block_bootstrap: bool,
    buffered_loro: bool,
    icp_recoverability: bool,
    loro_cv: bool,
    loro_equity: bool,
    with_uncertainty: bool,
    strict: bool,
    verify: bool,
    gpu_id: int | None,
    n_trials: int | None,
    hpo_tasks: str | None,
    hpo_models: str | None,
    hpo_sampler: str,
    hpo_pruner: str,
    use_tuned_params: bool,
    no_trial_isolation: bool,
    verbose: bool,
) -> None:
    """Run the AquaContam reproducibility pipeline."""
    global _STRICT
    _STRICT = strict
    set_strict(strict)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # GPU diagnostic: confirm CUDA_VISIBLE_DEVICES picked the intended device
    if gpu_id is not None:
        import torch as _torch

        cvd = _os.environ.get("CUDA_VISIBLE_DEVICES")
        order = _os.environ.get("CUDA_DEVICE_ORDER")
        click.echo(f"GPU env: CUDA_DEVICE_ORDER={order} CUDA_VISIBLE_DEVICES={cvd}")
        if _torch.cuda.is_available():
            for i in range(_torch.cuda.device_count()):
                name = _torch.cuda.get_device_name(i)
                vram = _torch.cuda.get_device_properties(i).total_memory / 1e9
                click.echo(f"  cuda:{i} = {name} ({vram:.1f} GB)")
        else:
            click.echo("  WARNING: torch.cuda.is_available() == False")

    # Global random seeds for reproducibility
    from aquacontam._reproducibility import set_seed

    set_seed(seed)

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Feature-exclusion set, computed once so training AND the LORO / multi-seed
    # robustness analyses all use the same restricted feature set.
    if provenance_free:
        exclude_feats: list[str] | None = list(PROVENANCE_FREE_EXCLUDE)
    elif no_monitoring_features:
        exclude_feats = ["n_samples", "mean_detection_limit"]
    else:
        exclude_feats = None

    # Verify-only mode: check checksums and exit
    if verify:
        ok = _verify_results_checksums(output_path)
        if ok:
            click.echo("All results checksums verified OK.")
            raise SystemExit(0)
        else:
            click.echo("Checksum verification FAILED.", err=True)
            raise SystemExit(1)

    # Record software versions for reproducibility
    versions = _save_software_versions(output_path)
    click.echo(
        f"Software versions recorded: Python {versions.get('python', '?')}, "
        f"numpy {versions.get('numpy', '?')}, sklearn {versions.get('scikit-learn', '?')}, "
        f"xgboost {versions.get('xgboost', '?')}"
    )

    if clean:
        click.echo("Cleaning interim caches...")
        _clean_caches(data_path)

    task_filter = [t.strip() for t in tasks.split(",")] if tasks else None
    model_filter = [m.strip() for m in models.split(",")] if models else None

    # --with-uncertainty enables LORO + bootstrap
    if with_uncertainty:
        bootstrap = True
        loro_cv = True
        loro_equity = True

    # --all implies all optional analyses
    if run_all:
        paper = True
        export_ds = True
        bootstrap = True
        ablation = True
        fair_tuning = True
        sensitivity = True
        multi_seed = True
        region_heterogeneity = True
        feature_ordering = True
        optuna_hpo = True
        mcl_exceedance = True
        val_reuse_bias = True
        power_analysis = True
        sensitivity_bounds = True
        detection_only_ablation = True
        shap_interactions = True
        seed_inflation = True
        loro_cv = True
        loro_equity = True

    do_download = run_all or download_only
    do_preprocess = run_all or preprocess_only
    do_train = run_all or train_only or evaluate_only
    do_export = run_all or train_only or evaluate_only

    if not any(
        [
            do_download,
            do_preprocess,
            do_train,
            do_export,
            paper,
            export_ds,
            bootstrap,
            ablation,
            fair_tuning,
            multi_seed,
            region_heterogeneity,
            feature_ordering,
            optuna_hpo,
            sensitivity,
            mcl_exceedance,
            val_reuse_bias,
            power_analysis,
            sensitivity_bounds,
            detection_only_ablation,
            shap_interactions,
            shap_analysis,
            seed_inflation,
            split_comparison,
            spatial_block_bootstrap,
            buffered_loro,
            icp_recoverability,
            loro_cv,
            loro_equity,
            t6_arsenic,
        ]
    ):
        click.echo("No stage selected. Use --all or a specific --*-only flag.")
        click.echo("Run 'python scripts/reproduce.py --help' for options.")
        return

    downloaded: dict[str, Path] = {}
    wq_df = None

    if do_download:
        click.echo("Stage 1/6: Downloading data...")
        downloaded, dl_stats = _download_data(data_path, skip_large=skip_large)
        n_avail = dl_stats["cached"] + dl_stats["new"]
        click.echo(
            f"  Data sources: {n_avail} available"
            f" ({dl_stats['cached']} cached, {dl_stats['new']} new)"
            f", {dl_stats['unavailable']} unavailable"
        )

    if do_preprocess:
        click.echo("Stage 2/6: Preprocessing data...")
        if not downloaded:
            # Discover existing data files from prior runs
            for subdir in ("processed", "interim", "raw"):
                for p in (data_path / subdir).glob("*.parquet"):
                    if p.stem not in downloaded:
                        downloaded[p.stem] = p
        wq_df = _preprocess_data(data_path, downloaded)
        click.echo(f"  Merged WQ data: {len(wq_df)} rows")

    feature_dfs: list[Any] = []
    if do_train or do_export or export_ds:
        if wq_df is None:
            import pandas as pd

            cached = data_path / "interim" / "merged_wq.parquet"
            if cached.exists():
                wq_df = pd.read_parquet(cached)
            else:
                click.echo("No preprocessed data found. Run --preprocess-only first.")
                return

        # If we didn't download this run, discover existing files
        if not downloaded:
            raw_dir = data_path / "raw"
            interim_dir = data_path / "interim"
            processed_dir = data_path / "processed"
            for d in (raw_dir, interim_dir, processed_dir):
                if not d.exists():
                    continue
                for p in d.glob("*.parquet"):
                    if p.stem not in downloaded:
                        downloaded[p.stem] = p
            for p in raw_dir.glob("*.shp"):
                downloaded[p.stem] = p
            for p in raw_dir.rglob("*.tif"):
                downloaded[p.stem] = p
            # Map well-known filenames → canonical keys
            if "us_aquifers" in downloaded:
                downloaded["usgs_aquifers"] = downloaded["us_aquifers"]
            # Discover NLCD raster in subdirectories if not already mapped
            if "nlcd" not in downloaded:
                try:
                    from aquacontam.features.land_use import download_nlcd

                    nlcd_path = download_nlcd(raw_dir)
                    downloaded["nlcd"] = nlcd_path
                except Exception:
                    if is_strict():
                        raise

        click.echo("Stage 3/6: Extracting features...")
        feature_dfs = _extract_features(data_path, wq_df, downloaded, skip_large=skip_large)
        click.echo(f"  Extracted {len(feature_dfs)} feature sets")

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, tuple[Any, str, float]] = {}
    icp_models: dict[str, Any] = {}
    if do_train:
        click.echo("Stage 4-5/6: Training and evaluating...")
        # exclude_feats computed once near the top (provenance-free / no-monitoring).
        results, fitted_models, icp_models = _train_and_evaluate(
            wq_df,
            feature_dfs,
            downloaded,
            output_path,
            seed=seed,
            task_filter=task_filter,
            model_filter=model_filter,
            use_random_split=random_split,
            exclude_features=exclude_feats,
            resume=not clean,
            use_tuned_params=use_tuned_params,
        )
        click.echo(f"  Completed {len(results)} task/model evaluations")

    if do_export and results:
        click.echo("Stage 6/6: Exporting results...")
        if random_split:
            random_results_path = output_path / "results_random_split.json"
            random_results_path.write_text(json.dumps(results, indent=2, default=str))
            click.echo(f"  Random-split results saved to {random_results_path}")
        else:
            _export_results(results, output_path)
        click.echo(f"  Results exported to {output_path}")

    if fitted_models:
        click.echo("  Generating webapp predictions...")
        try:
            _generate_webapp_predictions(wq_df, feature_dfs, fitted_models, output_path)
            click.echo(f"  Predictions written to {output_path / 'predictions'}")
        except Exception:
            logger.warning("Failed to generate webapp predictions", exc_info=True)
            if is_strict():
                raise
            click.echo("  Warning: webapp prediction generation failed (see log)")

        click.echo("  Exporting feature importances...")
        _export_feature_importances(fitted_models, output_path)

        click.echo("  Running equity analysis...")
        _run_equity_analysis(wq_df, feature_dfs, fitted_models, output_path)

        click.echo("  Running monitoring-inequity analysis...")
        _run_monitoring_inequity(wq_df, feature_dfs, output_path, seed=seed)

        click.echo("  Exporting ICP diagnostics...")
        _export_icp_diagnostics(results, fitted_models, output_path, icp_models=icp_models)

    if bootstrap:
        if not results:
            results_file = output_path / "results.json"
            if results_file.exists():
                results = json.loads(results_file.read_text())
                click.echo(f"Loaded {len(results)} existing results from {results_file}")
            else:
                click.echo("No results found. Run --train-only first.")

        if results:
            click.echo("Computing bootstrap confidence intervals...")
            results = _run_bootstrap_cis(results, output_path, n_bootstrap=1000, seed=seed)
            if not random_split:
                _export_results(results, output_path)
            click.echo("  Bootstrap CIs computed for all classification results")

    # Analyses that run automatically with --all
    if run_all and results:
        click.echo("Running spatial autocorrelation analysis...")
        _run_spatial_autocorrelation(results, output_path)

        click.echo("Running calibration analysis...")
        _run_calibration_analysis(results, output_path)

        click.echo("Running SHAP analysis...")
        _run_shap_analysis(results, fitted_models, output_path)

        click.echo("Running SHAP interaction analysis...")
        _run_shap_interaction_analysis(results, fitted_models, output_path)

        click.echo("Running detection-only source ablation...")
        _run_detection_only_ablation(wq_df, results, output_path, fitted_models=fitted_models)

        click.echo("Running conformal prediction analysis...")
        _run_conformal_analysis(results, output_path)

        click.echo("Running group-conditional conformal prediction analysis...")
        _run_group_conformal_analysis(wq_df, feature_dfs, output_path, seed=seed)

        click.echo("Running causal deconfounding analysis...")
        _run_causal_deconfounding(wq_df, feature_dfs, output_path, seed=seed)

        click.echo("Running transfer ablation (T5 with/without system characteristics)...")
        _run_transfer_ablation(wq_df, feature_dfs, downloaded, output_path, seed=seed)

        click.echo("Running deduplication impact comparison...")
        _run_dedup_comparison(wq_df, feature_dfs, output_path, seed=seed)

        click.echo("Running split strategy comparison...")
        _run_split_comparison(wq_df, feature_dfs, output_path, seed=seed)

        click.echo("Running coordinate sensitivity analysis...")
        _run_coordinate_sensitivity(wq_df, feature_dfs, output_path, seed=seed)

        click.echo("Running hyperparameter sensitivity analysis...")
        _run_hyperparam_sensitivity(wq_df, feature_dfs, output_path, seed=seed)

        click.echo("Running pairwise DeLong model comparisons...")
        _run_model_comparison(results, output_path)

        click.echo("Running lift/decile analysis...")
        _run_lift_analysis(wq_df, feature_dfs, fitted_models, output_path, seed=seed)

        click.echo("Running external validation on state databases...")
        _run_external_validation(wq_df, feature_dfs, fitted_models, downloaded, output_path)

        click.echo("Running sensitivity bounds (Rosenbaum/E-values)...")
        _run_sensitivity_bounds(output_path)

        click.echo("Running power analysis...")
        _run_power_analysis(output_path)

    # LORO cross-validation (runs with --all, --with-uncertainty, or --loro-cv/--loro-equity)
    if loro_cv:
        wq_df, feature_dfs = _ensure_data(wq_df, feature_dfs, data_path)
        if wq_df is not None:
            click.echo("Running LORO cross-validation...")
            _run_loro_cv(
                wq_df, feature_dfs, output_path, seed=seed, exclude_features=exclude_feats
            )

    if loro_equity:
        wq_df, feature_dfs = _ensure_data(wq_df, feature_dfs, data_path)
        if wq_df is not None:
            click.echo("Running LORO equity analysis...")
            _run_loro_equity_analysis(wq_df, feature_dfs, output_path, seed=seed)

    # Optional analyses that need data loaded
    for flag, label, fn, extra_kwargs in [
        (
            multi_seed,
            "multi-seed stability",
            _run_multi_seed_analysis,
            {"exclude_features": exclude_feats},
        ),
        (region_heterogeneity, "per-region heterogeneity", _run_region_heterogeneity, {}),
        (ablation, "feature category ablation", _run_feature_ablation, {}),
        (fair_tuning, "symmetric hyperparameter tuning", _run_fair_tuning, {}),
        (
            optuna_hpo,
            "Optuna Bayesian HPO",
            _run_optuna_tuning,
            {
                "n_trials": n_trials if n_trials is not None else 300,
                "tasks": hpo_tasks.split(",") if hpo_tasks else None,
                "models": hpo_models.split(",") if hpo_models else None,
                # After CUDA_VISIBLE_DEVICES remap (set in _early_gpu_env_setup),
                # the user-selected device appears as index 0 to all CUDA libs.
                # Tree libs (xgb/lgbm/cb) need this post-remap index, not the
                # raw --gpu-id PCI bus index.
                "gpu_id": 0 if gpu_id is not None else None,
                "sampler": hpo_sampler,
                "pruner": hpo_pruner,
                "inproc": no_trial_isolation,
            },
        ),
        (feature_ordering, "CNN1D feature ordering", _run_cnn1d_feature_ordering, {}),
        (sensitivity, "sensitivity analysis", _run_sensitivity_analysis, {}),
        (mcl_exceedance, "MCL exceedance analysis", _run_mcl_exceedance_analysis, {}),
        (val_reuse_bias, "validation reuse bias quantification", _run_val_reuse_bias, {}),
        (
            split_comparison and not run_all,
            "split strategy comparison",
            _run_split_comparison,
            {"size_matched": split_size_matched},
        ),
        (
            spatial_block_bootstrap and not run_all,
            "spatial block bootstrap",
            _run_spatial_block_bootstrap,
            {},
        ),
        (buffered_loro and not run_all, "buffered-boundary LORO", _run_buffered_loro, {}),
        (
            icp_recoverability and not run_all,
            "ICP recoverability probe",
            _run_icp_recoverability,
            {},
        ),
    ]:
        if flag:
            wq_df, feature_dfs = _ensure_data(wq_df, feature_dfs, data_path)
            if wq_df is not None:
                click.echo(f"Running {label}...")
                fn(wq_df, feature_dfs, output_path, seed=seed, **extra_kwargs)
                click.echo(f"  {label.capitalize()} complete")

    # Results-based analyses (standalone flags)
    if power_analysis and not run_all:
        click.echo("Running power analysis...")
        _run_power_analysis(output_path)
        click.echo("  Power analysis complete")

    if sensitivity_bounds and not run_all:
        click.echo("Running sensitivity bounds (Rosenbaum/E-values)...")
        _run_sensitivity_bounds(output_path)
        click.echo("  Sensitivity bounds complete")

    if detection_only_ablation and not run_all:
        results_file = output_path / "results.json"
        if results_file.exists() and not results:
            results = json.loads(results_file.read_text())
        wq_df, feature_dfs = _ensure_data(wq_df, feature_dfs, data_path)
        if wq_df is not None and results:
            click.echo("Running detection-only source ablation...")
            _run_detection_only_ablation(wq_df, results, output_path, fitted_models=fitted_models)
            click.echo("  Detection-only ablation complete")

    if shap_analysis and not run_all:
        # Local names: the refit's 2-entry results list must not leak into the
        # main `results` flow (summaries, checksums) of a combined invocation.
        shap_results, shap_models = results, fitted_models
        if not shap_models:
            import tempfile

            wq_df, feature_dfs = _ensure_data(wq_df, feature_dfs, data_path)
            if wq_df is not None:
                click.echo("Refitting T1/T4 XGBoost for SHAP (no fitted models in memory)...")
                # Scratch dir: _train_and_evaluate writes task checkpoints into
                # its output_dir; pointing it at output_path would clobber the
                # full run's checkpoints (e.g. results_provenance_free/).
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                    shap_results, shap_models, _ = _train_and_evaluate(
                        wq_df,
                        feature_dfs,
                        downloaded,
                        Path(tmp),
                        seed=seed,
                        task_filter=["T1", "T4"],
                        model_filter=["xgboost"],
                        use_random_split=random_split,
                        exclude_features=exclude_feats,
                        resume=False,
                        use_tuned_params=use_tuned_params,
                    )
        if shap_results and shap_models:
            click.echo("Running SHAP analysis...")
            _run_shap_analysis(shap_results, shap_models, output_path)
            click.echo("  SHAP analysis complete")
            # Let a combined --shap --shap-interactions invocation reuse the refit.
            if shap_interactions:
                results, fitted_models = shap_results, shap_models

    if shap_interactions and not run_all:
        results_file = output_path / "results.json"
        if results_file.exists() and not results:
            results = json.loads(results_file.read_text())
        if results and fitted_models:
            click.echo("Running SHAP interaction analysis...")
            _run_shap_interaction_analysis(results, fitted_models, output_path)
            click.echo("  SHAP interactions complete")

    if seed_inflation or (run_all and (output_path / "multi_seed_stability.json").exists()):
        click.echo("Running seed inflation check...")
        _run_seed_inflation_check(output_path)
        click.echo("  Seed inflation check complete")

    if t6_arsenic or run_all:
        from aquacontam.pipeline.t6_arsenic import run_t6_experiment

        # Tuned and default results coexist for comparison; --all stays default.
        t6_out = "t6_arsenic_tuned.json" if use_tuned_params else "t6_arsenic.json"
        click.echo("Running T6 arsenic transfer (public-supply to domestic)...")
        try:
            run_t6_experiment(
                data_path,
                out_path=output_path / t6_out,
                seed=seed,
                use_tuned_params=use_tuned_params,
            )
            click.echo("  T6 arsenic experiment complete")
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("T6 arsenic experiment skipped: %s", exc)
            if is_strict():
                raise
            click.echo("  T6 skipped (NGA data unavailable - see log)")

    # Tip for --train-only users who haven't enabled uncertainty quantification
    if (train_only or evaluate_only) and not (bootstrap or loro_cv or loro_equity):
        click.echo("Tip: add --with-uncertainty for LORO CV and bootstrap confidence intervals")

    if paper:
        click.echo("Generating paper assets...")
        _generate_paper_assets(output_path, data_path)

    if export_ds and wq_df is not None:
        click.echo("Exporting Zenodo dataset...")
        _export_zenodo_dataset(wq_df, feature_dfs, data_path, output_path)

    if results:
        _print_results_summary(results)

    # Generate checksums for all result files
    _generate_results_checksums(output_path)


if __name__ == "__main__":
    main()
