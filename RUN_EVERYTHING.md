# AquaContam — Complete Execution Guide

> Runs every script and experiment end to end, finishing with the deterministic
> paper gate (section 12). The webapp and deployment sections apply to the
> development tree; the webapp is not included in the public release.

## 0. Environment Setup

```bash
# Install with ALL optional dependency groups
pip install -e ".[dev,test,dl,paper,webapp,boost,interpret]"
```

## 1. Code Quality Checks

```bash
ruff check src/ tests/                     # Lint
ruff format --check src/ tests/            # Format check
mypy src/aquacontam/                       # Type check
```

## 2. Full Test Suite

```bash
pytest tests/unit/ -x --tb=short -q        # Unit tests (fast, fail-fast)
pytest tests/ --cov=aquacontam             # Full tests with coverage
```

## 3. Full Reproducibility Pipeline (geographic splits)

This is the master pipeline. `--all` enables every stage:
download, preprocess, extract features, train ALL models on ALL tasks,
evaluate, export results, bootstrap CIs, ablation, feature ordering,
spatial autocorrelation, calibration, SHAP, conformal prediction,
split strategy comparison, hyperparameter sensitivity, LORO CV,
paper assets, and Zenodo dataset export.

```bash
python scripts/reproduce.py --all --verbose --seed 42
```

## 4. Random-Split Comparison Run

Re-run training with random splits to quantify spatial leakage inflation.
Uses all tasks and all models.

```bash
python scripts/reproduce.py --train-only --random-split --verbose --seed 42
```

## 5. Individual Task Granular Runs

Run each task individually with every model to ensure full coverage.
These overlap with `--all` but ensure no task is silently skipped.

```bash
python scripts/reproduce.py --train-only --tasks T1 --verbose
python scripts/reproduce.py --train-only --tasks T2 --verbose
python scripts/reproduce.py --train-only --tasks T3 --verbose
python scripts/reproduce.py --train-only --tasks T4 --verbose
python scripts/reproduce.py --train-only --tasks T5 --verbose
python scripts/reproduce.py --train-only --tasks T7 --verbose
```

## 6. Standalone Analysis Stages

These can be run independently after training (they load cached results).

```bash
python scripts/reproduce.py --bootstrap --verbose
python scripts/reproduce.py --ablation --verbose
python scripts/reproduce.py --feature-ordering --verbose
```

## 7. Paper Asset Generation

```bash
# Generate publication figures (Nature Water format; regenerates from the frozen archive)
python paper/generate_figures.py --results results/paper_frozen --output paper/figures/ --data-dir data/

# Generate publication tables (Markdown/LaTeX)
python paper/generate_tables.py --results results/paper_frozen --output paper/tables/
```

## 8. CLI Commands

```bash
# Benchmark task listing
python -m aquacontam benchmark

# Dataset export (Zenodo-ready archive)
python -m aquacontam export --output aquacontam-dataset

# Leaderboard operations
python -m aquacontam leaderboard validate results/submissions/xgboost_classifier.json
python -m aquacontam leaderboard rank results/submissions/
python -m aquacontam leaderboard publish results/submissions/ -o LEADERBOARD.md
```

## 9. Interactive Webapp

Launch the Dash risk map webapp (runs until Ctrl-C).

```bash
python -m aquacontam webapp --port 8050 --debug --predictions-dir results/predictions
```

Alternative via module directly:

```bash
python -m aquacontam.webapp --port 8050 --debug --predictions-dir results/predictions
```

## 10. Hugging Face Spaces Deployment Test

Test the HF Spaces deployment entry point locally.

```bash
PREDICTIONS_DIR=results/predictions PORT=7860 python deploy/hf_spaces/app.py
```

## 11. Build & Package

```bash
python -m build
```

## 12. Paper Gate (deterministic consistency check)

Verifies every reader-facing number, table, stamp, and checksum against the
committed frozen archive in minutes — no pipeline run required.

```bash
PYTHONUTF8=1 python paper/final_gate.py --check
python paper/regenerate_leaderboard.py --check
```

---

## Notes

- **Step 3 is the heavyweight**: `--all` triggers data download (~8 GB including NLCD raster
  and EJScreen), trains 20 model families across 7 tasks (T1–T7), runs bootstrap
  (1000 iterations), feature ablation, CNN1D ordering sensitivity (5 random seeds), LORO CV
  (10 folds), SHAP analysis, conformal prediction, split strategy comparison, and hyperparameter
  grid search. Expect multiple hours on a modern machine with GPU.
- **Step 4** re-runs all training with random splits — adds significant compute time but is
  necessary for the split-strategy comparison (Table 5 in the paper).
- **Steps 9–10** launch interactive servers that run until terminated — run them last or in
  separate terminals.
- **T6** runs as an arsenic next-generation-analyte probe on the USGS NGA dataset
  (`--t6-arsenic`; frozen result `results/paper_frozen/t6_arsenic.json`). The NJ
  private-well definition in `configs/experiment.yaml` is a provenance stub only.
- **GPU models** (MLP, CNN1D, GNN): will use GPU if available via PyTorch CUDA, otherwise
  fall back to CPU.
- **Optional dependencies**: If `shap`, `lightgbm`, `catboost`, `lifelines`, or
  `torch-geometric` are not installed, the corresponding analyses/models are skipped with
  log messages — they won't cause failures.

## Verification Checklist

After running everything, verify these outputs exist:

1. `results/results.json` — entries for all task/model combinations
2. `results/results_random_split.json` — split comparison results
3. `results/submissions/` — per-model leaderboard JSONs
4. `results/predictions/` — Hive-partitioned Parquet files
5. `paper/figures/` — generated PNG/PDF figures
6. `paper/tables/` — generated Markdown tables
7. `LEADERBOARD.md` — published leaderboard
8. `aquacontam-dataset/` — Zenodo dataset export
9. `dist/` — built wheel
10. Analysis JSONs in `results/`:
    - `software_versions.json`
    - `feature_importance.json`
    - `equity_analysis.json`
    - `spatial_autocorrelation.json`
    - `calibration_analysis.json`
    - `shap_T1.json`, `shap_T4.json`
    - `conformal_results.json`
    - `split_comparison.json`
    - `hyperparam_sensitivity.json`
    - `loro_cv.json`
    - `feature_ablation.json`
    - `cnn1d_feature_ordering.json`
