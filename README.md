![AquaContam — Contamination Flow Field](assets/header.png)

# AquaContam

**A National-Scale ML-Ready Dataset and Benchmark for Predicting PFAS and Heavy Metal Contamination in U.S. Drinking Water**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Code license: Apache-2.0](https://img.shields.io/badge/code%20license-Apache--2.0-green)
![Data license: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-green)
![CI](https://github.com/tjnewton/aquacontam/actions/workflows/ci.yml/badge.svg)

[Contributing](CONTRIBUTING.md) | [Changelog](CHANGELOG.md) | [Leaderboard](LEADERBOARD.md)

> AquaContam is a research benchmark, not a regulatory or operational
> risk-assessment product.

---

## Motivation

172 million Americans drink PFAS-contaminated water, yet **no ML-ready integrated dataset exists** for contamination prediction. Researchers must manually assemble data from fifteen fragmented federal and state sources. AquaContam closes this gap by providing:

- An integrated, standardized dataset combining UCMR5 (~1.9M samples, 29 PFAS), UCMR3 (~1.1M samples, 38 contaminants), WQP, six state databases, TRI/DoD PFAS sites, heavy metals, and geospatial features
- 7 benchmark tasks spanning detection, regression, multi-contaminant profiling, transfer learning, and temporal prediction
- 20 baseline model families spanning classical ML, deep learning, gradient boosting, survival analysis, foundation models, and ensembles
- Environmental justice analysis via EJScreen demographic overlays
- A deterministic gate that re-derives the benchmark's tables, leaderboard, input-hash stamps, checksums, and Source Data from a frozen archive in minutes

## Installation

### Prerequisites

- Python >= 3.10
- conda (recommended for geospatial dependencies)

### Setup

```bash
# Clone the repository
git clone https://github.com/tjnewton/aquacontam.git
cd aquacontam

# Create conda environment (handles GDAL, PROJ, etc.)
conda env create -f environment.yml
conda activate aquacontam

# (environment.yml already installed the package with its extras via pip;
# re-run `pip install -e .` only after changing pyproject.toml)
```

### Extras

Install additional extras depending on your use case:

| Extra | Install command | When to use |
|-------|----------------|-------------|
| `dev` | `pip install -e ".[dev]"` | Linting, type checking, pre-commit hooks |
| `test` | `pip install -e ".[test]"` | Running the test suite |
| `dl` | `pip install -e ".[dl]"` | MLP + CNN1D deep learning models (PyTorch) |
| `paper` | `pip install -e ".[paper]"` | Figures and tables for the paper |
| `gate` | `pip install -e ".[gate]"` | Running the verification gate (`final_gate.py --check`) |
| `boost` | `pip install -e ".[boost]"` | LightGBM + CatBoost gradient boosting |
| `hpo` | `pip install -e ".[hpo]"` | Optuna hyperparameter optimization |
| `interpret` | `pip install -e ".[interpret]"` | SHAP explanations + survival models (Tobit, AFT) |
| `foundation` | `pip install -e ".[foundation]"` | TabPFN v2 foundation model |

Combine extras as needed:

```bash
pip install -e ".[dev,test]"        # development
pip install -e ".[dev,test,dl]"     # development + deep learning
```

### Verify installation

```bash
aquacontam --help
```

## Quick Start

### Reproducibility pipeline (recommended)

The fastest path to benchmark results is `scripts/reproduce.py`, which runs the full pipeline end-to-end:

```bash
# Full pipeline: download → preprocess → features → train → evaluate → export
python scripts/reproduce.py --all --skip-large --tasks T1 --models xgboost -v
```

**Pipeline stages:**

| Stage | Description |
|-------|-------------|
| 1. Download | Fetches all data sources (idempotent; skips existing files) |
| 2. Preprocess | Merges water quality data, joins ZIP codes, geocodes systems |
| 3. Features | Extracts proximity, land use, hydrogeology, and demographic features |
| 4-5. Train + Evaluate | Fits models and computes metrics per task |
| 6. Export | Writes `results/results.json`, `results/submissions/*.json`, and `results/predictions/` (Hive-partitioned Parquet per-system prediction export) |

**Key flags:**

| Flag | Description |
|------|-------------|
| `--all` | Run the full pipeline |
| `--download-only` | Only download data sources |
| `--preprocess-only` | Only preprocess/merge data |
| `--train-only` | Only train and evaluate models |
| `--skip-large` | Skip NLCD (~1 GB) and EJScreen (~6 GB) downloads |
| `--clean` | Delete interim caches (merged WQ + features) before running |
| `--tasks T1,T3` | Run specific tasks (comma-separated) |
| `--models xgboost,random_forest` | Run specific models (comma-separated) |
| `--seed 42` | Random seed for reproducibility |
| `-v` | Verbose logging |

### Reproducing the paper from the archive

Two supported paths, by depth:

1. **Minutes-scale verification (no pipeline run):** the committed frozen archive
   (`results/paper_frozen/`) is the copy of record for the paper's results. Verify
   the benchmark artifacts against it with:

   ```bash
   PYTHONUTF8=1 python paper/final_gate.py --check          # tables, leaderboard, stamps, checksums, Source Data
   python paper/regenerate_leaderboard.py --check           # the leaderboard
   ```

   The manuscript sources are not distributed with this repository (see
   [Repository provenance](#repository-provenance)); the gate's
   manuscript-consistency clauses report `not distributed -- skipped` here and
   enforce where the manuscript lives.

2. **Full reproduction:** `scripts/reproduce.py --all` regenerates all results
   from raw data downloads plus the request-based Minnesota staging documented
   in `docs/data_requests/mn_mdh_bulk_export.md` (the MDH workbook is obtained
   on written request and verified against `data/checksums.sha256`; without it,
   `--strict` fails rather than silently diverging). Multi-day, GPU-dependent.

### Using the benchmark (external users)

To evaluate **your own model** on an AquaContam task with the canonical geographic
splits — without running the full pipeline:

```python
import pandas as pd
from pathlib import Path

from aquacontam.models import XGBoostClassifier  # any aquacontam.models.base.BaseModel
from aquacontam.benchmark.registry import run_task
import aquacontam.benchmark.tasks  # registers T1–T7

# 1. Get the dataset: download the curated AquaContam release from Zenodo (the DOI
#    is recorded in CITATION.cff and LICENSE-DATA once the archive is deposited),
#    or generate it with `reproduce.py --all`.
ds = Path("aquacontam-dataset")

# 2. Sample-level water-quality records. run_task re-applies the fixed geographic
#    split internally, so concatenate the provided train/val/test parquets.
wq_df = pd.concat(
    pd.read_parquet(ds / "data" / f"{s}.parquet") for s in ("train", "val", "test")
)

# 3. Pre-computed feature tables (indexed by pwsid).
feature_dfs = [pd.read_parquet(p) for p in sorted((ds / "features").glob("*.parquet"))]

# 4. Evaluate. Swap XGBoostClassifier() for any model implementing the
#    aquacontam.models.base.BaseModel interface (fit / predict / predict_proba).
result = run_task("T1", model=XGBoostClassifier(), data=wq_df, feature_dfs=feature_dfs)
print(result.metrics)  # {"auroc": ..., "auprc": ...}
```

Geographic stratification (train EPA regions 1, 3, 4, 5, 6 / val 2, 7 / test 8, 9, 10)
is applied automatically, so scores are comparable to the paper and the leaderboard.
To submit a result, see [`submissions/README.md`](submissions/README.md) and
[`LEADERBOARD.md`](LEADERBOARD.md).

### Hyperparameter optimization

Optuna-driven Bayesian HPO with sqlite-resumable studies and GPU passthrough for tree libraries:

```bash
# Full sweep across T1/T2/T4 with per-task search spaces (300 trials/model)
python scripts/reproduce.py --optuna --gpu-id 2 \
    --hpo-tasks T1,T2,T4 --n-trials 300

# Restrict to specific models — interrupted runs resume automatically
python scripts/reproduce.py --optuna --gpu-id 2 \
    --hpo-models xgboost_classifier,mlp_classifier --n-trials 50

# Use tuned configs in a downstream training run
python scripts/reproduce.py --train-only --use-tuned-params --tasks T1,T2,T4
```

| Flag | Description |
|------|-------------|
| `--gpu-id N` | CUDA device index in PCI bus order (matches `nvidia-smi -L`) |
| `--hpo-tasks T1,T2,T4` | Tasks to tune (T3/T5/T6/T7 not yet wired) |
| `--hpo-models name1,name2` | Restrict to specific model families |
| `--n-trials 300` | Trials per `(model, task)` study |
| `--hpo-sampler tpe_multi` | Sampler: `tpe` / `tpe_multi` (default) / `cmaes` |
| `--hpo-pruner hyperband` | Pruner: `hyperband` (default) / `median` / `none` |
| `--use-tuned-params` | Load best configs from `results/optuna_tuning.json` for non-HPO runs |
| `--no-trial-isolation` | Debugging only. Disables per-trial subprocess wrapper (legacy in-process path). |

Tuning state lives at `results/optuna_studies/{model}_{task}.db` (resumable across runs and machine restarts). Tree libs (XGBoost / LightGBM / CatBoost / XGBoost-AFT) train on GPU; torch models (MLP / CNN1D / Deep-Tobit / ZIT / ICP / GNN) pick up the same GPU automatically. Each trial runs in an isolated `spawn`-context subprocess with a per-model wall-clock timeout (default 30 min) so a wedged CUDA kernel kills only the trial, not the sweep — set `AQUACONTAM_HPO_INPROC=1` (or pass `--no-trial-isolation`) to opt out for debugging. Best configs are merged into `results/optuna_tuning.json` keyed by `(model, task)`. Requires the `[hpo]` extra.

### Python API

```python
from pathlib import Path
from aquacontam.data.ucmr5 import UCMR5Source

source = UCMR5Source(
    raw_dir=Path("data/raw"),
    interim_dir=Path("data/interim"),
    processed_dir=Path("data/processed"),
)
parquet_path = source.run()  # download → parse → validate → save
```

Run a benchmark task programmatically:

```python
import aquacontam.benchmark.tasks  # registers tasks
from aquacontam.benchmark.registry import run_task

result = run_task("T1", model=my_model, data=wq_df)
print(result.metrics)  # {"auroc": 0.92, "auprc": 0.85, ...}
```

## CLI Reference

AquaContam provides a Click-based CLI accessible via `aquacontam` or `python -m aquacontam`:

| Command | Description | Extra required |
|---------|-------------|---------------|
| `aquacontam benchmark` | Run benchmark tasks | — |
| `aquacontam export` | Package dataset for distribution | — |
| `aquacontam leaderboard validate <path>` | Validate a submission JSON | — |
| `aquacontam leaderboard rank <dir>` | Rank all submissions in a directory | — |
| `aquacontam leaderboard publish <dir>` | Generate and write leaderboard document | — |

## Benchmark Tasks

All tasks use geographic stratification by EPA region for train/val/test splits, preventing spatial leakage. Systems are split at the state level across 10 EPA regions.

| Task | Type | Description | Default analytes | Primary metric |
|------|------|-------------|------------------|----------------|
| T1 | Classification | Binary PFAS detection | PFOS, PFOA, PFBS, PFHxS, HFPO-DA | AUPRC |
| T2 | Regression | PFAS concentration prediction | PFOS, PFOA, PFHxS | RMSE |
| T3 | Multilabel classification | Multi-PFAS profile prediction | PFOS, PFOA, PFBS, PFHxS, HFPO-DA | Macro AUPRC |
| T4 | Classification | Heavy metal prediction | lead, copper | AUPRC |
| T5 | Classification | Cross-contaminant transfer learning | lead ↔ PFOS | AUPRC |
| T6 | Classification | Arsenic transfer probe (public-supply → domestic wells) | arsenic | AUPRC |
| T7 | Classification | Temporal prediction (UCMR3 → UCMR5) | PFOS, PFOA, PFBS, PFHxS | AUPRC |

**Metrics by task type:**

- **Classification** (T1, T4, T5, T6, T7): AUROC, AUPRC, F1, precision, recall, accuracy, balanced accuracy
- **Regression** (T2): RMSE, MAE, R², explained variance, median absolute error
- **Multilabel** (T3): subset accuracy, hamming loss, macro/micro AUROC, macro/micro AUPRC, macro/micro F1, label ranking average precision, mean per-label AUROC

## Baseline Models

| Model | Extra required | Classifier | Regressor |
|-------|---------------|------------|-----------|
| Dummy | — (base) | `DummyClassifierBaseline` | — |
| Logistic Regression | — (base) | `LogisticRegressionClassifier` | — |
| XGBoost | — (base) | `XGBoostClassifier` | `XGBoostRegressor` |
| Random Forest | — (base) | `RandomForestClassifier` | `RandomForestRegressor` |
| LightGBM | `[boost]` | `LightGBMClassifier` | `LightGBMRegressor` |
| CatBoost | `[boost]` | `CatBoostClassifier` | `CatBoostRegressor` |
| MLP | `[dl]` | `MLPClassifier` | `MLPRegressor` |
| CNN1D | `[dl]` | `CNN1DClassifier` | `CNN1DRegressor` |
| GNN | `[dl]` | `GNNClassifier` | — |
| Deep Tobit | `[dl]` | `DeepTobitClassifier` | `DeepTobitRegressor` |
| Tobit | `[interpret]` | — | `TobitRegressor` |
| AFT | `[interpret]` | — | `AFTRegressor` |
| XGBoost AFT | — (base) | — | `XGBoostAFTRegressor` |
| Hurdle | — (base) | — | `HurdleRegressor` |
| Zero-Inflated Tobit | `[dl]` | `ZeroInflatedTobitClassifier` | `ZeroInflatedTobitRegressor` |
| ICP (monitoring-invariant) | `[dl]` | `ICPClassifier` | `ICPRegressor` |
| TabPFN v2 | `[foundation]` | `TabPFNClassifier` | — |
| Voting Ensemble | — (base) | `VotingEnsembleClassifier` | — |
| Stacking Ensemble | — (base) | `StackingEnsembleClassifier` | — |
| Averaging Ensemble | — (base) | — | `AveragingEnsembleRegressor` |

All models implement the `BaseModel` interface defined in `src/aquacontam/models/base.py`. `predict()` and `predict_proba()` have concrete defaults that delegate to `self._model`; subclasses override only `fit()` (required) and optionally `predict`/`predict_proba` for custom logic (e.g., torch models):

```python
class BaseModel(ABC):
    def fit(self, X_train, y_train, **kwargs) -> None: ...       # abstract — each model implements
    def predict(self, X, **kwargs) -> np.ndarray: ...             # concrete default
    def predict_proba(self, X, **kwargs) -> np.ndarray: ...       # concrete default
    def evaluate(self, X_test, y_test, metrics=None, *, task_type="classification") -> dict: ...
    def feature_importances(self) -> pd.Series: ...
```

### Usage Examples

**Standard classification** (XGBoost):

```python
from aquacontam.models.xgboost import XGBoostClassifier

model = XGBoostClassifier(config={"n_estimators": 300, "max_depth": 6})
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)  # early stopping on val set
metrics = model.evaluate(X_test, y_test)
print(metrics)  # {"auroc": 0.725, "auprc": 0.326, "f1": 0.42, ...}
```

**Censoring-aware classification** (Deep Tobit MLP) — uses Tobit likelihood to properly model left-censored non-detects:

```python
from aquacontam.models.deep_tobit import DeepTobitClassifier

model = DeepTobitClassifier(config={"hidden_sizes": [128, 64, 32], "dropout": 0.3})
model.fit(
    X_train, y_train,
    censored=censored_train,            # bool array: True for non-detect samples
    detection_limits=dl_train,           # float array: per-sample detection limits
    X_val=X_val, y_val=y_val,
)
proba = model.predict_proba(X_test)     # P(detected) = 1 - Phi((DL - mu) / sigma)
```

**Censoring-aware regression** (Tobit) — left-censored MLE via scipy:

```python
from aquacontam.models.tobit import TobitRegressor

model = TobitRegressor(config={"max_iter": 1000})
model.fit(X_train, y_train, censored=censored_train, detection_limits=dl_train)
metrics = model.evaluate(X_test, y_test, task_type="regression")
print(metrics)  # {"rmse": ..., "mae": ..., "r2": ...}
```

**Foundation model** (TabPFN v2) — in-context learning with no gradient training:

```python
from aquacontam.models.tabpfn import TabPFNClassifier

model = TabPFNClassifier(config={"n_estimators": 4})
model.fit(X_train, y_train)             # stores data (no gradient updates)
proba = model.predict_proba(X_test)     # runs in-context learning at inference
```

**Ensemble** (Voting) — soft-voting over tree-based classifiers:

```python
from aquacontam.models.ensemble import VotingEnsembleClassifier

model = VotingEnsembleClassifier(
    config={"base_models": ["xgboost", "random_forest", "lightgbm"], "voting": "soft"}
)
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

**Calibration** (Conformal prediction) — finite-sample coverage guarantees:

```python
from aquacontam.models.xgboost import XGBoostClassifier
from aquacontam.calibration.conformal import ConformalClassifier

# 1. Train a base model
base = XGBoostClassifier()
base.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# 2. Wrap with conformal predictor and calibrate
conformal = ConformalClassifier(model=base, alpha=0.1)  # target 90% coverage
conformal.calibrate(X_cal, y_cal)                       # held-out calibration set

# 3. Produce prediction sets
pred_sets = conformal.predict_sets(X_test)               # array of {0}, {1}, or {0, 1}
```

## Data Sources

The benchmark integrates fifteen water-quality data sources:

| Dataset | Records | Contaminants | Coverage | Format / status |
|---------|---------|-------------|----------|-----------------|
| UCMR5 | ~1.9M samples | 29 PFAS + lithium | National (systems >= 3,300 pop) | Text files |
| UCMR3 | ~1.1M samples | 38 contaminants (6 PFAS) | National (systems >= 10k pop) | Text files |
| SDWIS | 160k systems | Lead, copper | National | ECHO downloads |
| WQP | ~35K records | 13 PFAS | National (ambient monitoring) | REST API |
| Michigan MPART | ~5.6K records | 5 PFAS | Michigan | ArcGIS REST |
| California GeoTracker | ~324K records | 29 PFAS | California | CSV download |
| Minnesota MDH | ~247K records | 27 PFAS | Minnesota | Excel on written request (route: `docs/data_requests/mn_mdh_bulk_export.md`) |
| Missouri DNR | ~76K records | 29 PFAS | Missouri | ArcGIS REST |
| New Jersey DEP | 1,313 systems | 25 PFAS | New Jersey | Manual export + API |
| North Carolina DEQ | ~2K records | 5 PFAS (GenX focus) | North Carolina | PDF-parsed tables |
| Ohio EPA | 26,554 records | 6 PFAS | Ohio | ArcGIS REST; excluded from the merged dataset (PWSID overlap with SDWIS) |
| Washington DOH | ~9.3K records | 14 PFAS | Washington | ArcGIS REST |
| NJ Private Wells (PWTA) | — | PFAS, metals | New Jersey | Unavailable (provider removed bulk download) |
| Texas TCEQ | — | PFAS | Texas | Unavailable (manual request only; TX partially covered via WQP) |
| Maine CDC | — | PFAS | Maine | Unavailable |

Auxiliary and task-specific sources:

| Dataset | Records | Content | Coverage | Format |
|---------|---------|---------|----------|--------|
| EPA FRS | 40,828 sites | Facility locations | National | CSV/GeoJSON |
| TRI PFAS | ~2.5K facilities | PFAS releases | National | CSV download |
| DoD PFAS | ~700 sites | Military PFAS sites | National | CSV download |
| EJScreen | 220k+ block groups | Demographic + EJ indices | National | Zenodo archive |
| NLCD | Wall-to-wall 30m | Land use/land cover | National | Raster (GeoTIFF) |
| USGS aquifers | National | Hydrogeology | National | Shapefile/GDB |
| USGS NGA arsenic (T6 target) | Domestic wells | Arsenic | National | CSV (DOI 10.5066/P9JMUAPY) |

### Data caveats

- **Detection limits**: UCMR5 data is 97.1% censored (non-detect). Non-detect values are preserved with a `censored` boolean column — they are never silently dropped.
- **PWSID format**: Public Water System IDs are 9-character strings (e.g., `"CA0101001"`). Never cast to int — leading zeros are meaningful.
- **Large downloads**: NLCD raster is ~1 GB, EJScreen archive is ~6 GB. Use `--skip-large` to skip these during initial testing.
- **NJ DEP**: Chemical sample pages are reCAPTCHA-protected; export them manually with `scripts/scrape_nj_dep.py` (headed browser; no access circumvention). Inventory and facility coordinates download automatically via the waterviewer.nj.gov REST API. 248,107 PFAS records from 1,313 systems across 25 analytes.
- **NJ private wells (PWTA)**: Marked `unavailable` — bulk download removed by provider.
- **MI MPART**: ArcGIS server may return 503 errors intermittently. The loader retries with SSL fallback.
- **NC DEQ**: Source data is PDF-only; parsed values are included in the processed dataset.
- **ZIP code geocoding**: UCMR5/UCMR3 ZIP codes are in separate `*_ZIPCodes.txt` files (not in the main data files). The pipeline joins these by PWSID before geocoding via ZCTA centroids.
- **WQP**: API rejects requests with >3 `characteristicName` values. The loader batches into chunks of ≤3 and merges results. Non-liquid units (e.g., `NG/G`) are filtered.
- **MO DNR**: ArcGIS server is intermittently unavailable (503). Coordinates are returned in Missouri State Plane by default — the loader adds `outSR=4326` for WGS84.
- **OH EPA / WA DOH**: ArcGIS REST API sources; may experience intermittent 503 errors similar to MO DNR.

## Leaderboard

Submit benchmark results by creating a JSON file following the submission schema:

```json
{
  "schema_version": "1.0",
  "model_name": "MyModel",
  "author": "Jane Doe",
  "institution": "University of Example",
  "results": [
    {
      "task": "T1",
      "analyte": "PFOS",
      "metrics": {"auroc": 0.92, "auprc": 0.85, "f1": 0.78}
    }
  ]
}
```

**Expected metric keys per task:**

| Task | Required metrics |
|------|-----------------|
| T1, T4, T5, T6, T7 | `auroc`, `auprc`, `f1` |
| T2 | `rmse`, `mae`, `r2` |
| T3 | `macro_auroc`, `macro_auprc`, `macro_f1`, `hamming_loss`, `subset_accuracy` |

**CLI workflow:**

```bash
# Validate a submission
aquacontam leaderboard validate my_submission.json

# Rank all submissions in a directory
aquacontam leaderboard rank submissions/

# Write leaderboard to file
aquacontam leaderboard publish submissions/ -o LEADERBOARD.md
```

See `configs/submission_template.json` for a complete example.

## Project Structure

```
src/aquacontam/
├── data/           # Data source loaders (UCMR5, UCMR3, WQP, SDWIS, state DBs, TRI, DoD, EPA FRS, EJScreen)
├── preprocessing/  # Detection limit handling, data cleaning, merging
├── geo/            # CRS transforms, spatial indexing, raster stats, geocoding
├── features/       # Feature extractors (proximity, land use, hydrogeology, demographics,
│                   #   multiscale, spatial graph, system characteristics, TRI/DoD proximity, grid)
├── benchmark/      # Task registry, metrics, validation, LORO CV, HPO
├── models/         # BaseModel ABC + 20 model families (29 classes)
├── analysis/       # Ablation, causal deconfounding, equity, spatial autocorrelation,
│                   #   coordinate sensitivity, monitoring bias, regional, sensitivity,
│                   #   interpretability, split strategy comparison
├── calibration/    # Conformal prediction, group conformal coverage
├── inference/      # Prediction utilities
├── export/         # Dataset packaging for Zenodo
├── leaderboard/    # Submission schema, ranking, formatting
├── pipeline/       # Reproducibility pipeline stages (download, preprocess, features,
│                   #   assembly, training, analysis)
└── __main__.py     # CLI entry point
scripts/
├── reproduce.py    # Full reproducibility pipeline
paper/
├── final_gate.py   # Deterministic verification gate (G0-G14)
├── tables/ source_data/  # Display items regenerated from the frozen archive (gate-verified: G4, G14)
results/paper_frozen/  # Frozen results archive (copy of record)
configs/
├── data.yaml               # Data source URLs and column maps
├── experiment.yaml          # Task parameters, model configs, splits
└── submission_template.json # Leaderboard submission example
tests/
├── unit/           # Fast unit tests (2,000+ tests)
└── ...
data/
├── raw/            # Immutable downloaded files
├── interim/        # Intermediate outputs (merged data, cached features)
└── processed/      # Final parquet files
```

**Data flow**: `data/raw/` (immutable) → `data/interim/` → `data/processed/`

## Repository provenance

`results/paper_frozen/` is the frozen, checksummed copy of record for the results
reported in the accompanying paper. The deterministic gate (`paper/final_gate.py`)
re-derives the benchmark artifacts from it in minutes — generated tables, the
leaderboard, input-hash stamps on derived artifacts, archive checksums, and the
per-figure Source Data workbooks — and CI runs the gate on every push and pull
request to main.
`paper/DERIVATIONS.tsv` records the derivation map (artifact ← generator ← inputs).
The manuscript sources themselves (article text, extended data, supplementary
information, and their DOCX builds) are not distributed with this repository; they
accompany the journal submission, and the gate clauses that check them report
`not distributed -- skipped` here. Tables and Source Data in `paper/` are the
paper's display items, regenerated from the frozen archive and verified by the
gate (G4, G14); the article's figures are produced by `paper/generate_figures.py`
from the frozen archive plus the pipeline's interim data, and accompany the
journal article.

## Contributing

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/aquacontam/

# Run tests
pytest tests/unit/
```

Use [conventional commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`. See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidelines.

## Citation

If you use AquaContam in your research, please cite:

```bibtex
@article{newton2026aquacontam,
  title   = {AquaContam: machine-learning models of drinking-water contamination learn who is monitored as much as where contamination occurs},
  author  = {Newton, Tyler J.},
  journal = {Nature Water},
  year    = {2026}
}
```

To cite the software or dataset directly, see [`CITATION.cff`](CITATION.cff).

## License

| Artifact | License | Where |
|---|---|---|
| Source code (src/, scripts/, tests/, paper tooling) | [Apache-2.0](LICENSE) | `LICENSE`, `NOTICE` |
| Compiled dataset (Zenodo archive) | CC BY 4.0 where source terms permit | [`LICENSE-DATA`](LICENSE-DATA), [`docs/DATA_TERMS.md`](docs/DATA_TERMS.md) |
| Per-source raw data | Original agency terms | [`docs/DATA_TERMS.md`](docs/DATA_TERMS.md) |
| TabPFN v2 checkpoint (not redistributed) | Prior Labs License | [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |
| Paper Source Data (`paper/source_data/`) | (c) the author; not covered above | see [Repository provenance](#repository-provenance) |

Built with PriorLabs-TabPFN.
