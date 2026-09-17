# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **Citation metadata**: the preferred citation in `CITATION.cff`, the README and leaderboard
  BibTeX, and the export dataset card now cite the EarthArXiv preprint of the accompanying
  paper (doi:10.31223/X51V3G) until the peer-reviewed article is published

### Fixed

- **CI typecheck (pandas-stubs 3.0.5.260914 regression)**: the 2026-09-14 release repeats the 3.0.5.260730 overload bug under the project's numpy<2 pin (`Series.astype(str)` resolves to `Series[bool]`; the same 23 `[misc]` errors across 9 data loaders, with no change to our source). Excluded that release as well (`!=3.0.5.260914`); the constraint still resolves to 3.0.3.260530.

## [4.0.0] - 2026-08-20

### Changed

- **Dataset scale**: 95,223 U.S. water systems and ambient monitoring locations, 4,860,171
  samples across fifteen data sources (ZIP+4 geocoding, NJ DEP, Minnesota MDH in-split, SDWIS
  keep-parse: 916,899 lead/copper records with zero-measure rows retained as non-detects)
- **Canonical v2 re-freeze**: the entire frozen paper archive regenerated from ONE pinned
  DAG run (base + provenance-free passes + standalone stages + post-freeze generators),
  with input-hash derivation stamps and pre-registered acceptance checks (all hold)
- **Deterministic paper gate G0-G14**: derivation-staleness (G8), embedded-prose audit
  coverage, concept-consistency registry, frozen checksums, DOCX source-manifest
  provenance embeds (G12)
- **Publication metadata**: finalized author and citation metadata across `CITATION.cff`,
  `.zenodo.json`, `pyproject.toml`, and the README citation.
- **Benchmark-focused distribution**: paper figures and Sphinx documentation sources are
  not distributed with the benchmark repository (figures are regenerable with
  `paper/generate_figures.py`; tables, the leaderboard, and per-figure Source Data remain
  gate-verified). The `webapp` CLI subcommand now registers only when the optional webapp
  package is present.

### Fixed

- **CI typecheck (pandas-stubs 3.0.5.260730 regression)**: that release resolves `Series.astype(str)` to an overload returning `Series[bool]`, so every downstream `.str` accessor became `SeriesStringMethods[bool]` — 23 `[misc]` errors across 9 data loaders, with no change to our source. Excluded the single broken release (`!=3.0.5.260730`) rather than adding 23 casts for a third-party stub bug; `astype("string")` still yields `Series[str]` in the same release, which is what identifies it as a stub defect.
- **Source-census accuracy**: MI MPART analyte count (7→5 PFAS), OH EPA record count (1,184→26,554), specific analyte counts added for CA GeoTracker (29), WA DOH (14), MO DNR (29), OH EPA (6)
- **Monitoring AUPRC significance**: T1 monitoring ablation now correctly reported as statistically significant (Δ=−0.020, p_FDR=0.021) instead of non-significant
- **Display-item cross-references**: supplementary table references normalized
- **System count**: Updated "160,000" → "150,000" public water systems (conservative EPA estimate)
- **NJ DEP source metadata**: Updated analyte description from "PFNA, PFOA, PFOS" to "25 PFAS" with correct API source
- **Mypy type checking**: Fixed all 40 mypy errors across 15 files; CI typecheck job now blocks on failure
- **Exception handling**: Narrowed 21 bare `except Exception:` handlers to specific exception types across 16 source files
- **Deprecated `retrying` dependency**: Replaced custom retry loops with `tenacity` in ArcGIS data loader
- **Benchmark task table**: primary metric corrected to AUPRC (T1, T4-T7) / macro-AUPRC (T3), matching the task registry and submission schema; T6 described as the arsenic transfer probe
- **Leaderboard citation**: BibTeX key unified with the README (`newton2026aquacontam`); stale status note removed
- **Paper test wiring**: data-consistency tests repointed at the frozen archive (`results/paper_frozen/`), reviving 15 dormant assertions; per-module dependency guards replace the blanket conftest skip, so the paper tests run in CI

### Added

- **Analysis note**: negligible val-reuse effect (0.8pp AUROC) vs geographic heterogeneity (14.5pp gap)
- **CI pipeline test job**: New `test-pipeline` job runs 3 previously-skipped test files (pipeline analysis, pipeline core, reproduce stages)
- **Checksum validation**: Downloaded data files verified against `data/checksums.sha256` at end of pipeline download stage
- **Model weight integrity**: SHA-256 hash computed on export and verified before `joblib.load()` on import; `verify_model_integrity()` function for batch checking
- **Detection limit sensitivity**: `detection_limit_sensitivity()` function in `analysis/sensitivity.py`; `DEFAULT_DL_FILLNA` constant extracted in `benchmark/tasks.py`
- **`--with-uncertainty` CLI flag**: Enables LORO CV and bootstrap CIs together; `--all` now includes LORO by default
- **Substitution timing documentation**: Module docstrings in `detection_limits.py` and `preprocess.py` clarify binary detection target design choice
- **`SECURITY.md`**: vulnerability-disclosure policy (GitHub Security Advisories)
- **Zenodo archive**: dataset and code snapshot deposited at [10.5281/zenodo.22073200](https://doi.org/10.5281/zenodo.22073200)

## [3.0.0] - 2026-03-03

### Added

- **6 new data source loaders**: WQP (Water Quality Portal), MO DNR (Missouri), OH EPA (Ohio), WA DOH (Washington), TRI PFAS (Toxics Release Inventory), DoD PFAS (Department of Defense sites)
- **3 new feature modules**: DoD proximity features, TRI proximity features, system characteristics (service connections, population, source water type)
- **Shared data utilities**: `_arcgis.py` (ArcGIS REST pagination + SSL retry), `_download.py` (shared download/extract), `_utils.py` (column finding, PWSID normalization, unit conversion)
- **Full pipeline rerun**: expanded dataset to 10,746 systems with 12 PFAS analytes across all sources
- **109 new unit tests**: 1,229 → 1,338 total

### Fixed

- WQP API batching: requests with >3 `characteristicName` values now batched into chunks of ≤3
- MO DNR CRS: added `outSR=4326` parameter to get WGS84 coordinates (was Missouri State Plane)
- MO DNR detection limits: `DETECT_LMT=0` for detected samples no longer flagged as invalid
- WQP monitoring locations: skip PWSID validation for non-PWS sites (`skip_pwsid_check=True`)

## [2.0.0] - 2026-02-27

### Added

- **Sphinx API docs**: Complete coverage for all modules — added `calibration.rst`, `inference.rst`, `export.rst`; expanded `models.rst` (+4 modules) and `analysis.rst` (+6 submodules); updated `index.rst` toctree
- **README usage examples**: 6 code examples covering XGBoost, Deep Tobit, Tobit, TabPFN, Voting Ensemble, and Conformal Classifier
- **environment.yml**: Added `foundation` extra, `pyg-lib`, and `torch-geometric` conda deps
- **11 new model families**: CatBoost, LightGBM, TabPFN v2 (foundation model), Deep Tobit MLP (censoring-aware), Tobit regressor, AFT regressor, GNN, Logistic Regression, Dummy baseline, Voting Ensemble, Stacking Ensemble — total 15 families (23 classes)
- **Analysis modules**: feature ablation, causal deconfounding (double ML), coordinate sensitivity, regional analysis, split strategy comparison, spatial autocorrelation (Moran's I)
- **Calibration**: conformal prediction, group conformal coverage guarantees
- **Benchmark enhancements**: leave-one-region-out (LORO) cross-validation, Optuna hyperparameter optimization, external validation, threshold optimization, DeLong statistical tests
- **System characteristics features**: service connections, population served, source water type
- **Grid features**: latitude/longitude grid cells for spatial encoding
- **Paper**: 5 main figures, 9 extended data figures, 11+ extended data tables, DOCX conversion pipeline
- **Dependency groups**: `[boost]` (LightGBM, CatBoost), `[hpo]` (Optuna), `[interpret]` (SHAP, lifelines), `[foundation]` (TabPFN v2)

### Fixed

- Deep Tobit regressor: truncated mean overflow (asymptotic z-value approximation, log-space clamping, output clamping to 2× training max)
- Deep Tobit: default detection limit 2.0 → 0.004 μg/L, censoring metadata passthrough from benchmark tasks
- DML causal deconfounding: HC1 sandwich standard errors (heteroscedasticity-robust)
- T3 multilabel: corrected multi-label classification pipeline
- T4 target: use action-level exceedance (29.1% positive) instead of detection (100%)
- CI: lowered coverage threshold to 65% for partial-extras CI runs

## [1.0.0] - 2026-02-20

### Added

- **9 data sources**: UCMR5 (~1.9M samples, 29 PFAS + lithium), UCMR3 (~1.1M samples, 38 contaminants), SDWIS (lead/copper/arsenic), EPA FRS (40k facilities), EJScreen (block group demographics), MI MPART, CA GeoTracker, NJ DEP, NC DEQ
- **7 benchmark tasks**: T1 (binary PFAS detection), T2 (concentration regression), T3 (multi-PFAS profile), T4 (heavy metal detection), T5 (cross-contaminant transfer), T6 (private well extrapolation), T7 (temporal UCMR3-to-UCMR5 prediction)
- **4 model families**: XGBoost, Random Forest, MLP, CNN1D — each with classifier and regressor variants
- **Feature extractors**: facility proximity (industrial, military, WWTP, airport, landfill), NLCD land use fractions, USGS aquifer hydrogeology, EJScreen demographics
- **Environmental justice analysis**: burden ratios, group-stratified metrics, disparity reports across demographic subgroups
- **Interactive risk map**: Dash webapp with Scattermap (MapLibre), task/analyte selection, EJ overlay, ZIP search
- **Reproducibility pipeline**: `scripts/reproduce.py` with download, preprocess, geocode, feature extraction, train, evaluate, and export stages
- **Click CLI**: `aquacontam webapp`, `aquacontam benchmark`, `aquacontam export`, `aquacontam leaderboard` subcommands
- **Dataset export**: Zenodo-ready directory with geographic splits, feature files, manifest, and checksums
- **Leaderboard**: JSON submission schema, per-task ranking, markdown/JSON output, GitHub Actions validation
- **Paper infrastructure**: 8 publication figures, 6 tables (CSV + LaTeX + markdown), Nature Water style
- **Geographic stratification**: EPA region-based train/val/test splits to prevent spatial leakage
- **ZIP code geocoding**: ZCTA centroid lookup for 67,594 water systems
- **Target leakage prevention**: automatic removal of derived monitoring columns from feature matrices
- **Zenodo metadata export**: dataset card, analyte info, and feature column metadata in dataset archive

### Fixed

- T4 heavy metal task: action-level exceedance target (was single-class, now AUROC 0.950)
- T5 cross-contaminant transfer: single-class source guard prevents degenerate evaluation
- SDWIS censoring: use `<` prefix (not `L`) for left-censored values
- EJScreen: column renames for 2024 schema + block group gazetteer geocoding
- Deep learning models: store `feature_names_in_` for prediction column alignment
- Webapp geocoding: full ZCTA centroid lookup (58 → 33k ZIPs)
- Feature assembly: deduplicate columns from `pd.get_dummies` with mixed NaN types
- Plotly migration: `Scattermapbox` → `Scattermap`, `layout.mapbox` → `layout.map` (MapLibre)
