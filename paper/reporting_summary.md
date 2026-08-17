# Nature Portfolio Reporting Summary

> This document is a prose surrogate that mirrors the content of the official
> Nature Portfolio Reporting Summary; the official form is completed and
> uploaded at submission.

*This document provides information required by the Nature Portfolio reporting
checklist. It accompanies the manuscript "AquaContam: machine-learning models
of drinking-water contamination learn who is monitored as much as where contamination occurs."*

## Statistics

### Statistical tests used

- **DeLong tests** (two-sided) for pairwise AUROC comparisons between model
  families, with Benjamini-Hochberg FDR correction for multiple comparisons.
- **Bootstrap confidence intervals** (1,000 iterations, percentile method)
  for all classification metrics (AUROC, AUPRC, F1, precision, recall).
- **Permutation tests** (10,000 permutations) for environmental justice
  burden ratio significance, including the monitoring-adjusted (stabilized
  inverse-propensity-weighted) burden ratio and the monitoring-intensity
  inequity ratios.
- **Grouped permutation test** (10,000 permutations) for DML category-level
  rank displacement significance, with Benjamini-Hochberg FDR correction
  across feature categories.
- **Rosenbaum sensitivity / E-value analysis** on standardized DML
  coefficients (reported per 1 s.d.) for unmeasured-confounding robustness.
- **Moran's I** with randomization inference (999 permutations) for spatial
  autocorrelation in model residuals.
- **Kolmogorov-Smirnov tests** for distributional shift between EPA regions.
- **Benjamini-Hochberg FDR correction** applied to all multiple comparison
  settings (DeLong tests across model pairs, burden ratios across demographic
  dimensions).

### Justification for statistical tests

All tests are standard for their respective applications. DeLong tests are the
established method for comparing correlated AUROCs on the same test set.
Permutation tests are used where parametric assumptions are not justified
(burden ratios). Bootstrap CIs provide distribution-free interval estimates.

### Sample sizes

Sample sizes were determined by complete data availability from federal and
state monitoring programs. No a priori power analysis was performed; this is
an observational study using the complete population of public water systems
monitored under EPA programs:

- **T1 (PFAS detection)**: 7,930 train / 3,414 validation / 2,846 test systems
- **T4 (Heavy metal)**: 51,283 train / 13,126 validation / 18,911 test systems
- **Total geocoded systems**: <!--pn:ds_geocoded-->87,450<!--/pn-->

### Data exclusions

No monitoring records were excluded on outcome-related grounds. Technical
exclusions apply: the Ohio EPA source (26,554 records) is excluded because its
PWSIDs overlap SDWIS; 7,773 systems without geocodable coordinates (lacking
ZCTA centroids) are excluded from geospatial feature extraction; systems
observed only through single-class detection-only sources cannot contribute to
AUROC where computed; the linguistic-isolation demographic dimension is dropped
from the equity analysis (empty reference group); and EPA regions with fewer
than 30 matched systems are excluded from per-region statistics. These are
technical, not scientific, exclusions.

### Replication

All results are reproducible via the `scripts/reproduce.py` pipeline with
fixed random seed (42). Multi-seed stability analysis (5 seeds) confirms small
single-seed AUROC standard deviations (small for T1 tree models, negligible for T4),
and a tracked, slimmed result snapshot (`results/paper_frozen/`) lets reviewers
verify every reported number against archived results.

### Randomization

Not applicable. This is an observational study of administrative monitoring
data. Geographic stratification (train/validation/test by EPA region) is used
for evaluation, not randomization.

### Blinding

Not applicable. This is an observational study of administrative data with no
experimental intervention.

## Data

### Data availability

All data sources are public records. All but two are directly downloadable;
Minnesota MDH is obtained by written request under the Minnesota Government Data
Practices Act and is not currently redistributable (permission pending), and New
Jersey DEP is accessed via its public WaterViewer portal (public records under the
NJ Open Public Records Act):

| Source | URL / Access |
|--------|-------------|
| UCMR5 | https://www.epa.gov/dwucmr |
| UCMR3 | https://www.epa.gov/dwucmr |
| SDWIS | EPA ECHO system |
| EPA FRS | https://www.epa.gov/frs |
| EJScreen | EPA (archived on Zenodo) |
| NLCD 2021 | https://doi.org/10.5066/P9JZ7AO3 |
| USGS Aquifers | https://water.usgs.gov/ogw/aquifer/map.html |
| State DBs (MI, CA, NJ, NC) | MI MPART, CA GeoTracker, NJ DEP (WaterViewer portal, NJ OPRA public records), NC DEQ |
| MN MDH | Minnesota Dept. of Health (written request under MN Government Data Practices Act; not redistributable, permission pending) |
| WQP | USGS/EPA Water Quality Portal (https://www.waterqualitydata.us) |
| MO DNR | Missouri Dept. of Natural Resources (ArcGIS REST API) |
| OH EPA | Ohio EPA PFAS sampling (ArcGIS REST API) |
| WA DOH | Washington Dept. of Health PFAS monitoring (CSV) |
| TRI | EPA Toxics Release Inventory (https://www.epa.gov/toxics-release-inventory-tri-program) |
| DoD PFAS Sites | EPA PFAS Analytic Tools (federal sites database) |
| Census ZCTA | US Census Bureau Gazetteer Files |

### Data collection

All data were obtained from public federal and state environmental monitoring
programs, most by direct download, with Minnesota MDH by written request. New
Jersey DEP chemical-sample records were collected with a headed-browser scraper that
reads the same public WaterViewer pages a browser would (public records under the NJ
Open Public Records Act; no authentication or access circumvention). No primary data
were generated: every record is a public agency measurement, not a new observation
collected for this study. Download dates and URLs are recorded in `configs/data.yaml`.

### Data processing

Data processing steps are fully documented in the Methods section and
implemented in the open-source codebase. Key steps: schema harmonization,
unit conversion, censored value retention, ZIP code geocoding, geospatial
feature extraction. Raw data are never modified (`data/raw/` is immutable).

## Code

### Code availability

- **Repository**: https://github.com/tjnewton/aquacontam
- **License**: Apache License 2.0 (code); CC BY 4.0 where source terms permit (compiled data)
- **Archive**: Zenodo (<!-- TODO: insert Zenodo DOI before submission -->DOI: [to be inserted before submission])
- **Software versions**: Recorded in `results/paper_frozen/software_versions.json` (committed, checksummed)
- **Reproducibility**: `scripts/reproduce.py` regenerates all results from
  raw data, plus the request-based Minnesota staging (Data Availability), with
  fixed random seed (42)

### Software dependencies

Python >= 3.10 with: NumPy, pandas, scikit-learn, XGBoost, LightGBM,
CatBoost, PyTorch, GeoPandas, SHAP, Optuna. Full specifications in
`environment.yml` (conda) and `pyproject.toml` (pip).

## Reproducibility

- Fixed random seed (42) for all stochastic operations
- Software versions logged alongside results
- `scripts/reproduce.py` pipeline generates all figures, tables, and results
  from raw data downloads plus the request-based Minnesota staging
- Multi-seed stability analysis confirms robustness (small T1 tree-model across-seed SD)
- Leave-one-region-out cross-validation provides an independent robustness
  check (T1 mean AUROC <!--pn:loro_full_auroc-->0.782<!--/pn--> with the full feature set;
  <!--pn:loro_pf_auroc-->0.691<!--/pn--> provenance-reduced)
