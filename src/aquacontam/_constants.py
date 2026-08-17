"""Project-wide constants for AquaContam.

Centralizes magic values used across data loaders, preprocessing,
and tests. Import from here rather than hard-coding values.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------

CRS_STORAGE: str = "EPSG:4326"
"""WGS 84 — used for all stored geometries (lat/lon)."""

CRS_DISTANCE: str = "EPSG:5070"
"""Conus Albers Equal Area — used for distance calculations (meters)."""

# ---------------------------------------------------------------------------
# Public Water System ID (PWSID)
# ---------------------------------------------------------------------------

PWSID_LENGTH: int = 9
"""PWSID is always a 9-character string (e.g. 'CA0101001'). Never cast to int."""

# ---------------------------------------------------------------------------
# Water-quality source modality
# ---------------------------------------------------------------------------

AMBIENT_WQ_SOURCES: frozenset[str] = frozenset({"ca_geotracker", "wqp"})
"""Ambient environmental-monitoring sources (groundwater wells / surface-water sites).

These sample a different population than public-water-system (PWS) finished-water
compliance monitoring, so they are excluded from the main benchmark train/val/test
splits and reserved for the external-validation framework (which tests cross-source
transfer explicitly). The main benchmark uses PWS finished-water/compliance sources
(UCMR5, UCMR3, SDWIS, and the PWS state databases). See
``benchmark/external_validation.py`` and ``pipeline/training.py``.
"""

# ---------------------------------------------------------------------------
# Standard schema columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: tuple[str, ...] = (
    "pwsid",
    "analyte",
    "concentration",
    "unit",
    "censored",
    "detection_limit",
    "sample_date",
    "latitude",
    "longitude",
)
"""Columns every water-quality DataFrame must contain."""

# ---------------------------------------------------------------------------
# Detection limit substitution factors
# ---------------------------------------------------------------------------

DL_SUBSTITUTION_FACTOR_HALF: float = 0.5
"""Simple substitution: replace non-detect with DL / 2."""

DL_SUBSTITUTION_FACTOR_SQRT2: float = 0.7071067811865476
"""DL / sqrt(2) substitution (Hornung & Reed, 1990)."""

# ---------------------------------------------------------------------------
# CONUS bounding box (for coordinate validation)
# ---------------------------------------------------------------------------

CONUS_LAT_MIN: float = 24.396308
"""Southernmost point of contiguous US (Key West, FL)."""

CONUS_LAT_MAX: float = 49.384358
"""Northernmost point of contiguous US (Northwest Angle, MN)."""

CONUS_LON_MIN: float = -124.848974
"""Westernmost point of contiguous US (Cape Flattery, WA)."""

CONUS_LON_MAX: float = -66.885444
"""Easternmost point of contiguous US (Quoddy Head, ME)."""

# ---------------------------------------------------------------------------
# UCMR5 analytes (29 PFAS + lithium = 30 total)
# ---------------------------------------------------------------------------

UCMR5_ANALYTES: tuple[str, ...] = (
    # EPA Method 533 (25 PFAS)
    "11Cl-PF3OUdS",
    "9Cl-PF3ONS",
    "ADONA",
    "HFPO-DA",
    "NFDHA",
    "PFBA",
    "PFBS",
    "8:2 FTS",
    "PFDA",
    "PFDoA",
    "PFEESA",
    "PFHpS",
    "PFHpA",
    "4:2 FTS",
    "PFHxS",
    "PFHxA",
    "PFMPA",
    "PFMBA",
    "PFNA",
    "6:2 FTS",
    "PFOS",
    "PFOA",
    "PFPeA",
    "PFPeS",
    "PFUnA",
    # EPA Method 537.1 (4 PFAS)
    "NEtFOSAA",
    "NMeFOSAA",
    "PFTA",
    "PFTrDA",
    # Non-PFAS
    "lithium",
)
"""All 30 analytes monitored under UCMR5 (29 PFAS + lithium)."""

# ---------------------------------------------------------------------------
# UCMR3 analytes (6 PFAS)
# ---------------------------------------------------------------------------

UCMR3_ANALYTES: tuple[str, ...] = (
    "PFOA",
    "PFOS",
    "PFNA",
    "PFHxS",
    "PFBS",
    "PFHpA",
)
"""The 6 PFAS compounds monitored under UCMR3."""

# ---------------------------------------------------------------------------
# NLCD land cover class codes
# ---------------------------------------------------------------------------

NLCD_WATER: frozenset[int] = frozenset({11})
"""Open water."""

NLCD_ICE_SNOW: frozenset[int] = frozenset({12})
"""Perennial ice/snow."""

NLCD_DEVELOPED: frozenset[int] = frozenset({21, 22, 23, 24})
"""Developed: open space (21), low (22), medium (23), high intensity (24)."""

NLCD_BARREN: frozenset[int] = frozenset({31})
"""Barren land (rock/sand/clay)."""

NLCD_FOREST: frozenset[int] = frozenset({41, 42, 43})
"""Forest: deciduous (41), evergreen (42), mixed (43)."""

NLCD_SHRUB: frozenset[int] = frozenset({52})
"""Shrub/scrub."""

NLCD_GRASSLAND: frozenset[int] = frozenset({71})
"""Grassland/herbaceous."""

NLCD_AGRICULTURE: frozenset[int] = frozenset({81, 82})
"""Agriculture: pasture/hay (81), cultivated crops (82)."""

NLCD_WETLAND: frozenset[int] = frozenset({90, 95})
"""Wetland: woody (90), emergent herbaceous (95)."""

NLCD_CLASS_MAP: dict[int, str] = {
    11: "water",
    12: "ice_snow",
    21: "developed",
    22: "developed",
    23: "developed",
    24: "developed_high",
    31: "barren",
    41: "forest",
    42: "forest",
    43: "forest",
    52: "shrub",
    71: "grassland",
    81: "agriculture",
    82: "agriculture",
    90: "wetland",
    95: "wetland",
}
"""Map individual NLCD class codes to grouped category names."""

# ---------------------------------------------------------------------------
# EPA FRS facility type codes (SIC / NAICS / interest types)
# ---------------------------------------------------------------------------

FRS_INTEREST_TYPES: dict[str, list[str]] = {
    "industrial": [
        "CERCLIS",  # Superfund sites
        "RCRAINFO",  # Hazardous waste handlers
        "TRIS",  # Toxic Release Inventory
    ],
    "military": [
        "MILITARY",
    ],
    "airport": [
        "AIRPORT",
    ],
    "wwtp": [
        "NPDES",  # National Pollutant Discharge Elimination System
    ],
    # NOTE: RCRAINFO overlap with "industrial" is intentional — RCRA covers both
    # hazardous waste handlers and landfills. Features are computed per-type
    # independently (separate columns), so there is no double-counting.
    "landfill": [
        "RCRAINFO",  # Many landfills are RCRA-regulated
    ],
}
"""EPA FRS interest type strings used to classify facilities."""

FRS_SIC_CODES: dict[str, list[str]] = {
    "industrial": [
        "2281",  # Yarn throwing and winding mills (textiles — PFAS in stain repellents)
        "2295",  # Coated fabrics, not rubberized (textiles — PFAS coatings)
        "2650",  # Paperboard containers and boxes (PFAS in food packaging)
        "2670",  # Converted paper and paperboard (PFAS grease-proofing)
        "2819",  # Industrial inorganic chemicals
        "2869",  # Industrial organic chemicals
        "2911",  # Petroleum refining
        "3312",  # Steel works and blast furnaces
        "3339",  # Primary smelting/refining of nonferrous metals
        "3471",  # Electroplating, plating, polishing (chrome plating — PFAS mist suppressants)
        "3674",  # Semiconductors and related devices (PFAS in photolithography)
    ],
    "wwtp": ["4952"],  # Sewerage systems
    "landfill": ["4953"],  # Refuse systems (landfills)
    "airport": ["4512", "4581"],  # Air transportation, airports
    "military": ["9711"],  # National security (military)
}
"""SIC codes for facility type classification."""

FRS_NAICS_CODES: dict[str, list[str]] = {
    "wwtp": ["221320"],  # Sewage treatment facilities
    "landfill": ["562212"],  # Solid waste landfill
    "airport": ["481111", "488119"],  # Air transport, airport operations
}
"""NAICS codes for facility type classification."""

# ---------------------------------------------------------------------------
# Default feature extraction buffer radii (meters)
# ---------------------------------------------------------------------------

DEFAULT_PROXIMITY_RADII: tuple[float, ...] = (1000.0, 5000.0, 10000.0)
"""Default radii for proximity count features (meters)."""

DEFAULT_LAND_USE_BUFFERS: tuple[float, ...] = (1000.0, 5000.0)
"""Default buffer radii for NLCD land use fraction extraction (meters)."""

# ---------------------------------------------------------------------------
# Download settings
# ---------------------------------------------------------------------------

DOWNLOAD_CHUNK_SIZE: int = 8192
"""Bytes per read chunk for streaming downloads."""

DOWNLOAD_TIMEOUT_DEFAULT: int = 300
"""Default HTTP timeout in seconds for data downloads."""

DOWNLOAD_TIMEOUT_LARGE: int = 600
"""Extended HTTP timeout for large file downloads (FRS, NLCD, shapefiles)."""

# ---------------------------------------------------------------------------
# Unit conversion factors
# ---------------------------------------------------------------------------

PPT_TO_UGL: float = 0.001
"""1 ppt (ng/L) = 0.001 ug/L."""

MGL_TO_UGL: float = 1000.0
"""1 mg/L = 1000 ug/L."""

# ---------------------------------------------------------------------------
# EPA regions (10 regions, mapping to state 2-letter codes)
# ---------------------------------------------------------------------------

EPA_REGIONS: dict[int, tuple[str, ...]] = {
    1: ("CT", "MA", "ME", "NH", "RI", "VT"),
    2: ("NJ", "NY", "PR", "VI"),
    3: ("DC", "DE", "MD", "PA", "VA", "WV"),
    4: ("AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"),
    5: ("IL", "IN", "MI", "MN", "OH", "WI"),
    6: ("AR", "LA", "NM", "OK", "TX"),
    7: ("IA", "KS", "MO", "NE"),
    8: ("CO", "MT", "ND", "SD", "UT", "WY"),
    9: ("AZ", "CA", "GU", "HI", "NV"),
    10: ("AK", "ID", "OR", "WA"),
}
"""EPA region number → tuple of 2-letter state/territory codes."""

STATE_TO_EPA_REGION: dict[str, int] = {
    state: region for region, states in EPA_REGIONS.items() for state in states
}
"""2-letter state code → EPA region number (reverse of EPA_REGIONS)."""

# ---------------------------------------------------------------------------
# SDWIS contaminant codes (heavy metals)
# ---------------------------------------------------------------------------

HEAVY_METAL_ANALYTES: tuple[str, ...] = ("lead", "copper")
"""Heavy metal analytes tracked in SDWIS data (LCR 90th-percentile results)."""

SDWIS_CONTAMINANT_CODES: dict[str, str] = {
    "lead": "PB90",
    "copper": "CU90",
}
"""Analyte name → SDWIS contaminant code mapping.

EPA ECHO LCR bulk data uses ``PB90``/``CU90`` for the 90th-percentile
lead/copper results.
"""

SDWIS_CODE_TO_ANALYTE: dict[str, str] = {v: k for k, v in SDWIS_CONTAMINANT_CODES.items()}
"""Reverse mapping: SDWIS contaminant code → analyte name."""

LCR_ACTION_LEVELS: dict[str, float] = {
    "lead": 15.0,  # µg/L (EPA action level)
    "copper": 1300.0,  # µg/L (EPA action level)
}
"""EPA Lead and Copper Rule action levels in µg/L.

SDWIS LCR data reports 90th-percentile compliance results that are almost
never censored — ``RESULT_SIGN_CODE`` is rarely ``"<"``.  Using binary
detection as the target yields a single-class problem.  Instead, predict
whether the 90th-percentile result **exceeds the EPA action level**.
"""

ACTION_LEVELS_UGL: dict[str, float] = {
    **LCR_ACTION_LEVELS,
    "arsenic": 10.0,  # µg/L (EPA arsenic MCL)
}
"""EPA regulatory exceedance thresholds (µg/L) for ``action_level`` targets.

Superset of :data:`LCR_ACTION_LEVELS`: lead/copper are Lead and Copper Rule
action levels; arsenic is the Maximum Contaminant Level (Arsenic Rule). Used by
``aggregate_to_system_level(target="action_level")`` to label a well/system as
positive when its maximum measured concentration meets or exceeds the threshold.
Arsenic backs benchmark task T6 (private-well risk; see ``data/nga_arsenic.py``).
"""

# ---------------------------------------------------------------------------
# State database analytes
# ---------------------------------------------------------------------------

MI_MPART_ANALYTES: tuple[str, ...] = (
    "HFPO-DA",
    "PFBS",
    "PFHxA",
    "PFHxS",
    "PFNA",
    "PFOA",
    "PFOS",
)
"""Michigan MPART regulated PFAS compounds."""

NJ_DEP_ANALYTES: tuple[str, ...] = (
    "11Cl-PF3OUdS",
    "4:2 FTS",
    "6:2 FTS",
    "8:2 FTS",
    "9Cl-PF3ONS",
    "HFPO-DA",
    "PFBA",
    "PFBS",
    "PFDA",
    "PFDoA",
    "PFEESA",
    "PFHpA",
    "PFHpS",
    "PFHxA",
    "PFHxS",
    "PFMBA",
    "PFMPA",
    "PFNA",
    "PFOA",
    "PFOS",
    "PFPeA",
    "PFPeS",
    "PFTA",
    "PFTrDA",
    "PFUnA",
)
"""New Jersey DEP PFAS compounds tracked in waterviewer.nj.gov (30 analyte codes)."""

NJ_DEP_ANALYTE_CODES: dict[int, str] = {
    2800: "PFAS RULE",
    2801: "PFBS",
    2802: "PFHpA",
    2803: "PFHxS",
    2804: "PFNA",
    2805: "PFOS",
    2806: "PFOA",
    2807: "PFDA",
    2808: "PFDoA",
    2809: "PFHxA",
    2810: "PFTA",
    2811: "PFTrDA",
    2812: "PFUnA",
    2813: "11Cl-PF3OUdS",
    2814: "9Cl-PF3ONS",
    2816: "HFPO-DA",
    2819: "PFBA",
    2820: "6:2 FTS",
    2821: "4:2 FTS",
    2822: "8:2 FTS",
    2823: "PFMPA",
    2824: "PFPeA",
    2825: "PFMBA",
    2826: "PFEESA",
    2828: "PFPeS",
    2829: "PFHpS",
    2830: "TOTAL PFOA AND PFOS",
    2840: "HAZARD INDEX PFAS",
}
"""NJ DEP analyte code → standard short name mapping (from waterviewer.nj.gov /LookUp/AnalyteCodes)."""

NC_DEQ_ANALYTES: tuple[str, ...] = ("HFPO-DA", "PFOA", "PFOS", "PFNA", "PFHxS")
"""North Carolina DEQ primary PFAS compounds (GenX focus)."""

MO_DNR_ANALYTES: tuple[str, ...] = (
    "11Cl-PF3OUdS",
    "4:2 FTS",
    "6:2 FTS",
    "8:2 FTS",
    "9Cl-PF3ONS",
    "ADONA",
    "HFPO-DA",
    "NEtFOSAA",
    "NFDHA",
    "NMeFOSAA",
    "PFBA",
    "PFBS",
    "PFDA",
    "PFDoA",
    "PFEESA",
    "PFHpA",
    "PFHpS",
    "PFHxA",
    "PFHxS",
    "PFMBA",
    "PFMPA",
    "PFNA",
    "PFOA",
    "PFOS",
    "PFPeA",
    "PFPeS",
    "PFTA",
    "PFTrDA",
    "PFUnA",
)
"""Missouri DNR PFAS compounds monitored in public water systems (normalized short codes)."""

WQP_PFAS_ANALYTES: tuple[str, ...] = (
    "HFPO-DA",
    "PFBA",
    "PFBS",
    "PFDA",
    "PFHpA",
    "PFHpS",
    "PFHxA",
    "PFHxS",
    "PFNA",
    "PFOA",
    "PFOS",
    "PFPeA",
    "PFUnA",
)
"""PFAS analytes queried from the Water Quality Portal."""

MN_MDH_ANALYTES: tuple[str, ...] = (
    "11Cl-PF3OUdS",
    "4:2 FTS",
    "6:2 FTS",
    "8:2 FTS",
    "9Cl-PF3ONS",
    "ADONA",
    "FOSA",
    "HFPO-DA",
    "NEtFOSAA",
    "NFDHA",
    "PFBA",
    "PFBS",
    "PFDA",
    "PFDoA",
    "PFEESA",
    "PFHpA",
    "PFHpS",
    "PFHxA",
    "PFHxS",
    "PFMBA",
    "PFMPA",
    "PFNA",
    "PFOA",
    "PFOS",
    "PFPeA",
    "PFPeS",
    "PFUnA",
)
"""Minnesota MDH PFAS compounds monitored in public water systems (normalized short codes).

``FOSA`` (perfluorooctane sulfonamide) is MN-only — it has no equivalent in the
other sources' analyte vocabularies and will not cross-link with them.
"""

# ---------------------------------------------------------------------------
# Benchmark evaluation metrics
# ---------------------------------------------------------------------------

CLASSIFICATION_METRICS: tuple[str, ...] = (
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "balanced_accuracy",
    "brier_score",
    "ece",
)
"""Standard classification metrics for benchmark tasks."""

REGRESSION_METRICS: tuple[str, ...] = (
    "rmse",
    "mae",
    "r2",
    "explained_variance",
    "median_ae",
)
"""Standard regression metrics for benchmark tasks."""

# ---------------------------------------------------------------------------
# Benchmark task default analytes
# ---------------------------------------------------------------------------

T1_DEFAULT_ANALYTES: tuple[str, ...] = ("PFOS", "PFOA", "PFBS", "PFHxS", "HFPO-DA")
"""T1 (binary PFAS detection) — analytes with highest detection rates in UCMR5."""

T2_DEFAULT_ANALYTES: tuple[str, ...] = ("PFOS", "PFOA", "PFHxS")
"""T2 (concentration regression) — analytes with enough detects for regression."""

T3_DEFAULT_ANALYTES: tuple[str, ...] = ("PFOS", "PFOA", "PFBS", "PFHxS", "HFPO-DA")
"""T3 (multi-PFAS profile) — analytes for multilabel classification."""

T4_DEFAULT_ANALYTES: tuple[str, ...] = ("lead", "copper")
"""T4 (heavy metal prediction) — SDWIS heavy metal analytes (LCR)."""

# ---------------------------------------------------------------------------
# System-level aggregation columns
# ---------------------------------------------------------------------------

SYSTEM_AGG_STATS: tuple[str, ...] = (
    "max_concentration",
    "any_detected",
    "detection_rate",
    "n_samples",
    "mean_detection_limit",
)
"""Columns produced by system-level aggregation of water quality data."""

# ---------------------------------------------------------------------------
# EJScreen demographic and EJ index columns
# ---------------------------------------------------------------------------

EJSCREEN_DEMOGRAPHIC_COLS: tuple[str, ...] = (
    "pct_people_of_color",
    "pct_low_income",
    "pct_less_hs_education",
    "pct_limited_english",
    "pct_under_5",
    "pct_over_64",
)
"""Demographic indicator columns from EJScreen block group data."""

EJSCREEN_EJ_INDEX_COLS: tuple[str, ...] = (
    "ej_index_water",
    "ej_supplemental_index",
    "prox_wastewater_discharge",
    "prox_superfund",
    "prox_hazardous_waste",
    "prox_rmp_facility",
)
"""Environmental justice index columns from EJScreen block group data."""

# ---------------------------------------------------------------------------
# T5 cross-contaminant transfer directions
# ---------------------------------------------------------------------------

T5_TRANSFER_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("lead", "PFOS"),
    ("PFOS", "lead"),
)
"""Default (source_analyte, target_analyte) pairs for T5 cross-contaminant transfer."""

# ---------------------------------------------------------------------------
# T6 private well risk extrapolation
# ---------------------------------------------------------------------------

T6_DEFAULT_ANALYTES: tuple[str, ...] = ("arsenic",)
"""Default analytes for T6 private well risk extrapolation task."""

CONUS_GRID_RESOLUTION_KM: float = 10.0
"""Default resolution (km) for CONUS risk surface grid generation."""

# ---------------------------------------------------------------------------
# Webapp defaults
# ---------------------------------------------------------------------------

WEBAPP_DEFAULT_PORT: int = 8050
"""Default port for the Dash risk map webapp."""

WEBAPP_MAX_DISPLAY_POINTS: int = 50_000
"""Maximum number of grid points to render on the map at once."""

# ---------------------------------------------------------------------------
# Leaderboard / submission schema
# ---------------------------------------------------------------------------

T7_SHARED_ANALYTES: tuple[str, ...] = ("PFOS", "PFOA", "PFNA", "PFHxS", "PFBS", "PFHpA")
"""The 6 PFAS analytes shared between UCMR3 and UCMR5."""

T7_DEFAULT_ANALYTES: tuple[str, ...] = ("PFOS", "PFOA", "PFBS", "PFHxS")
"""T7 (temporal prediction) — default analytes for UCMR3→UCMR5 task."""

# ---------------------------------------------------------------------------
# Multilabel benchmark metrics (T3)
# ---------------------------------------------------------------------------

MULTILABEL_METRICS: tuple[str, ...] = (
    "subset_accuracy",
    "hamming_loss",
    "macro_auroc",
    "micro_auroc",
    "macro_auprc",
    "micro_auprc",
    "macro_f1",
    "micro_f1",
    "label_ranking_avg_precision",
    "mean_per_label_auroc",
)
"""Multilabel classification metrics for T3 benchmark task."""

# ---------------------------------------------------------------------------
# Leaderboard / submission schema
# ---------------------------------------------------------------------------

SUBMISSION_SCHEMA_VERSION: str = "1.0"
"""Current version of the leaderboard submission JSON schema."""

DATASET_VERSION: str = "4.0.0"
"""Current version of the AquaContam benchmark dataset (matches CITATION.cff/.zenodo.json)."""

# ---------------------------------------------------------------------------
# Publication-ready model name mapping (code name → display name)
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "dummy": "Dummy",
    "dummy_classifier": "Dummy",
    "logistic_regression": "Logistic Reg.",
    "xgboost": "XGBoost",
    "xgboost_classifier": "XGBoost",
    "xgboost_regressor": "XGBoost",
    "xgboost_default": "XGBoost (default)",
    "random_forest": "Random Forest",
    "random_forest_classifier": "Random Forest",
    "random_forest_regressor": "Random Forest",
    "mlp": "MLP",
    "mlp_classifier": "MLP",
    "mlp_regressor": "MLP",
    "cnn1d": "CNN1D",
    "cnn1d_classifier": "CNN1D",
    "cnn1d_regressor": "CNN1D",
    "lightgbm_classifier": "LightGBM",
    "lightgbm_regressor": "LightGBM",
    "catboost_classifier": "CatBoost",
    "catboost_regressor": "CatBoost",
    "deep_tobit_classifier": "Deep Tobit",
    "deep_tobit_regressor": "Deep Tobit",
    "tabpfn_classifier": "TabPFN",
    "tobit_regressor": "Tobit",
    "aft_regressor": "AFT (Weibull)",
    "gnn_gcn_classifier": "GCN (linear fallback)",
    "gnn_sage_classifier": "GraphSAGE (linear fallback)",
    "hurdle_regressor": "Hurdle (XGB)",
    "xgboost_aft_regressor": "XGBoost AFT",
    "zi_tobit_classifier": "ZI-Tobit",
    "zi_tobit_regressor": "ZI-Tobit",
    "hurdle_aft_ensemble": "Hurdle+AFT Ensemble",
    "icp_classifier": "ICP",
    "icp_regressor": "ICP",
    "stacking_ensemble": "Stacking",
    "voting_ensemble": "Voting",
}
"""Map internal model code names to publication display names."""

# ---------------------------------------------------------------------------
# Domain-ordered feature prefixes (for CNN1D domain ordering)
# ---------------------------------------------------------------------------

DOMAIN_FEATURE_ORDER_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("nearest_", "count_"),  # proximity features
    ("frac_",),  # land use features
    ("aquifer_", "lithology_", "confinement_"),  # hydrogeology features
    ("pct_", "ej_", "prox_"),  # demographics features
)
