"""Geospatial, temporal, and land-use feature engineering."""

from aquacontam.features.assembly import (
    aggregate_to_system_level,
    assemble_feature_matrix,
    build_system_geodataframe,
    drop_leakage_columns,
    prepare_train_val_test,
)
from aquacontam.features.demographics import extract_demographic_features
from aquacontam.features.dod_proximity import extract_dod_features
from aquacontam.features.grid import generate_conus_grid
from aquacontam.features.hydrogeology import (
    download_principal_aquifers,
    extract_aquifer_features,
)
from aquacontam.features.land_use import (
    compute_land_use_fractions,
    download_nlcd,
    extract_land_use_features,
)
from aquacontam.features.proximity import (
    compute_proximity_features,
    extract_all_proximity_features,
)
from aquacontam.features.system_characteristics import extract_system_characteristics
from aquacontam.features.tri_proximity import extract_tri_features

__all__ = [
    "aggregate_to_system_level",
    "assemble_feature_matrix",
    "build_system_geodataframe",
    "compute_land_use_fractions",
    "compute_proximity_features",
    "download_nlcd",
    "download_principal_aquifers",
    "drop_leakage_columns",
    "extract_all_proximity_features",
    "extract_aquifer_features",
    "extract_demographic_features",
    "extract_dod_features",
    "extract_land_use_features",
    "extract_system_characteristics",
    "extract_tri_features",
    "generate_conus_grid",
    "prepare_train_val_test",
]
