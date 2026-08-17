"""System characteristic features from SDWIS metadata.

Extracts source water type, population served, system type, and owner
type from SDWA_PUB_WATER_SYSTEMS.csv. These are standard features in
the water contamination prediction literature (Hu et al. 2016, Fernandez
et al. 2023, Tokranov et al. 2024).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam.data._utils import find_col

logger = logging.getLogger(__name__)

# Standard EPA GW/SW codes
_SOURCE_WATER_TYPES = {"GW": "groundwater", "SW": "surface_water", "GU": "groundwater_udi"}

# PWS types
_SYSTEM_TYPES = {
    "CWS": "community",
    "NTNCWS": "non_transient",
    "TNCWS": "transient",
}

# Owner types
_OWNER_TYPES = {
    "F": "federal",
    "S": "state",
    "L": "local",
    "P": "private",
    "N": "native_american",
}


def extract_system_characteristics(
    raw_dir: Path,
    *,
    config_path: Path | str | None = None,
) -> pd.DataFrame:
    """Extract system characteristics from SDWIS metadata.

    Reads ``SDWA_PUB_WATER_SYSTEMS.csv`` and extracts:
    - ``source_water_type``: GW/SW/GU (categorical)
    - ``population_served``: integer count (numeric)
    - ``system_type``: CWS/NTNCWS/TNCWS (categorical)
    - ``owner_type``: F/S/L/P/N (categorical)

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw SDWIS files.
    config_path : Path or str or None
        Path to data config YAML.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with system characteristic columns.
    """
    config = load_data_config(config_path, source_name="sdwis")
    fmt = config.get("format", {})
    encoding = fmt.get("encoding", "latin-1")

    systems_path = raw_dir / config["expected_files"][1]  # SDWA_PUB_WATER_SYSTEMS.csv
    if not systems_path.exists():
        logger.warning("SDWIS system metadata not found at %s", systems_path)
        return cast(
            pd.DataFrame,
            pd.DataFrame(columns=["population_served", "source_water_type"]),
        )

    logger.info("Reading SDWIS system metadata from %s …", systems_path)
    systems = pd.read_csv(
        systems_path,
        encoding=encoding,
        dtype={"PWSID": str},
        low_memory=False,
    )
    logger.info("Read %d system rows", len(systems))

    # Deduplicate
    pwsid_col = find_col(systems, ["PWSID"])
    systems = systems.drop_duplicates(subset=[pwsid_col], keep="first")

    result = pd.DataFrame(index=systems[pwsid_col])
    result.index.name = "pwsid"

    # Source water type
    gw_sw_col = find_col(systems, ["GW_SW_CODE", "PRIMARY_SOURCE_CODE"], required=False)
    if gw_sw_col:
        result["source_water_type"] = systems[gw_sw_col].to_numpy()
        result["source_water_type"] = (
            result["source_water_type"].astype(str).str.strip().str.upper()
        )
    else:
        result["source_water_type"] = np.nan

    # Population served
    pop_col = find_col(systems, ["POPULATION_SERVED_COUNT", "POP_CAT_11_CODE"], required=False)
    if pop_col:
        result["population_served"] = pd.to_numeric(systems[pop_col].to_numpy(), errors="coerce")
        # Log-transform population (wide range: 25 to 10M+)
        result["log_population_served"] = np.log1p(
            result["population_served"].fillna(0).clip(lower=0)
        )
    else:
        result["population_served"] = np.nan
        result["log_population_served"] = np.nan

    # System type
    type_col = find_col(systems, ["PWS_TYPE_CODE", "PWS_ACTIVITY_CODE"], required=False)
    if type_col:
        result["system_type"] = systems[type_col].to_numpy()
        result["system_type"] = result["system_type"].astype(str).str.strip().str.upper()
    else:
        result["system_type"] = np.nan

    # Owner type
    owner_col = find_col(systems, ["OWNER_TYPE_CODE"], required=False)
    if owner_col:
        result["owner_type"] = systems[owner_col].to_numpy()
        result["owner_type"] = result["owner_type"].astype(str).str.strip().str.upper()
    else:
        result["owner_type"] = np.nan

    n_valid = result.notna().sum()
    logger.info(
        "Extracted system characteristics for %d systems: "
        "source_water=%d, population=%d, type=%d, owner=%d",
        len(result),
        n_valid.get("source_water_type", 0),
        n_valid.get("population_served", 0),
        n_valid.get("system_type", 0),
        n_valid.get("owner_type", 0),
    )
    return cast(pd.DataFrame, result)
