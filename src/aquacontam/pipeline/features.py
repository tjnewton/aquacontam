"""Pipeline feature extraction stage."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FeatureSpec:
    """Specification for a single feature extraction step.

    Parameters
    ----------
    name : str
        Human-readable name used in log messages and cache filenames.
    cache_file : str
        Parquet filename under ``interim/features/``.
    module : str
        Fully-qualified module to import the extractor from.
    func_name : str
        Name of the extractor function within *module*.
    downloaded_key : str | None
        Key into the ``downloaded`` dict.  ``None`` for extractors that do not
        use a downloaded file (e.g. system_characteristics).
    load_as : str
        How to load the downloaded file before passing it to the extractor:
        * ``"gpd"`` -- ``geopandas.read_parquet(path)``
        * ``"pd"``  -- ``pandas.read_parquet(path)``
        * ``"path"`` -- pass the ``Path`` object directly
        * ``"raw_dir"`` -- pass ``data_dir / "raw"`` (ignores downloaded dict)
    missing_msg : str
        Warning message logged when the downloaded file is missing.
    requires_large : bool
        If True, the step is skipped when ``skip_large=True``.
    pre_warning : str | None
        Optional warning emitted when the downloaded key is absent from *downloaded*,
        **before** checking the cache.  ``None`` to skip.
    log_count : bool
        If True, log an info message with the number of extracted systems.
    allow_empty : bool
        If True, an empty result DataFrame is **not** appended to the output
        and a warning is logged instead.
    """

    name: str
    cache_file: str
    module: str
    func_name: str
    downloaded_key: str | None
    load_as: str  # "gpd" | "pd" | "path" | "raw_dir"
    missing_msg: str
    requires_large: bool = False
    pre_warning: str | None = None
    log_count: bool = False
    allow_empty: bool = False


_FEATURE_SPECS: list[_FeatureSpec] = [
    _FeatureSpec(
        name="proximity",
        cache_file="proximity.parquet",
        module="aquacontam.features.proximity",
        func_name="extract_all_proximity_features",
        downloaded_key="epa_frs",
        load_as="gpd",
        missing_msg="EPA FRS parquet not found — skipping proximity features",
        pre_warning=(
            "EPA FRS not in downloaded sources — proximity features will be absent. "
            "Feature ablation for 'proximity' category will show zero matching columns."
        ),
    ),
    _FeatureSpec(
        name="land use",
        cache_file="land_use.parquet",
        module="aquacontam.features.land_use",
        func_name="extract_land_use_features",
        downloaded_key="nlcd",
        load_as="path",
        missing_msg="NLCD raster not found — skipping land use features",
        requires_large=True,
    ),
    _FeatureSpec(
        name="hydrogeology",
        cache_file="hydrogeology.parquet",
        module="aquacontam.features.hydrogeology",
        func_name="extract_aquifer_features",
        downloaded_key="usgs_aquifers",
        load_as="path",
        missing_msg="USGS aquifer data not found — skipping hydrogeology features",
    ),
    _FeatureSpec(
        name="demographics",
        cache_file="demographics.parquet",
        module="aquacontam.features.demographics",
        func_name="extract_demographic_features",
        downloaded_key="ejscreen",
        load_as="gpd",
        missing_msg="EJScreen parquet not found — skipping demographics features",
    ),
    _FeatureSpec(
        name="system characteristics",
        cache_file="system_characteristics.parquet",
        module="aquacontam.features.system_characteristics",
        func_name="extract_system_characteristics",
        downloaded_key=None,
        load_as="raw_dir",
        missing_msg="No SDWIS system metadata found — skipping system characteristics",
        log_count=True,
        allow_empty=True,
    ),
    _FeatureSpec(
        name="TRI proximity",
        cache_file="tri_proximity.parquet",
        module="aquacontam.features.tri_proximity",
        func_name="extract_tri_features",
        downloaded_key="tri_pfas",
        load_as="pd",
        missing_msg="TRI PFAS parquet not found — skipping TRI proximity features",
        log_count=True,
    ),
    _FeatureSpec(
        name="DoD proximity",
        cache_file="dod_proximity.parquet",
        module="aquacontam.features.dod_proximity",
        func_name="extract_dod_features",
        downloaded_key="dod_pfas",
        load_as="pd",
        missing_msg="DoD PFAS parquet not found — skipping DoD proximity features",
        log_count=True,
    ),
]


def _extract_one(
    spec: _FeatureSpec,
    systems_gdf: Any,
    data_dir: Path,
    interim_dir: Path,
    downloaded: dict[str, Path],
) -> Any | None:
    """Run a single feature extraction step.

    Returns the resulting DataFrame, or ``None`` if the step was skipped or
    failed (in non-strict mode).
    """
    import pandas as pd

    from aquacontam.pipeline._strict import is_strict

    cache_path = interim_dir / spec.cache_file

    # Load from cache if available
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        # Lazy-import the extractor
        mod = importlib.import_module(spec.module)
        extract_fn = getattr(mod, spec.func_name)

        if spec.load_as == "raw_dir":
            # system_characteristics: takes raw_dir, no downloaded dependency
            result_df = extract_fn(data_dir / "raw")
            if spec.allow_empty and result_df.empty:
                logger.warning(spec.missing_msg)
                return None
        else:
            # All other specs require a downloaded file
            assert spec.downloaded_key is not None
            data_path = downloaded.get(spec.downloaded_key)
            if not data_path or not data_path.exists():
                logger.warning(spec.missing_msg)
                return None

            if spec.load_as == "gpd":
                import geopandas as gpd

                loaded = gpd.read_parquet(data_path)
            elif spec.load_as == "pd":
                loaded = pd.read_parquet(data_path)
            else:
                # "path" — pass the Path directly
                loaded = data_path

            result_df = extract_fn(systems_gdf, loaded)

        # Cache and return
        result_df.to_parquet(cache_path)
        if spec.log_count:
            logger.info("Extracted %s features: %d systems", spec.name, len(result_df))
        return result_df

    except (ImportError, OSError, ValueError, RuntimeError):
        logger.warning("Failed to extract %s features", spec.name, exc_info=True)
        if is_strict():
            raise
        return None


def extract_features(
    data_dir: Path,
    wq_df: Any,
    downloaded: dict[str, Path],
    *,
    skip_large: bool = False,
    cache_subdir: str = "features",
    specs: list[_FeatureSpec] | None = None,
) -> list[Any]:
    """Extract feature DataFrames.

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    wq_df : pd.DataFrame
        Merged water quality DataFrame.
    downloaded : dict[str, Path]
        Mapping of source name -> downloaded file path from ``download_data``.
    skip_large : bool
        If True, skip NLCD-dependent features.

    Returns
    -------
    list[pd.DataFrame]
        Feature DataFrames indexed by pwsid.
    """
    import pandas as pd

    from aquacontam.features.assembly import build_system_geodataframe

    interim_dir = data_dir / "interim" / cache_subdir
    interim_dir.mkdir(parents=True, exist_ok=True)

    if wq_df.empty:
        return []

    systems_gdf = build_system_geodataframe(wq_df)
    feature_dfs: list[pd.DataFrame] = []

    for spec in specs if specs is not None else _FEATURE_SPECS:
        # Emit pre-warning when downloaded key is absent
        if spec.pre_warning and spec.downloaded_key not in downloaded:
            logger.warning(spec.pre_warning)

        # Skip large-data features when requested (still load from cache)
        if spec.requires_large and skip_large:
            cache_path = interim_dir / spec.cache_file
            if cache_path.exists():
                feature_dfs.append(pd.read_parquet(cache_path))
            continue

        result = _extract_one(spec, systems_gdf, data_dir, interim_dir, downloaded)
        if result is not None:
            feature_dfs.append(result)

    logger.info("Extracted %d feature DataFrames", len(feature_dfs))

    if skip_large:
        missing = []
        lu_path = interim_dir / "land_use.parquet"
        demo_path = interim_dir / "demographics.parquet"
        if not lu_path.exists():
            missing.append("land_use")
        if not demo_path.exists():
            missing.append("demographics")
        if missing:
            logger.warning(
                "%d feature set(s) unavailable due to --skip-large (%s). "
                "Model performance will be reduced.",
                len(missing),
                ", ".join(missing),
            )

    return feature_dfs
