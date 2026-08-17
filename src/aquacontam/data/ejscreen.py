"""EPA EJScreen data loader — environmental justice screening indices.

Downloads and parses the national EJScreen dataset at the Census block group
level. This is NOT a ``DataSource`` subclass — EJScreen provides demographic
and environmental justice indices, not water quality samples.

Typical usage::

    from aquacontam.data.ejscreen import download_ejscreen, load_ejscreen

    download_ejscreen(Path("data/raw"))
    ejscreen = load_ejscreen(Path("data/raw"))
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import (
    CONUS_LAT_MAX,
    CONUS_LAT_MIN,
    CONUS_LON_MAX,
    CONUS_LON_MIN,
    CRS_STORAGE,
    DOWNLOAD_TIMEOUT_DEFAULT,
    DOWNLOAD_TIMEOUT_LARGE,
    EJSCREEN_DEMOGRAPHIC_COLS,
    EJSCREEN_EJ_INDEX_COLS,
)
from aquacontam.data._download import download_and_extract

logger = logging.getLogger(__name__)


def download_bg_gazetteer(raw_dir: Path, *, force: bool = False) -> Path:
    """Download the Census block group gazetteer file.

    .. deprecated::
        The Census Bureau does not publish BG-level gazetteer files. The
        primary geocoding path now uses EJScreen GDB polygon centroids via
        ``_try_load_gdb_centroids()``. This function is retained for
        backwards compatibility but will raise ``RuntimeError`` when the
        config marks the source as unavailable.

    Parameters
    ----------
    raw_dir : Path
        Directory to save the downloaded file.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the extracted gazetteer text file.

    Raises
    ------
    RuntimeError
        If the data source is marked unavailable in config.
    """
    from aquacontam._config import load_data_config

    config = load_data_config()["bg_gazetteer"]
    if config.get("unavailable"):
        reason = config.get("unavailable_reason", "data source unavailable")
        raise RuntimeError(f"Census block group gazetteer is unavailable: {reason}")
    paths = download_and_extract(
        config["url"],
        raw_dir,
        config["expected_files"],
        force=force,
        timeout=DOWNLOAD_TIMEOUT_DEFAULT,
        progress_desc="BG gazetteer",
    )
    return paths[0]


def _load_bg_centroids(raw_dir: Path) -> pd.DataFrame:
    """Load block group centroid lookup table.

    .. deprecated::
        This is the fallback path when GDB centroids are unavailable. The
        primary geocoding path uses ``_try_load_gdb_centroids()`` which
        extracts centroids from the EJScreen GDB polygon layer.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the gazetteer text file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``latitude``, ``longitude``
        indexed by block group GEOID (string).
    """
    from aquacontam._config import load_data_config

    config = load_data_config()["bg_gazetteer"]
    filepath = raw_dir / config["expected_files"][0]

    if not filepath.exists():
        if config.get("unavailable"):
            reason = config.get("unavailable_reason", "data source unavailable")
            raise RuntimeError(f"Census block group gazetteer is unavailable: {reason}")
        filepath = download_bg_gazetteer(raw_dir)

    fmt = config.get("format", {})

    df = pd.read_csv(
        filepath,
        sep=fmt.get("delimiter", "\t"),
        encoding=fmt.get("encoding", "utf-8"),
        dtype={"GEOID": str},
    )

    # Strip whitespace from column names (Census gazetteer has trailing spaces)
    df.columns = df.columns.str.strip()

    df["GEOID"] = df["GEOID"].str.strip()
    df["latitude"] = pd.to_numeric(df["INTPTLAT"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["INTPTLONG"], errors="coerce")

    result: pd.DataFrame = df[["GEOID", "latitude", "longitude"]].dropna().set_index("GEOID")
    logger.info("Loaded %d block group centroids", len(result))
    return result


def find_ejscreen_gdb(raw_dir: Path) -> Path | None:
    """Search for the EJScreen GDB file in a directory tree.

    Parameters
    ----------
    raw_dir : Path
        Directory tree to search.

    Returns
    -------
    Path | None
        Path to the GDB file (zip or directory), or ``None`` if not found.
    """
    # Search for the GDB zip first, then extracted directory
    candidates = list(raw_dir.rglob("EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb.zip"))
    if not candidates:
        candidates = list(raw_dir.rglob("EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb"))
    if not candidates:
        # Broader search for any .gdb file
        candidates = list(raw_dir.rglob("*.gdb.zip")) + list(raw_dir.rglob("*.gdb"))
        candidates = [c for c in candidates if "EJSCREEN" in c.name.upper()]
    return candidates[0] if candidates else None


def _try_load_gdb_centroids(raw_dir: Path) -> pd.DataFrame | None:
    """Try to extract block group centroids from the EJScreen GDB file.

    Searches for ``EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb.zip`` (or
    extracted ``.gdb`` directory) under *raw_dir*.  If found, reads the
    polygon layer and computes centroids in EPSG:4326.

    Parameters
    ----------
    raw_dir : Path
        Directory tree to search for the GDB file.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with ``latitude``, ``longitude`` indexed by block group
        GEOID string, or ``None`` if no GDB file is found.
    """
    # Search for the GDB zip or extracted directory
    gdb_candidates = list(raw_dir.rglob("EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb.zip"))
    if not gdb_candidates:
        gdb_candidates = list(raw_dir.rglob("EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb"))
    if not gdb_candidates:
        return None

    gdb_path = gdb_candidates[0]
    logger.info("Loading EJScreen GDB for centroid extraction: %s", gdb_path)

    try:
        import fiona

        gdb_str = str(gdb_path)
        if gdb_str.endswith(".zip"):
            gdb_str = f"zip://{gdb_str}"

        # Find the block-group layer (the one with polygon geometry).
        # EJScreen GDB may contain multiple layers; the default layer
        # (e.g. "USA") often lacks geometry entirely.
        layer_name = None
        try:
            layers = fiona.listlayers(gdb_str)
            for lyr in layers:
                with fiona.open(gdb_str, layer=lyr) as src:
                    if src.schema.get("geometry") in (
                        "Polygon",
                        "MultiPolygon",
                        "3D Polygon",
                        "3D MultiPolygon",
                    ):
                        layer_name = lyr
                        break
        except (OSError, KeyError, RuntimeError):
            logger.debug("Could not inspect GDB layers; trying default layer")

        # Read only the ID column + geometry to minimise memory.
        read_kwargs: dict[str, object] = {}
        if layer_name is not None:
            read_kwargs["layer"] = layer_name
            logger.info("Using GDB layer %r (%d features expected)", layer_name, 0)
        try:
            gdf = gpd.read_file(gdb_str, columns=["ID"], **read_kwargs)
        except TypeError:
            # Older geopandas versions may not support columns kwarg
            gdf = gpd.read_file(gdb_str, **read_kwargs)[["ID", "geometry"]]

        if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
            logger.warning("GDB layer lacks geometry column — cannot extract centroids")
            return None

        # Reproject to a projected CRS for accurate centroid calculation,
        # then convert centroids back to WGS84 for lat/lon output.
        gdf_proj = gdf.to_crs("EPSG:5070") if gdf.crs is None or gdf.crs.to_epsg() == 4326 else gdf

        centroids_proj = gdf_proj.geometry.centroid
        # Convert centroids back to WGS84
        centroids_gdf = gpd.GeoDataFrame(geometry=centroids_proj, crs=gdf_proj.crs)
        centroids_wgs = centroids_gdf.to_crs(CRS_STORAGE)

        result = pd.DataFrame(
            {
                "latitude": centroids_wgs.geometry.y.to_numpy(),
                "longitude": centroids_wgs.geometry.x.to_numpy(),
            },
            index=gdf["ID"].astype(str).str.strip().to_numpy(),
        )
        result.index.name = "GEOID"
        logger.info("Extracted %d block group centroids from GDB", len(result))
        return cast(pd.DataFrame, result)
    except (OSError, ValueError, TypeError, RuntimeError):
        logger.warning("Failed to read EJScreen GDB: %s", gdb_path, exc_info=True)
        return None


def download_ejscreen(raw_dir: Path, *, force: bool = False) -> Path:
    """Download the national EJScreen CSV from EPA (or Zenodo archive).

    Parameters
    ----------
    raw_dir : Path
        Directory to save the downloaded file.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the downloaded/extracted CSV file.
    """
    config = load_data_config()["ejscreen"]
    url: str = config["url"]
    expected_files: list[str] = config["expected_files"]
    expected_name = expected_files[0]

    # Check if expected file already exists (flat or nested)
    flat_path = raw_dir / expected_name
    if flat_path.exists() and not force:
        logger.info("EJScreen CSV already exists: %s", flat_path)
        return flat_path

    # Search recursively for previously-extracted file
    if not force:
        found = list(raw_dir.rglob(expected_name))
        if found:
            logger.info("EJScreen CSV found at: %s", found[0])
            return found[0]

    try:
        paths = download_and_extract(
            url,
            raw_dir,
            expected_files,
            force=force,
            timeout=DOWNLOAD_TIMEOUT_LARGE,
            progress_desc="EJScreen download",
        )
        # Log GDB discovery after extraction
        gdb_found = find_ejscreen_gdb(raw_dir)
        if gdb_found:
            logger.info("EJScreen GDB discovered after extraction: %s", gdb_found)
        else:
            logger.info("EJScreen GDB not found in archive — CSV-only download")
        return paths[0]
    except FileNotFoundError:
        # Zenodo archive may nest files in subdirectories
        found = list(raw_dir.rglob(expected_name))
        if found:
            logger.info("Found EJScreen CSV in nested path: %s", found[0])
            return found[0]
        # Also search for inner zip files that might contain the CSV
        for inner_zip in raw_dir.rglob("*.zip"):
            with zipfile.ZipFile(inner_zip) as zf:
                for member in zf.namelist():
                    if expected_name in member:
                        zf.extract(member, raw_dir)
                        extracted = raw_dir / member
                        if extracted.exists():
                            logger.info("Extracted EJScreen CSV from nested zip: %s", extracted)
                            return extracted
        raise


def load_ejscreen(
    raw_dir: Path,
    *,
    columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Load and parse EJScreen block group data into a GeoDataFrame.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the EJScreen CSV file.
    columns : list[str] | None
        If provided, keep only these output columns (plus geometry and ID).
        Default keeps all demographic + EJ index columns defined in constants.

    Returns
    -------
    GeoDataFrame
        Block group centroids with demographic and EJ index columns.
        CRS is EPSG:4326.
    """
    config = load_data_config()["ejscreen"]
    expected_file: str = config["expected_files"][0]
    filepath = raw_dir / expected_file

    # If not at flat path, search recursively (Zenodo archive nests files)
    if not filepath.exists():
        found = list(raw_dir.rglob(expected_file))
        if found:
            filepath = found[0]
        else:
            raise FileNotFoundError(
                f"EJScreen data file not found: {filepath}. Run download_ejscreen() first."
            )

    # Fail fast if the CSV lacks coordinates and no centroid source is available
    # — avoids reading the full 243k-row CSV only to crash later.
    bg_config = load_data_config().get("bg_gazetteer", {})
    if bg_config.get("unavailable"):
        gaz_file = raw_dir / bg_config.get("expected_files", [""])[0]
        gdb_available = find_ejscreen_gdb(raw_dir) is not None
        if not gaz_file.exists() and not gdb_available:
            header_cols = pd.read_csv(
                filepath,
                nrows=0,
                encoding=config.get("format", {}).get("encoding", "utf-8"),
            ).columns
            has_coords = ("LATITUDE" in header_cols or "latitude" in header_cols) and (
                "LONGITUDE" in header_cols or "longitude" in header_cols
            )
            if not has_coords:
                reason = bg_config.get("unavailable_reason", "data source unavailable")
                raise RuntimeError(
                    f"EJScreen CSV lacks coordinate columns and no centroid source "
                    f"available (gazetteer: {reason}; GDB: not found)"
                )

    col_map: dict[str, str] = config["column_map"]
    encoding: str = config.get("format", {}).get("encoding", "utf-8")

    logger.info("Reading EJScreen data from %s …", filepath)
    df = pd.read_csv(filepath, encoding=encoding, dtype={"ID": str}, low_memory=False)
    logger.info("Read %d block groups from EJScreen", len(df))

    # Rename columns per config
    rename = {raw: std for raw, std in col_map.items() if raw in df.columns}
    df = df.rename(columns=rename)

    # Ensure block group ID column exists
    if "block_group_id" not in df.columns:
        raise ValueError("EJScreen data missing block group ID column after rename")

    # Coerce numeric columns
    for col in list(EJSCREEN_DEMOGRAPHIC_COLS) + list(EJSCREEN_EJ_INDEX_COLS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce coordinates — check for both renamed (lowercase) and raw
    # (uppercase) column names. Zenodo 2023 CSV lacks LATITUDE/LONGITUDE
    # entirely; derive from Census block group centroids in that case.
    has_lat = "latitude" in df.columns or "LATITUDE" in df.columns
    has_lon = "longitude" in df.columns or "LONGITUDE" in df.columns

    if has_lat and has_lon:
        # Normalize uppercase to lowercase if needed
        if "LATITUDE" in df.columns and "latitude" not in df.columns:
            df = df.rename(columns={"LATITUDE": "latitude", "LONGITUDE": "longitude"})
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    else:
        # Try GDB centroid extraction first, fall back to BG gazetteer
        gdb_centroids = _try_load_gdb_centroids(raw_dir)
        if gdb_centroids is not None:
            logger.info(
                "EJScreen CSV lacks coordinates — "
                "deriving from GDB polygon centroids (%d block groups)",
                len(gdb_centroids),
            )
            bg_id = df["block_group_id"].str.strip()
            df["latitude"] = bg_id.map(gdb_centroids["latitude"])
            df["longitude"] = bg_id.map(gdb_centroids["longitude"])
        else:
            logger.info(
                "EJScreen CSV lacks latitude/longitude columns — "
                "deriving from Census block group gazetteer centroids"
            )
            bg_centroids = _load_bg_centroids(raw_dir)
            bg_id = df["block_group_id"].str.strip()
            df["latitude"] = bg_id.map(bg_centroids["latitude"])
            df["longitude"] = bg_id.map(bg_centroids["longitude"])

    # Drop rows without valid coordinates
    valid_coords = df["latitude"].notna() & df["longitude"].notna()
    n_dropped = int((~valid_coords).sum())
    if n_dropped > 0:
        logger.info("Dropping %d block groups without valid coordinates", n_dropped)
    df = df[valid_coords].copy()

    # Filter to CONUS bounding box
    conus_mask = (
        (df["latitude"] >= CONUS_LAT_MIN)
        & (df["latitude"] <= CONUS_LAT_MAX)
        & (df["longitude"] >= CONUS_LON_MIN)
        & (df["longitude"] <= CONUS_LON_MAX)
    )
    n_outside = int((~conus_mask).sum())
    if n_outside > 0:
        logger.info("Dropping %d block groups outside CONUS bounding box", n_outside)
    df = df[conus_mask].copy()

    # Select output columns
    keep_cols = ["block_group_id"]
    if columns is not None:
        keep_cols += [c for c in columns if c in df.columns]
    else:
        keep_cols += [
            c
            for c in list(EJSCREEN_DEMOGRAPHIC_COLS) + list(EJSCREEN_EJ_INDEX_COLS)
            if c in df.columns
        ]

    # Always keep lat/lon for geometry
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[[*available_cols, "latitude", "longitude"]].copy()

    # Create GeoDataFrame with block group centroids
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=CRS_STORAGE,
    )

    logger.info("Loaded %d EJScreen block groups", len(gdf))
    return gdf
