"""Pipeline data download stage."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_data(
    data_dir: Path, *, skip_large: bool = False
) -> tuple[dict[str, Path], dict[str, int]]:
    """Download all data sources (idempotent).

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    skip_large : bool
        If True, skip NLCD raster (~1 GB).

    Returns
    -------
    tuple[dict[str, Path], dict[str, int]]
        ``(downloaded, stats)`` — mapping of source name -> downloaded file
        path, and a dict with keys ``"new"``, ``"cached"``, ``"unavailable"``
        counting how each source was resolved.
    """
    from aquacontam.pipeline._strict import is_strict

    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    for d in (raw_dir, interim_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    _stats: dict[str, int] = {"new": 0, "cached": 0, "unavailable": 0}

    # DataSource subclasses: (name, module_path, class_name)
    datasource_loaders = [
        ("ucmr5", "aquacontam.data.ucmr5", "UCMR5Source"),
        ("ucmr3", "aquacontam.data.ucmr3", "UCMR3Source"),
        ("sdwis", "aquacontam.data.sdwis", "SDWISSource"),
        ("mi_mpart", "aquacontam.data.mi_mpart", "MiMpartSource"),
        ("ca_geotracker", "aquacontam.data.ca_geotracker", "CaGeoTrackerSource"),
        ("nj_dep", "aquacontam.data.nj_dep", "NjDepSource"),
        ("nc_deq", "aquacontam.data.nc_deq", "NcDeqSource"),
        ("wqp", "aquacontam.data.wqp", "WqpSource"),
        ("mo_dnr", "aquacontam.data.mo_dnr", "MoDnrSource"),
        ("tx_tceq", "aquacontam.data.tx_tceq", "TxTceqSource"),
        ("oh_epa", "aquacontam.data.oh_epa", "OhEpaSource"),
        ("wa_doh", "aquacontam.data.wa_doh", "WaDohSource"),
        ("mn_mdh", "aquacontam.data.mn_mdh", "MnMdhSource"),
    ]

    from aquacontam._config import load_data_config

    full_config = load_data_config()

    for name, module_path, class_name in datasource_loaders:
        output_path = processed_dir / f"{name}.parquet"

        # Check if source is marked unavailable in config
        source_cfg = full_config.get(name, {})
        if isinstance(source_cfg, dict) and source_cfg.get("unavailable"):
            if output_path.exists():
                logger.warning(
                    "Removing stale cache %s — source %r is marked unavailable: %s",
                    output_path,
                    name,
                    source_cfg.get("unavailable_reason", "no reason given"),
                )
                output_path.unlink()
            else:
                logger.info(
                    "Skipping %s (unavailable: %s)",
                    name,
                    source_cfg.get("unavailable_reason", "no reason given"),
                )
            _stats["unavailable"] += 1
            continue

        if output_path.exists():
            logger.info("Skipping %s (already exists)", name)
            downloaded[name] = output_path
            _stats["cached"] += 1
            continue

        try:
            import importlib

            mod = importlib.import_module(module_path)
            source_cls = getattr(mod, class_name)
            source = source_cls(
                raw_dir=raw_dir,
                interim_dir=interim_dir,
                processed_dir=processed_dir,
            )
            output_path = source.run()
            downloaded[name] = output_path
            _stats["new"] += 1
            logger.info("Downloaded %s → %s", name, output_path)
        except (ImportError, AttributeError, KeyError, OSError, ValueError, RuntimeError):
            logger.warning("Failed to download %s", name, exc_info=True)
            if is_strict():
                raise
            _stats["unavailable"] += 1

    # EPA FRS (function-based, not a DataSource subclass)
    frs_path = interim_dir / "epa_frs.parquet"
    if not frs_path.exists():
        try:
            from aquacontam.data.epa_frs import download_frs, load_frs

            download_frs(raw_dir)
            frs_gdf = load_frs(raw_dir)
            frs_gdf.to_parquet(frs_path, index=False)
            downloaded["epa_frs"] = frs_path
            _stats["new"] += 1
            logger.info("Downloaded epa_frs: %d facilities", len(frs_gdf))
        except (ImportError, OSError, ValueError):
            logger.warning("Failed to download EPA FRS", exc_info=True)
            if is_strict():
                raise
            _stats["unavailable"] += 1
    else:
        downloaded["epa_frs"] = frs_path
        _stats["cached"] += 1

    # EJScreen (~6 GB from Zenodo)
    ejscreen_path = interim_dir / "ejscreen.parquet"
    if not skip_large:
        if not ejscreen_path.exists():
            try:
                from aquacontam.data.ejscreen import download_ejscreen, load_ejscreen

                download_ejscreen(raw_dir)

                # Log GDB discovery status for centroid extraction
                from aquacontam.data.ejscreen import find_ejscreen_gdb

                gdb_path = find_ejscreen_gdb(raw_dir)
                if gdb_path:
                    logger.info("EJScreen GDB found: %s", gdb_path)
                else:
                    logger.warning(
                        "EJScreen GDB not found after download — "
                        "centroid extraction from polygons will not be available"
                    )

                ej_gdf = load_ejscreen(raw_dir)
                ej_gdf.to_parquet(ejscreen_path, index=False)
                downloaded["ejscreen"] = ejscreen_path
                _stats["new"] += 1
                logger.info("Downloaded ejscreen: %d block groups", len(ej_gdf))
            except (ImportError, OSError, ValueError, RuntimeError):
                logger.warning("Failed to download EJScreen", exc_info=True)
                if is_strict():
                    raise
                _stats["unavailable"] += 1
        else:
            downloaded["ejscreen"] = ejscreen_path
            _stats["cached"] += 1
    else:
        if ejscreen_path.exists():
            downloaded["ejscreen"] = ejscreen_path
            _stats["cached"] += 1
        else:
            logger.info("Skipping EJScreen download (--skip-large)")
            _stats["unavailable"] += 1

    # NJ private wells (function-based)
    njpw_path = interim_dir / "nj_private_wells.parquet"
    if not njpw_path.exists():
        try:
            from aquacontam.data.nj_private_wells import (
                download_nj_private_wells,
                load_nj_private_wells,
            )

            download_nj_private_wells(raw_dir)
            njpw_gdf = load_nj_private_wells(raw_dir)
            njpw_gdf.to_parquet(njpw_path, index=False)
            downloaded["nj_private_wells"] = njpw_path
            _stats["new"] += 1
            logger.info("Downloaded nj_private_wells: %d rows", len(njpw_gdf))
        except (ImportError, OSError, ValueError, RuntimeError):
            logger.warning("Failed to download NJ private wells", exc_info=True)
            # NJ PWTA bulk data URL returns 404 (known-unavailable source).
            # Don't fail strict mode — only needed for optional T6 task.
            _stats["unavailable"] += 1
    else:
        downloaded["nj_private_wells"] = njpw_path
        _stats["cached"] += 1

    # TRI PFAS facilities (function-based, not a DataSource subclass)
    tri_path = interim_dir / "tri_pfas.parquet"
    if not tri_path.exists():
        try:
            from aquacontam.data.tri import download_tri_pfas, load_tri_pfas

            download_tri_pfas(raw_dir)
            tri_df = load_tri_pfas(raw_dir)
            tri_df.to_parquet(tri_path, index=False)
            downloaded["tri_pfas"] = tri_path
            _stats["new"] += 1
            logger.info("Downloaded tri_pfas: %d facilities", len(tri_df))
        except (ImportError, OSError, ValueError):
            logger.warning("Failed to download TRI PFAS", exc_info=True)
            if is_strict():
                raise
            _stats["unavailable"] += 1
    else:
        downloaded["tri_pfas"] = tri_path
        _stats["cached"] += 1

    # DoD PFAS sites (function-based, not a DataSource subclass)
    dod_path = interim_dir / "dod_pfas.parquet"
    if not dod_path.exists():
        try:
            from aquacontam.data.dod_pfas import download_dod_pfas, load_dod_pfas

            download_dod_pfas(raw_dir)
            dod_df = load_dod_pfas(raw_dir)
            dod_df.to_parquet(dod_path, index=False)
            downloaded["dod_pfas"] = dod_path
            _stats["new"] += 1
            logger.info("Downloaded dod_pfas: %d sites", len(dod_df))
        except (ImportError, OSError, ValueError):
            logger.warning("Failed to download DoD PFAS", exc_info=True)
            if is_strict():
                raise
            _stats["unavailable"] += 1
    else:
        downloaded["dod_pfas"] = dod_path
        _stats["cached"] += 1

    # ZCTA gazetteer (for ZIP code geocoding)
    try:
        from aquacontam.geo.geocoding import download_zcta_gazetteer

        _existed = any(raw_dir.glob("*zcta*"))
        zcta_path = download_zcta_gazetteer(raw_dir)
        downloaded["zcta_gazetteer"] = zcta_path
        _stats["cached" if _existed else "new"] += 1
        logger.info("Downloaded ZCTA gazetteer → %s", zcta_path)
    except (ImportError, OSError, ValueError):
        logger.warning("Failed to download ZCTA gazetteer", exc_info=True)
        if is_strict():
            raise
        _stats["unavailable"] += 1

    # NLCD raster (~1 GB)
    if not skip_large:
        try:
            from aquacontam.features.land_use import download_nlcd

            _existed = any(raw_dir.glob("*nlcd*")) or any(raw_dir.glob("*NLCD*"))
            nlcd_path = download_nlcd(raw_dir)
            downloaded["nlcd"] = nlcd_path
            _stats["cached" if _existed else "new"] += 1
            logger.info("Downloaded NLCD raster → %s", nlcd_path)
        except (ImportError, OSError, ValueError):
            logger.warning(
                "Failed to download NLCD raster. To add NLCD land use features, "
                "manually download from https://www.sciencebase.gov/catalog/item/"
                "655ceb8ad34ee4b6e05cc51a and place the .img or .tif file in %s",
                raw_dir,
                exc_info=True,
            )
            if is_strict():
                raise
            _stats["unavailable"] += 1
    else:
        logger.info("Skipping NLCD raster download (--skip-large)")
        _stats["unavailable"] += 1

    # USGS principal aquifers
    try:
        from aquacontam.features.hydrogeology import download_principal_aquifers

        _existed = any(raw_dir.glob("*aquifer*"))
        aquifer_path = download_principal_aquifers(raw_dir)
        downloaded["usgs_aquifers"] = aquifer_path
        _stats["cached" if _existed else "new"] += 1
        logger.info("Downloaded USGS aquifers → %s", aquifer_path)
    except (ImportError, OSError, ValueError):
        logger.warning("Failed to download USGS aquifers", exc_info=True)
        if is_strict():
            raise
        _stats["unavailable"] += 1

    # Verify downloaded file checksums against data/checksums.sha256
    from aquacontam.data._download import verify_checksum

    n_verified, n_failed = 0, 0
    for name, path in downloaded.items():
        if path.exists():
            if not verify_checksum(path):
                n_failed += 1
                logger.error("Checksum FAILED for %s: %s", name, path)
            n_verified += 1

    if n_failed:
        msg = f"Checksum verification failed for {n_failed}/{n_verified} files"
        logger.error(msg)
        if is_strict():
            raise ValueError(msg)
    elif n_verified:
        logger.info("Checksum verification passed for %d files", n_verified)

    return downloaded, _stats
