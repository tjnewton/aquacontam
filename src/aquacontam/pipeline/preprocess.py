"""Pipeline data preprocessing stage.

Design choice: the pipeline uses **binary detection targets** (any PFAS
detected at a public water system) rather than substituting censored
concentrations.  This sidesteps the need for a concentration substitution
method (DL/2, sqrt2, ROS) at the preprocessing stage.  Censored
concentration metadata is only passed to specialised models (Deep Tobit,
ICP) that handle censoring internally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def preprocess_data(
    data_dir: Path,
    downloaded: dict[str, Path],
) -> Any:
    """Load and merge all water quality data sources.

    Returns
    -------
    pd.DataFrame
        Merged water quality DataFrame.
    """
    import pandas as pd

    from aquacontam.pipeline._strict import is_strict
    from aquacontam.preprocessing.cleaning import merge_datasets

    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    cached = interim_dir / "merged_wq.parquet"

    if cached.exists():
        logger.info("Loading cached merged WQ data from %s", cached)
        return pd.read_parquet(cached)

    wq_sources = [
        "ucmr5",
        "ucmr3",
        "sdwis",
        "mi_mpart",
        "ca_geotracker",
        "nj_dep",
        "nc_deq",
        "wqp",
        "mo_dnr",
        "wa_doh",
        "mn_mdh",
    ]
    dfs = []
    for name in wq_sources:
        path = downloaded.get(name)
        if path and path.exists():
            df = pd.read_parquet(path)
            df["source"] = name
            dfs.append(df)
            logger.info("Loaded %s: %d rows", name, len(dfs[-1]))

    if not dfs:
        logger.error("No water quality data available")
        return pd.DataFrame()

    merged = merge_datasets(*dfs)

    # Validate merged schema (catch column mismatches across sources early)
    from aquacontam.data.schema import validate_schema

    merged = validate_schema(
        merged, drop_invalid_rows=False, skip_coord_check=True, coerce_dtypes=True
    )
    logger.info(
        "Post-merge schema validation: %d rows, %d columns, %d unique PWSIDs",
        len(merged),
        len(merged.columns),
        merged["pwsid"].nunique() if "pwsid" in merged.columns else 0,
    )

    # Backfill missing coordinates from other sources in the merged dataset.
    # WA DOH rows have NaN lat/lon; SDWIS/UCMR may have coords for same PWSIDs.
    if merged["latitude"].isna().any():
        coord_lookup = (
            merged[merged["latitude"].notna()]
            .sort_values(["source", "sample_date"], na_position="last")
            .groupby("pwsid")[["latitude", "longitude"]]
            .first()
        )
        missing_mask = merged["latitude"].isna() & merged["pwsid"].isin(coord_lookup.index)
        n_backfilled = missing_mask.sum()
        if n_backfilled > 0:
            merged.loc[missing_mask, "latitude"] = merged.loc[missing_mask, "pwsid"].map(
                coord_lookup["latitude"]
            )
            merged.loc[missing_mask, "longitude"] = merged.loc[missing_mask, "pwsid"].map(
                coord_lookup["longitude"]
            )
            logger.info("Backfilled coordinates for %d rows from other sources", n_backfilled)

    # Join UCMR ZIP code files (PWSID -> ZIPCODE) if raw_ZipCode not already present
    raw_dir = data_dir / "raw"
    if "raw_ZipCode" not in merged.columns or merged["raw_ZipCode"].isna().all():
        zip_dfs = []
        for zf in ["UCMR5_ZIPCodes.txt", "UCMR3_ZIPCodes.txt"]:
            zpath = raw_dir / zf
            if zpath.exists():
                zdf = pd.read_csv(zpath, sep="\t", dtype=str)
                zip_dfs.append(zdf)
                logger.info("Loaded %s: %d PWSID→ZIP mappings", zf, len(zdf))
        if zip_dfs:
            all_zips = pd.concat(zip_dfs).drop_duplicates(subset=["PWSID"])
            zip_lookup = all_zips.set_index("PWSID")["ZIPCODE"]
            # Only fill where raw_ZipCode is missing
            if "raw_ZipCode" not in merged.columns:
                merged["raw_ZipCode"] = ""
            mask = merged["raw_ZipCode"].isna() | (merged["raw_ZipCode"] == "")
            merged.loc[mask, "raw_ZipCode"] = merged.loc[mask, "pwsid"].map(zip_lookup).fillna("")
            n_filled = (merged["raw_ZipCode"] != "").sum()
            logger.info("Filled raw_ZipCode for %d rows from UCMR ZIP files", n_filled)

    # Backfill raw_ZipCode from other rows with the same PWSID
    # (e.g. SDWIS rows may have ZIP codes that MI MPART / WA DOH rows lack)
    if "raw_ZipCode" in merged.columns:
        zip_missing = merged["raw_ZipCode"].isna() | (merged["raw_ZipCode"] == "")
        if zip_missing.any():
            zip_by_pwsid = (
                merged.loc[~zip_missing]
                .sort_values(["source", "sample_date"], na_position="last")
                .groupby("pwsid")["raw_ZipCode"]
                .first()
            )
            fill_mask = zip_missing & merged["pwsid"].isin(zip_by_pwsid.index)
            if fill_mask.any():
                merged.loc[fill_mask, "raw_ZipCode"] = merged.loc[fill_mask, "pwsid"].map(
                    zip_by_pwsid
                )
                logger.info(
                    "Backfilled raw_ZipCode for %d rows from other sources",
                    fill_mask.sum(),
                )

    # Geocode systems via ZIP code centroids
    if "raw_ZipCode" in merged.columns:
        try:
            from aquacontam.geo.geocoding import geocode_by_zipcode, load_zcta_centroids

            centroids = load_zcta_centroids(raw_dir)
            merged = geocode_by_zipcode(merged, centroids, zip_col="raw_ZipCode")
            _mask = merged.get("coord_source") == "zip_centroid"
            n_geocoded = int(_mask.sum()) if not isinstance(_mask, bool) else int(_mask)
            logger.info("Geocoded %d rows via ZIP centroids", n_geocoded)

            # Compute and persist coordinate source statistics
            try:
                from aquacontam.geo.geocoding import compute_coord_source_stats

                coord_stats = compute_coord_source_stats(merged, centroids, zip_col="raw_ZipCode")
                stats_path = interim_dir / "coord_source_stats.json"
                import json as _json

                stats_path.write_text(_json.dumps(coord_stats, indent=2))
                logger.info(
                    "Coord source stats: %s",
                    {
                        k: v
                        for k, v in coord_stats.get("system_counts", {}).items()
                        if not k.endswith("_pct")
                    },
                )
            except (KeyError, ValueError, AttributeError):
                logger.warning("Failed to compute coord source stats", exc_info=True)
        except (OSError, KeyError, ValueError):
            logger.warning("ZIP geocoding failed; coordinates may be incomplete", exc_info=True)
            if is_strict():
                raise

    # Second pass: propagate geocoded coordinates to remaining rows
    # (MI MPART, WA DOH, etc. that now share a PWSID with a geocoded row)
    if merged["latitude"].isna().any():
        coord_lookup2 = (
            merged[merged["latitude"].notna()]
            .sort_values(["source", "sample_date"], na_position="last")
            .groupby("pwsid")[["latitude", "longitude"]]
            .first()
        )
        missing2 = merged["latitude"].isna() & merged["pwsid"].isin(coord_lookup2.index)
        if missing2.any():
            merged.loc[missing2, "latitude"] = merged.loc[missing2, "pwsid"].map(
                coord_lookup2["latitude"]
            )
            merged.loc[missing2, "longitude"] = merged.loc[missing2, "pwsid"].map(
                coord_lookup2["longitude"]
            )
            logger.info("Backfilled coordinates for %d rows (post-geocoding)", missing2.sum())

    # --- Coordinate completeness summary (after all backfill passes) ---
    n_total = len(merged)
    n_with_coords = int(merged["latitude"].notna().sum())
    n_missing_coords = n_total - n_with_coords
    logger.info(
        "Coordinate summary: %d/%d rows have coordinates (%d still missing, %.1f%%)",
        n_with_coords,
        n_total,
        n_missing_coords,
        100 * n_missing_coords / max(n_total, 1),
    )
    if n_missing_coords > 0 and "source" in merged.columns:
        missing_by_source = (
            merged[merged["latitude"].isna()].groupby("source")["pwsid"].agg(["count", "nunique"])
        )
        missing_by_source.columns = ["rows_missing", "pws_missing"]
        for src, row in missing_by_source.iterrows():
            logger.warning(
                "  Missing coords — %s: %d rows (%d unique PWSIDs)",
                src,
                row["rows_missing"],
                row["pws_missing"],
            )

    # Post-geocoding validation: check coordinate quality
    from aquacontam.data.schema import validate_coordinates

    coord_rows = merged[merged["latitude"].notna()]
    if len(coord_rows) > 0:
        coord_valid = validate_coordinates(coord_rows)
        n_oob = int((~coord_valid).sum())
        if n_oob:
            logger.warning("%d rows have coordinates outside CONUS bounds (post-geocoding)", n_oob)

    merged.to_parquet(cached, index=False)
    logger.info("Merged WQ data: %d rows, cached to %s", len(merged), cached)
    return merged
