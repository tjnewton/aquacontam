"""Shared wrong-ZIP geocoding flag for paper figures and sensitivity analyses.

A system is *region-mismatched* when the EPA region derived from its PWSID
state prefix (administrative truth; drives the geographic split) disagrees
with the EPA region derived from its plotted coordinates (wrong-ZIP
mailing/operator-address centroids place some systems deep inside foreign
regions; see Supplementary S7b). The flag is purely diagnostic: the split
itself uses the PWSID prefix and is uncontaminated.

Coordinate-derived regions use exact state boundary polygons (US Census
cartographic boundaries, ``paper/assets/us_states_cb2023_5m.geojson``): a
point maps to every state polygon that covers it, falling back to the
nearest state within ``_NEAREST_MAX_DEG`` degrees for water/offshore points
(shoreline-clipped polygons miss lake- and ocean-jittered centroids). The
earlier bounding-box rule (``region_mismatch_mask_bbox_legacy``) flagged
correctly-located border systems whose neighbor state's loose rectangle
covered them; it is retained verbatim as the reproduction witness for
``paper/compute_misgeocode_sensitivity_v2.py`` gate B.

Both the geographic figures (``generate_figures*.py``) and the mis-geocode
sensitivity compute (``compute_misgeocode_sensitivity_v2.py``) must use this
one implementation so the excluded-count printed in captions and the frozen
sensitivity artifact (``misgeocode_sensitivity_v2.json``) agree by
construction.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree

from aquacontam.preprocessing._region_lookup import latlon_to_epa_region
from aquacontam.preprocessing.splits import STATE_TO_EPA_REGION

_STATES_GEOJSON = Path(__file__).resolve().parent / "assets" / "us_states_cb2023_5m.geojson"

_NEAREST_MAX_DEG = 3.0
"""Nearest-state fallback radius (degrees) for points no polygon covers.

Large enough that a wrong-ZIP centroid dropped in open water (Lake Michigan
is ~1.5 degrees wide) still resolves to a nearby state and gets flagged;
points farther than this from every state are effectively mid-ocean and are
never flagged (mirrors the legacy rule's "outside every bounding box" case).
"""


@lru_cache(maxsize=1)
def _load_state_polygons() -> tuple[STRtree, list[str], list[int | None]]:
    """Load the committed state polygons into an STRtree (cached).

    Returns
    -------
    tuple
        ``(tree, stusps, regions)`` where ``tree`` indexes the 56 state
        geometries and ``stusps[i]`` / ``regions[i]`` give the postal code
        and EPA region (``None`` for territories outside
        ``STATE_TO_EPA_REGION``, e.g. AS/MP) of tree geometry ``i``.
    """
    with open(_STATES_GEOJSON, encoding="utf-8") as f:
        fc = json.load(f)
    geoms = []
    stusps: list[str] = []
    regions: list[int | None] = []
    for feat in fc["features"]:
        geoms.append(shape(feat["geometry"]))
        code = feat["properties"]["STUSPS"]
        stusps.append(code)
        regions.append(STATE_TO_EPA_REGION.get(code))
    return STRtree(geoms), stusps, regions


def _coord_region_candidates(lats: np.ndarray, lons: np.ndarray) -> list[set[int]]:
    """EPA-region candidate sets for coordinate points.

    A point's candidates are the regions of every state polygon that covers
    it (boundary points can yield two); if none covers it, the nearest state
    within ``_NEAREST_MAX_DEG`` degrees (equidistant ties both count); else
    the empty set. Territories without an EPA region mapping contribute no
    candidate.

    Parameters
    ----------
    lats, lons : np.ndarray
        Coordinate arrays (WGS 84 decimal degrees), equal length.

    Returns
    -------
    list[set[int]]
        One candidate set per input point, in order.
    """
    tree, _stusps, regions = _load_state_polygons()
    points = shapely.points(lons, lats)
    candidates: list[set[int]] = [set() for _ in range(len(points))]

    covered = tree.query(points, predicate="covered_by")
    for pt_idx, geom_idx in zip(covered[0], covered[1], strict=True):
        region = regions[geom_idx]
        if region is not None:
            candidates[pt_idx].add(region)

    uncovered = np.array([i for i, c in enumerate(candidates) if not c], dtype=np.intp)
    if len(uncovered):
        near = tree.query_nearest(
            points[uncovered], max_distance=_NEAREST_MAX_DEG, all_matches=True
        )
        for sub_idx, geom_idx in zip(near[0], near[1], strict=True):
            region = regions[geom_idx]
            if region is not None:
                candidates[uncovered[sub_idx]].add(region)

    return candidates


def coord_epa_region_series(coords: pd.DataFrame) -> pd.Series:
    """Polygon-derived EPA region per row (for figure-coloring fallback).

    For each system the region of a covering state polygon, else the nearest
    state within ``_NEAREST_MAX_DEG`` degrees, else NaN. Where a boundary
    point yields multiple candidate regions, the smallest region number is
    used (deterministic; such points are visually on the drawn border).

    Parameters
    ----------
    coords : pd.DataFrame
        One row per system with ``latitude`` and ``longitude`` columns.

    Returns
    -------
    pd.Series
        Float series of EPA region numbers (NaN where none), on
        ``coords.index``.
    """
    cand = _coord_region_candidates(
        coords["latitude"].to_numpy(dtype=float),
        coords["longitude"].to_numpy(dtype=float),
    )
    out: pd.Series = pd.Series(
        [min(c) if c else np.nan for c in cand], index=coords.index, dtype=float
    )
    return out


def region_mismatch_mask(coords: pd.DataFrame) -> pd.Series:
    """Boolean mask: PWSID-prefix EPA region disagrees with coordinate region.

    A system is flagged when its prefix region is derivable, its coordinate
    yields at least one candidate region (covering state polygon, or nearest
    state within ``_NEAREST_MAX_DEG`` degrees), and the prefix region is not
    among the candidates. Points on shared border arcs get the benefit of
    the doubt (both adjacent regions are candidates).

    Parameters
    ----------
    coords : pd.DataFrame
        One row per system with ``pwsid``, ``latitude``, ``longitude``
        columns.

    Returns
    -------
    pd.Series
        True where both regions are derivable and disagree. Systems with a
        non-state PWSID prefix (e.g. synthetic WQP ids) or coordinates
        farther than ``_NEAREST_MAX_DEG`` degrees from every state polygon
        are never flagged.
    """
    prefix_region = coords["pwsid"].astype(str).str[:2].str.upper().map(STATE_TO_EPA_REGION)
    cand = _coord_region_candidates(
        coords["latitude"].to_numpy(dtype=float),
        coords["longitude"].to_numpy(dtype=float),
    )
    flagged = [
        pr == pr and bool(c) and int(pr) not in c  # pr == pr is a NaN check
        for pr, c in zip(prefix_region.tolist(), cand, strict=True)
    ]
    mask: pd.Series = pd.Series(flagged, index=coords.index, dtype=bool)
    return mask


def figure_plot_regions(coords: pd.DataFrame) -> pd.Series:
    """EPA region used to color each system on the split map.

    Prefix-derived region where the PWSID prefix is a known state code
    (split truth), else the polygon-derived coordinate region (systems with
    synthetic or tribal prefixes, whose pipeline split assignment is also
    coordinate-based). NaN regions are plotted in no split layer.

    Parameters
    ----------
    coords : pd.DataFrame
        One row per system with ``pwsid``, ``latitude``, ``longitude``
        columns.

    Returns
    -------
    pd.Series
        Float series of EPA region numbers (NaN where underivable), on
        ``coords.index``.
    """
    prefix_region: pd.Series = (
        coords["pwsid"].astype(str).str[:2].str.upper().map(STATE_TO_EPA_REGION)
    ).astype(float)
    missing = prefix_region.isna()
    if missing.any():
        prefix_region.loc[missing] = coord_epa_region_series(coords.loc[missing])
    return prefix_region


def region_mismatch_mask_bbox_legacy(coords: pd.DataFrame) -> pd.Series:
    """Legacy bounding-box variant of :func:`region_mismatch_mask`.

    The pre-2026-07-12 rule: coordinate region from loose, overlapping,
    first-match-wins state bounding boxes
    (:func:`aquacontam.preprocessing._region_lookup.latlon_to_epa_region`).
    Retained verbatim ONLY as the reproduction witness for
    ``compute_misgeocode_sensitivity_v2.py`` gate B (it must keep
    reproducing the frozen v1 count of 6,889 on the v1 input frame); the
    figures and the v2 sensitivity use :func:`region_mismatch_mask`.
    """
    prefix_region = coords["pwsid"].astype(str).str[:2].str.upper().map(STATE_TO_EPA_REGION)
    coord_region = pd.Series(
        [
            latlon_to_epa_region(float(lat), float(lon))
            for lat, lon in zip(coords["latitude"], coords["longitude"], strict=True)
        ],
        index=coords.index,
    )
    mask: pd.Series = (
        prefix_region.notna() & coord_region.notna() & (prefix_region != coord_region)
    )
    return mask
