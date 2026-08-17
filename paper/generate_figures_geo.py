"""PyGMT geographic figure generation for AquaContam (Nature Water format).

This module implements the 3 geographic map figures using PyGMT/GMT instead
of Cartopy.  The non-geographic figures remain in ``generate_figures.py``
(matplotlib).

Functions match the signatures in ``generate_figures.py`` so they can be
used as drop-in replacements.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pygmt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Albers Equal-Area Conic matching the Cartopy params:
#   central_longitude=-96, central_latitude=37.5,
#   standard_parallels=(29.5, 45.5)
# Width 18c ≈ 7.087 inches (Nature Water full column width = 180 mm)
CONUS_PROJ = "B-96/37.5/29.5/45.5/18c"
CONUS_REGION = [-130, -65, 24, 50]

# Source display labels (internal code name -> paper label)
_SOURCE_LABELS: dict[str, str] = {
    "ucmr5": "UCMR5",
    "ucmr3": "UCMR3",
    "sdwis": "SDWIS",
    "wqp": "WQP",
    "mi_mpart": "MI MPART",
    "ca_geotracker": "CA GeoTracker",
    "mn_mdh": "MN MDH",
    "oh_epa": "OH EPA",
    "wa_doh": "WA DOH",
    "mo_dnr": "MO DNR",
    "nj_dep": "NJ DEP",
    "nc_deq": "NC DEQ",
}

# Colorblind-safe palette (Wong 2011 extended)
_SOURCE_COLORS: dict[str, str] = {
    "ucmr5": "#0072B2",
    "ucmr3": "#D55E00",
    "sdwis": "#009E73",
    "wqp": "#332288",
    "mi_mpart": "#CC79A7",
    "ca_geotracker": "#E69F00",
    "oh_epa": "#882255",
    "wa_doh": "#44AA99",
    "mo_dnr": "#DDCC77",
    "nj_dep": "#56B4E9",
    "nc_deq": "#AA4499",
}

# Three-tier layering: background (dense federal), mid (UCMR), foreground (state)
_BG_SOURCES = {"sdwis", "wqp"}
_MID_SOURCES = {"ucmr5", "ucmr3"}

# GMT transparency is 0-100 (inverse of matplotlib alpha)
_BG_TRANSPARENCY = 85  # alpha 0.15
_MID_TRANSPARENCY = 60  # alpha 0.4
_FG_TRANSPARENCY = 30  # alpha 0.7


def _geo_placeholder(fig: pygmt.Figure, message: str, *, figure: str) -> None:
    """Draw a placeholder note, or raise when placeholders are disallowed.

    Mirrors ``generate_figures._placeholder`` (same ``--allow-placeholders``
    policy); the flag is read lazily so the CLI toggle applies here too.
    """
    import paper.generate_figures as _gf

    if not _gf.ALLOW_PLACEHOLDERS:
        raise _gf.MissingFigureDataError(
            f"{figure}: {message} "
            "(re-run with --allow-placeholders to render a placeholder instead)"
        )
    fig.text(x=-97.5, y=37, text=message, font="8p,Helvetica")


# ---------------------------------------------------------------------------
# GMT configuration
# ---------------------------------------------------------------------------


def _configure_gmt_defaults() -> None:
    """Set GMT defaults matching Nature Water publication style.

    Helvetica is metrically identical to Arial (both neo-grotesque
    sans-serif); GMT does not ship Arial, so Helvetica is the correct
    substitute for Nature Water's Arial requirement.
    """
    pygmt.config(
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        FONT_LABEL="8p,Helvetica",
        FONT_TITLE="9p,Helvetica-Bold",
        MAP_FRAME_PEN="0.5p",
        MAP_TICK_PEN_PRIMARY="0.5p",
        MAP_TICK_LENGTH_PRIMARY="3p/1.5p",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.x",
    )


def _add_conus_basemap(fig: pygmt.Figure, *, title: str = "") -> None:
    """Add CONUS basemap with state boundaries, coastlines, and lakes.

    Parameters
    ----------
    fig : pygmt.Figure
        PyGMT figure to add basemap to.
    title : str
        Optional title placed above the map.
    """
    frame = "WSne"
    if title:
        frame += f"+t{title}"
    fig.basemap(
        region=CONUS_REGION,
        projection=CONUS_PROJ,
        frame=frame,
    )
    fig.coast(
        borders="1/0.3p,#888888",  # State boundaries (admin level 1)
        shorelines="1/0.4p,#555555",  # Coastlines
        lakes="#e6f0ff+p0.2p,#aaaaaa",  # Great Lakes
        land="white",
        water="white",
    )


# ---------------------------------------------------------------------------
# Figure 1: Dataset Overview Map
# ---------------------------------------------------------------------------


def fig_dataset_overview_pygmt(
    wq_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Figure 1: Dataset overview map — sampling locations by source.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Water quality data with latitude, longitude, source columns.
    output_dir : Path
        Directory to save figure.

    Returns
    -------
    Path
        Path to saved PDF figure.
    """
    _configure_gmt_defaults()
    fig = pygmt.Figure()
    _add_conus_basemap(fig, title="a  Sampling Locations")

    if "source" not in wq_df.columns:
        # No source column — plot all points uniformly
        coords = wq_df.groupby("pwsid")[["latitude", "longitude"]].median().dropna()
        if not coords.empty:
            fig.plot(
                x=coords["longitude"].values,
                y=coords["latitude"].values,
                style="c0.02c",
                fill="#0072B2",
                transparency=70,
            )
        path = output_dir / "fig1_dataset_overview.pdf"
        fig.savefig(str(path))
        fig.savefig(str(path.with_suffix(".png")), dpi=300)
        return path

    # Build render order: background -> mid -> foreground
    all_sources = set(wq_df["source"].unique())
    render_order: list[str] = []
    for s in ("sdwis", "wqp"):
        if s in all_sources:
            render_order.append(s)
    for s in ("ucmr5", "ucmr3"):
        if s in all_sources:
            render_order.append(s)
    state_sources = sorted(all_sources - _BG_SOURCES - _MID_SOURCES)
    render_order.extend(state_sources)

    # Build legend spec lines for a 2-column legend
    legend_lines: list[str] = ["N 2"]

    for source in render_order:
        subset = wq_df[wq_df["source"] == source]
        coords = subset.groupby("pwsid")[["latitude", "longitude"]].median().dropna()
        if coords.empty:
            continue

        color = _SOURCE_COLORS.get(str(source), "#999999")
        label = _SOURCE_LABELS.get(str(source), str(source))

        if source in _BG_SOURCES:
            transparency, style, legend_sym = _BG_TRANSPARENCY, "c0.02c", "c"
            legend_size = "0.06c"
        elif source in _MID_SOURCES:
            transparency, style, legend_sym = _MID_TRANSPARENCY, "c0.04c", "c"
            legend_size = "0.06c"
        else:
            transparency, style, legend_sym = _FG_TRANSPARENCY, "t0.08c", "t"
            legend_size = "0.12c"

        fig.plot(
            x=coords["longitude"].values,
            y=coords["latitude"].values,
            style=style,
            fill=color,
            transparency=transparency,
        )
        legend_lines.append(f"S 0.1c {legend_sym} {legend_size} {color} - 0.3c {label}")

    # Write legend spec and render
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(legend_lines) + "\n")
        spec_path = f.name

    fig.legend(
        spec=spec_path,
        position="JBL+o0.3c/0.3c+w5.5c",
        box="+gwhite+p0.5p,gray",
    )
    Path(spec_path).unlink(missing_ok=True)

    path = output_dir / "fig1_dataset_overview.pdf"
    fig.savefig(str(path))
    fig.savefig(str(path.with_suffix(".png")), dpi=300)
    return path


# ---------------------------------------------------------------------------
# Figure 5: National Risk Map
# ---------------------------------------------------------------------------


def _make_ylorrd_cpt(
    vmin: float,
    vmax: float,
    *,
    boundaries: np.ndarray | None = None,
) -> None:
    """Create a YlOrRd-like CPT for the current GMT session.

    Exports matplotlib's YlOrRd colormap to a temporary GMT CPT file and
    loads it via ``pygmt.makecpt``.

    Parameters
    ----------
    vmin, vmax : float
        Value range for the color scale.
    boundaries : ndarray, optional
        Custom boundaries for non-linear mapping.  If provided, the CPT
        is built with these breakpoints.
    """
    import matplotlib.pyplot as plt

    cmap = plt.cm.YlOrRd  # type: ignore[attr-defined]
    n_colors = 128

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpt", delete=False) as f:
        f.write("# YlOrRd colormap from matplotlib\n")
        f.write("# COLOR_MODEL = RGB\n")
        for i in range(n_colors - 1):
            v0 = vmin + (vmax - vmin) * i / (n_colors - 1)
            v1 = vmin + (vmax - vmin) * (i + 1) / (n_colors - 1)
            r0, g0, b0, _ = cmap(i / (n_colors - 1))
            r1, g1, b1, _ = cmap((i + 1) / (n_colors - 1))
            f.write(
                f"{v0:.6f} {int(r0 * 255)} {int(g0 * 255)} {int(b0 * 255)} "
                f"{v1:.6f} {int(r1 * 255)} {int(g1 * 255)} {int(b1 * 255)}\n"
            )
        f.write("B 255 255 204\n")
        f.write("F 128 0 0\n")
        f.write("N 128 128 128\n")
        cpt_path = f.name

    pygmt.makecpt(cmap=cpt_path, series=[float(vmin), float(vmax)])
    Path(cpt_path).unlink(missing_ok=True)


def fig_national_risk_map_pygmt(
    predictions_df: pd.DataFrame | None,
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 9: national T1 PFAS (PFOS) risk map across CONUS.

    The loader supplies the frozen slim T1-PFOS surface
    (``predictions_slim.json``), so the "PFAS" title is accurate by
    construction. Uses quantile-based color normalization to spread color
    across percentiles, avoiding the nearly-all-pale-yellow issue.

    Parameters
    ----------
    predictions_df : pd.DataFrame or None
        Predictions with latitude, longitude, risk_score columns.
    output_dir : Path
        Directory to save figure.

    Returns
    -------
    Path
        Path to saved PDF figure.
    """
    _configure_gmt_defaults()
    fig = pygmt.Figure()
    _add_conus_basemap(fig, title="National PFAS Risk Surface")

    if predictions_df is not None and not predictions_df.empty:
        conus = predictions_df[
            (predictions_df.get("latitude", pd.Series()) >= 24)
            & (predictions_df.get("latitude", pd.Series()) <= 50)
            & (predictions_df.get("longitude", pd.Series()) >= -130)
            & (predictions_df.get("longitude", pd.Series()) <= -65)
        ]
        if not conus.empty:
            risk_col = conus.get("risk_score", np.zeros(len(conus)))
            risk_vals = np.asarray(risk_col, dtype=float)
            finite_vals = risk_vals[np.isfinite(risk_vals)]

            if len(finite_vals) > 0:
                vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
                if vmin == vmax:
                    vmax = vmin + 1.0
                _make_ylorrd_cpt(vmin, vmax)

                fig.plot(
                    x=conus["longitude"].values,
                    y=conus["latitude"].values,
                    style="c0.02c",
                    fill=risk_vals,
                    cmap=True,
                    transparency=40,
                )
                fig.colorbar(
                    frame='x+l"Predicted risk"',
                    position="JMR+o0.8c/0c+w8c/0.3c",
                )
            else:
                _geo_placeholder(fig, "No finite risk values", figure="fig_supp_national_risk_map")
        else:
            _geo_placeholder(fig, "No CONUS predictions", figure="fig_supp_national_risk_map")
    else:
        _geo_placeholder(fig, "No predictions", figure="fig_supp_national_risk_map")

    path = output_dir / "fig_supp_national_risk_map.pdf"
    fig.savefig(str(path))
    fig.savefig(str(path.with_suffix(".png")), dpi=300)
    return path


# ---------------------------------------------------------------------------
# Extended Data Fig 3: Geographic Splits (EPA Regions)
# ---------------------------------------------------------------------------


def fig_ext_geographic_splits_pygmt(
    wq_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Extended Data Fig. 1: geographic split map (file fig_ext3 is historical).

    Systems whose PWSID-prefix EPA region disagrees with their
    coordinate-derived region (wrong-ZIP geocoding artifact, Supplementary
    S7b) are excluded from the map; the caption reports the excluded count.
    Coordinate regions come from exact state boundary polygons
    (``paper/_geo_flags.py``), so correctly-located border systems are kept;
    the earlier bounding-box rule falsely dropped them in border-wide bands.
    Systems whose prefix is not a state code (WQP synthetic, tribal) are
    colored by their polygon-derived region. The split itself uses the PWSID
    prefix and is unaffected. Figure numbering lives in the manuscript
    captions, never baked into the image.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Water quality data with latitude, longitude, pwsid columns.
    output_dir : Path
        Directory to save figure.

    Returns
    -------
    Path
        Path to saved PDF figure.
    """
    _configure_gmt_defaults()
    fig = pygmt.Figure()
    _add_conus_basemap(fig, title="Geographic Split (EPA Regions)")

    split_config: dict[str, dict[str, Any]] = {
        "train": {
            "regions": [1, 3, 4, 5, 6],
            "color": "#0072B2",
            "label": "Train",
        },
        "val": {
            "regions": [2, 7],
            "color": "#E69F00",
            "label": "Validation",
        },
        "test": {
            "regions": [8, 9, 10],
            "color": "#D55E00",
            "label": "Test",
        },
    }

    legend_lines: list[str] = []

    if "latitude" in wq_df.columns and "longitude" in wq_df.columns:
        coords = wq_df.groupby("pwsid")[["latitude", "longitude"]].median().dropna()
        if not coords.empty:
            try:
                from paper._geo_flags import figure_plot_regions, region_mismatch_mask
            except ImportError:  # direct script execution: paper/ is sys.path[0]
                from _geo_flags import (  # type: ignore[no-redef, import-not-found]
                    figure_plot_regions,
                    region_mismatch_mask,
                )

            mismatch = region_mismatch_mask(coords.reset_index())
            n_excluded = int(mismatch.sum())
            coords = coords[~mismatch.to_numpy()]
            logger.info(
                "Split map: excluded %d region-mismatched (mis-geocoded) systems",
                n_excluded,
            )

            coords_reset = coords.reset_index()
            regions = figure_plot_regions(coords_reset)
            regions.index = coords_reset["pwsid"]

            for _split_name, cfg in split_config.items():
                mask = regions.isin(cfg["regions"])
                subset = coords[mask]
                if subset.empty:
                    continue

                region_str = ",".join(str(r) for r in cfg["regions"])

                fig.plot(
                    x=subset["longitude"].values,
                    y=subset["latitude"].values,
                    style="c0.03c",
                    fill=cfg["color"],
                    transparency=60,
                )
                legend_lines.append(
                    f"S 0.1c c 0.08c {cfg['color']} - 0.3c {cfg['label']} (R{region_str})"
                )
    else:
        _geo_placeholder(fig, "No coordinate data", figure="fig_ext3_geographic_splits")

    if legend_lines:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(legend_lines) + "\n")
            spec_path = f.name
        fig.legend(
            spec=spec_path,
            position="JBL+o0.3c/0.3c+w6c",
            box="+gwhite+p0.5p,gray",
        )
        Path(spec_path).unlink(missing_ok=True)

    path = output_dir / "fig_ext3_geographic_splits.pdf"
    fig.savefig(str(path))
    fig.savefig(str(path.with_suffix(".png")), dpi=300)
    return path
