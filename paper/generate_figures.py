#!/usr/bin/env python
"""Generate publication figures for AquaContam (Nature Water format).

Usage::

    python paper/generate_figures.py --results results/ --output paper/figures/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aquacontam._constants import MODEL_DISPLAY_NAMES

logger = logging.getLogger(__name__)

# Use non-interactive backend for CI/scripts
matplotlib.use("Agg")

STYLE_PATH = Path(__file__).parent / "style.mplstyle"

# Colorblind-safe palette (Wong 2011) + hatching for grayscale
MODEL_COLORS: dict[str, str] = {
    "Dummy": "#BBBBBB",
    "Logistic Reg.": "#999999",
    "XGBoost": "#0072B2",
    "XGBoost (default)": "#56B4E9",
    "Random Forest": "#009E73",
    "MLP": "#D55E00",
    "CNN1D": "#CC79A7",
    "LightGBM": "#E69F00",
    "CatBoost": "#F0E442",
    "Tobit": "#332288",
    "AFT (Weibull)": "#882255",
    "GCN (linear fallback)": "#44AA99",
    "GraphSAGE (linear fallback)": "#117733",
    "Deep Tobit": "#661100",
    "TabPFN": "#AA4499",
    "ICP": "#88CCEE",
}
MODEL_HATCHES: dict[str, str] = {
    "Dummy": "...",
    "Logistic Reg.": "///",
    "XGBoost": "",
    "XGBoost (default)": "---",
    "Random Forest": "\\\\\\",
    "MLP": "xxx",
    "CNN1D": "+++",
    "LightGBM": "OO",
    "CatBoost": "**",
    "Tobit": "||",
    "AFT (Weibull)": "..",
    "GCN (linear fallback)": "//",
    "GraphSAGE (linear fallback)": "\\\\",
    "Deep Tobit": "--",
    "TabPFN": "oo",
    "ICP": "xx",
}


def _pub_name(code_name: str) -> str:
    """Convert internal model code name to publication display name."""
    return MODEL_DISPLAY_NAMES.get(code_name, code_name)


def _load_style() -> None:
    """Load the Nature Water matplotlib style."""
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))


class MissingFigureDataError(RuntimeError):
    """A figure's required input is missing and a placeholder panel would ship."""


# Loud-fail policy: canonical regeneration (from the frozen archive) must never
# silently ship a placeholder panel. ``--allow-placeholders`` flips this for
# exploratory runs on partial results directories.
ALLOW_PLACEHOLDERS = False


def _placeholder(ax: Any, message: str, *, figure: str) -> None:
    """Draw a placeholder panel, or raise when placeholders are disallowed."""
    if not ALLOW_PLACEHOLDERS:
        raise MissingFigureDataError(
            f"{figure}: {message.replace(chr(10), ' ')} "
            "(re-run with --allow-placeholders to render a placeholder instead)"
        )
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
    )


def fig_dataset_overview(
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
        Path to saved figure.
    """
    try:
        from paper.generate_figures_geo import fig_dataset_overview_pygmt

        return fig_dataset_overview_pygmt(wq_df, output_dir)
    except ImportError:
        pass

    _load_style()
    proj = _get_conus_projection()
    if proj is not None:
        import cartopy.crs as ccrs

        fig, ax = plt.subplots(1, 1, figsize=(7.087, 4.5), subplot_kw={"projection": proj})
        data_crs = ccrs.PlateCarree()
        ax.set_extent([-125, -66, 24, 50], crs=data_crs)  # type: ignore[attr-defined]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.087, 4.5))
        data_crs = None

    # Map internal source column values to human-readable names
    _SOURCE_LABELS: dict[str, str] = {
        # Federal
        "ucmr5": "UCMR5",
        "ucmr3": "UCMR3",
        "sdwis": "SDWIS",
        "wqp": "WQP",
        # State
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
    source_colors: dict[str, str] = {
        # Federal
        "ucmr5": "#0072B2",
        "ucmr3": "#D55E00",
        "sdwis": "#009E73",
        "wqp": "#332288",
        # State
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
    # Everything else is foreground (state databases)

    # Add state boundaries as basemap
    _add_conus_basemap(ax)

    if "source" in wq_df.columns:
        # Render order: background → mid → foreground
        _render_order: list[str] = []
        all_sources = set(wq_df["source"].unique())
        # Background first (lowest zorder)
        for s in ("sdwis", "wqp"):
            if s in all_sources:
                _render_order.append(s)
        # Mid-layer next
        for s in ("ucmr5", "ucmr3"):
            if s in all_sources:
                _render_order.append(s)
        # Foreground: state sources last (highest zorder)
        state_sources = sorted(all_sources - _BG_SOURCES - _MID_SOURCES)
        _render_order.extend(state_sources)

        for zorder, source in enumerate(_render_order, start=1):
            subset = wq_df[wq_df["source"] == source]
            coords = subset.groupby("pwsid")[["latitude", "longitude"]].median()
            label = _SOURCE_LABELS.get(str(source), str(source))
            color = source_colors.get(str(source), "#999999")

            if source in _BG_SOURCES:
                alpha, size, marker = 0.15, 1, "o"
            elif source in _MID_SOURCES:
                alpha, size, marker = 0.4, 2, "o"
            else:
                alpha, size, marker = 0.7, 6, "^"

            scatter_kw: dict[str, Any] = {}
            if data_crs is not None:
                scatter_kw["transform"] = data_crs
            ax.scatter(
                coords["longitude"],
                coords["latitude"],
                s=size,
                alpha=alpha,
                label=label,
                color=color,
                marker=marker,
                zorder=zorder,
                **scatter_kw,
            )
        ax.legend(
            loc="lower left",
            markerscale=2,
            fontsize=5.5,
            ncol=2,
            columnspacing=1,
            handletextpad=0.3,
            framealpha=0.9,
        )
    else:
        coords = wq_df.groupby("pwsid")[["latitude", "longitude"]].median()
        scatter_kw_else: dict[str, Any] = {}
        if data_crs is not None:
            scatter_kw_else["transform"] = data_crs
        ax.scatter(
            coords["longitude"],
            coords["latitude"],
            s=2,
            alpha=0.3,
            color="#0072B2",
            zorder=1,
            **scatter_kw_else,
        )

    ax.set_title("a  Sampling Locations", loc="left")
    if data_crs is None:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-130, -65)
        ax.set_ylim(24, 50)

    path = output_dir / "fig1_dataset_overview.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_data_distribution(
    wq_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 8: detection rates and concentration distributions."""
    _load_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.087, 3.0))

    # Panel a: Detection rates by analyte
    ax = axes[0]
    analyte_stats = wq_df.groupby("analyte")["censored"].agg(["count", "mean"])
    analyte_stats["detection_rate"] = 1 - analyte_stats["mean"]
    top = analyte_stats.nlargest(15, "count")
    ax.barh(range(len(top)), top["detection_rate"], color="#0072B2")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Detection Rate")
    ax.set_title("a  Detection Rates", loc="left")

    # Panel b: Concentration distributions (detected only)
    ax = axes[1]
    detected = wq_df[~wq_df["censored"]]
    if not detected.empty and "concentration" in detected.columns:
        conc_vals = detected["concentration"]
        conc_vals = conc_vals[conc_vals > 0]
        if not conc_vals.empty:
            ax.hist(np.log10(conc_vals), bins=50, color="#D55E00", alpha=0.7)
    ax.set_xlabel("log10(Concentration)")
    ax.set_ylabel("Count")
    ax.set_title("b  Concentration Distribution", loc="left")

    fig.tight_layout()
    path = output_dir / "fig_ext5_data_distribution.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


# Key models to show in main-text Fig 2 (8 models for legibility)
MAIN_FIGURE_MODELS: list[str] = [
    "Dummy",
    "Logistic Reg.",
    "XGBoost",
    "Random Forest",
    "MLP",
    "CatBoost",
    "TabPFN",
    "Deep Tobit",
]


def fig_benchmark_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    *,
    main_only: bool = True,
) -> Path:
    """Extended Data Fig. 5: benchmark results — AUPRC and RMSE panels.

    Uses publication-ready model names, colorblind-safe palette with hatching
    for grayscale legibility, and bootstrap error bars when available.

    Parameters
    ----------
    results : list[dict]
        Benchmark result dicts.
    output_dir : Path
        Directory to save figure.
    main_only : bool
        If True, show only 8 key models (for main figure). Full version
        goes to Extended Data.
    """
    _load_style()

    if not results:
        fig, ax = plt.subplots(1, 1, figsize=(7.087, 3.5))
        _placeholder(ax, "No results", figure="fig_ext11_benchmark_results")
        path = output_dir / "fig_ext11_benchmark_results.pdf"
        fig.savefig(path)
        plt.close(fig)
        return path

    df = pd.DataFrame(results)

    # Map code names to publication names
    df["display_name"] = df["model"].map(_pub_name)

    # Split tasks into classification and regression
    classification_tasks = sorted(t for t in df["task"].unique() if t != "T2")
    regression_tasks = sorted(t for t in df["task"].unique() if t == "T2")
    has_regression = bool(regression_tasks)

    ncols = 2 if has_regression else 1
    widths = [3, 1] if has_regression else [1]
    fig, axes = plt.subplots(1, ncols, figsize=(7.087, 3.5), gridspec_kw={"width_ratios": widths})
    if ncols == 1:
        axes = [axes]

    # Deduplicated ordered model list (preserving display name order)
    seen: set[str] = set()
    display_models: list[str] = []
    for name in df["display_name"]:
        if name not in seen:
            seen.add(name)
            display_models.append(name)

    # Filter to key models for main figure
    if main_only:
        display_models = [m for m in display_models if m in MAIN_FIGURE_MODELS]

    # Panel a: Classification (AUPRC)
    ax = axes[0]
    x = np.arange(len(classification_tasks))
    width = 0.8 / max(len(display_models), 1)

    for i, display_name in enumerate(display_models):
        model_df = df[df["display_name"] == display_name]
        values: list[float] = []
        ci_lo: list[float] = []
        ci_hi: list[float] = []
        for task in classification_tasks:
            task_rows = model_df[model_df["task"] == task]
            if not task_rows.empty:
                metrics = task_rows.iloc[0]["metrics"]
                val = metrics.get("auprc", metrics.get("macro_auprc", 0))
                values.append(float(val) if val == val else 0.0)  # NaN guard
                # Bootstrap CI from metadata
                bootstrap_ci = task_rows.iloc[0].get("metadata", {}).get("bootstrap_ci", {})
                auprc_ci = bootstrap_ci.get("auprc", {})
                ci_lo.append(float(auprc_ci.get("ci_lower", val)))
                ci_hi.append(float(auprc_ci.get("ci_upper", val)))
            else:
                values.append(0.0)
                ci_lo.append(0.0)
                ci_hi.append(0.0)

        color = MODEL_COLORS.get(display_name, "#666666")
        hatch = MODEL_HATCHES.get(display_name, "")
        yerr_lo = [max(0, v - lo) for v, lo in zip(values, ci_lo, strict=False)]
        yerr_hi = [max(0, hi - v) for v, hi in zip(values, ci_hi, strict=False)]
        has_ci = any(lo > 0 or hi > 0 for lo, hi in zip(yerr_lo, yerr_hi, strict=False))

        ax.bar(
            x + i * width,
            values,
            width,
            label=display_name,
            color=color,
            hatch=hatch,
            edgecolor="white" if not hatch else "black",
            linewidth=0.3,
            yerr=[yerr_lo, yerr_hi] if has_ci else None,
            capsize=2 if has_ci else 0,
            error_kw={"linewidth": 0.5},
        )

    ax.set_xticks(x + width * (len(display_models) - 1) / 2)
    ax.set_xticklabels(classification_tasks, fontsize=8)
    ax.set_ylabel("AUPRC")
    ax.set_title("a  Classification Tasks", loc="left")
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.15))
    ax.legend(fontsize=6, ncol=2, loc="upper right")

    # Panel b: Regression (RMSE) — lower is better
    if has_regression:
        ax = axes[1]
        x = np.arange(len(regression_tasks))
        for i, display_name in enumerate(display_models):
            model_df = df[df["display_name"] == display_name]
            values = []
            for task in regression_tasks:
                task_rows = model_df[model_df["task"] == task]
                if not task_rows.empty:
                    metrics = task_rows.iloc[0]["metrics"]
                    val = metrics.get("rmse", 0)
                    values.append(float(val) if val == val else 0.0)
                else:
                    values.append(0.0)
            color = MODEL_COLORS.get(display_name, "#666666")
            hatch = MODEL_HATCHES.get(display_name, "")
            ax.bar(
                x + i * width,
                values,
                width,
                color=color,
                hatch=hatch,
                edgecolor="white" if not hatch else "black",
                linewidth=0.3,
            )

        ax.set_xticks(x + width * (len(display_models) - 1) / 2)
        ax.set_xticklabels(regression_tasks, fontsize=8)
        ax.set_ylabel("RMSE (µg/L)")
        ax.set_title("b  Regression", loc="left")

    fig.tight_layout()
    path = output_dir / "fig_ext11_benchmark_results.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


# Human-readable feature name mapping for SHAP plots
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "n_samples": "Monitoring intensity",
    "population_served": "Population served",
    "pct_wetland_5km": "Wetland fraction (5 km)",
    "pct_wetland": "Wetland fraction",
    "pct_forest": "Forest fraction",
    "pct_forest_5km": "Forest fraction (5 km)",
    "pct_developed_high": "Developed (high intensity)",
    "pct_developed_med": "Developed (medium intensity)",
    "pct_developed_low": "Developed (low intensity)",
    "pct_agriculture_crop": "Cropland fraction",
    "pct_agriculture_pasture": "Pasture fraction",
    "dist_nearest_industrial": "Dist. to industrial facility",
    "dist_nearest_military": "Dist. to military installation",
    "dist_nearest_wwtp": "Dist. to WWTP",
    "dist_nearest_airport": "Dist. to airport",
    "dist_nearest_landfill": "Dist. to landfill",
    "n_industrial_5km": "Industrial facilities (5 km)",
    "n_military_5km": "Military facilities (5 km)",
    "n_wwtp_5km": "WWTP count (5 km)",
    "n_airport_5km": "Airports (5 km)",
    "n_landfill_5km": "Landfills (5 km)",
    "pct_people_of_color": "% People of color",
    "pct_low_income": "% Low income",
    "pct_less_hs": "% Less than HS education",
    "pct_limited_english": "% Limited English",
    "ej_index": "EJ Index",
    "mean_detection_limit": "Mean detection limit",
    "log_population_served": "Population served (log)",
    "source_water_type_GW": "Source: Groundwater",
    "source_water_type_SW": "Source: Surface water",
    "system_type_CWS": "System: Community",
    "system_type_NTNCWS": "System: Non-transient",
    "owner_type_Private": "Owner: Private",
    "owner_type_Local government": "Owner: Local govt",
}


def _readable_feature_name(raw_name: str) -> str:
    """Convert raw feature name to human-readable label."""
    if raw_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[raw_name]
    # One-hot missingness indicators are data-source provenance proxies
    if raw_name.lower().endswith("_nan"):
        prefix = raw_name[: -len("_nan")].replace("_", " ").capitalize()
        return f"{prefix} missing (source proxy)"
    # Handle aquifer type one-hot features
    if raw_name.startswith("aquifer_type_"):
        return f"Aquifer: {raw_name[len('aquifer_type_') :]}"
    if raw_name.startswith("aquifer_lithology_"):
        return f"Lithology: {raw_name[len('aquifer_lithology_') :]}"
    if raw_name.startswith("aquifer_confinement_"):
        return f"Confinement: {raw_name[len('aquifer_confinement_') :]}"
    # Fallback: replace underscores with spaces, title case
    return raw_name.replace("_", " ").title()


def fig_feature_importance(
    importance_df: pd.DataFrame | None,
    output_dir: Path,
) -> Path:
    """Supplementary asset: feature importance — top 20 features with readable names."""
    _load_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.087, 4.0))

    if importance_df is not None and not importance_df.empty:
        top20 = importance_df.head(20)
        readable_labels = [_readable_feature_name(name) for name in top20.index]
        ax.barh(range(len(top20)), top20.values, color="#009E73")
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(readable_labels)
        ax.invert_yaxis()
    else:
        _placeholder(ax, "No data", figure="fig_supp_feature_importance")

    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (T1 XGBoost)", loc="left")
    fig.tight_layout()

    path = output_dir / "fig_supp_feature_importance.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def _get_conus_projection() -> Any:
    """Return a cartopy Albers Equal Area projection for CONUS."""
    try:
        import cartopy.crs as ccrs

        return ccrs.AlbersEqualArea(
            central_longitude=-96,
            central_latitude=37.5,
            standard_parallels=(29.5, 45.5),
        )
    except ImportError:
        return None


def _add_conus_basemap(ax: plt.Axes) -> None:
    """Add US state boundaries, coastlines, and lakes to a map axes.

    Uses cartopy for publication-quality maps with proper projection.
    Falls back to a geopandas country boundary or a CONUS bounding
    rectangle if cartopy/geopandas is unavailable.
    """
    try:
        import cartopy.feature as cfeature

        ax.add_feature(  # type: ignore[attr-defined]
            cfeature.STATES, linewidth=0.3, edgecolor="#888888", facecolor="none", zorder=0
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#555555", zorder=0)  # type: ignore[attr-defined]
        ax.add_feature(  # type: ignore[attr-defined]
            cfeature.LAKES, linewidth=0.2, edgecolor="#aaaaaa", facecolor="#e6f0ff", zorder=0
        )
        ax.add_feature(  # type: ignore[attr-defined]
            cfeature.BORDERS, linewidth=0.4, edgecolor="#555555", linestyle="--", zorder=0
        )
        return
    except (ImportError, AttributeError):
        pass

    try:
        import geopandas as gpd

        cache_dir = Path.home() / ".cache" / "aquacontam"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "ne_110m_admin_0_countries.gpkg"

        if cache_file.exists():
            world = gpd.read_file(cache_file)
        else:
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            world = gpd.read_file(url)
            world.to_file(cache_file, driver="GPKG")

        usa = world[world["ISO_A3"] == "USA"]
        usa.boundary.plot(ax=ax, linewidth=0.3, color="gray", zorder=0)
    except Exception:
        # Fallback: draw CONUS bounding box
        from matplotlib.patches import Rectangle

        rect = Rectangle((-125, 24), 60, 26, linewidth=0.5, edgecolor="gray", facecolor="none")
        ax.add_patch(rect)


def fig_national_risk_map(
    predictions_df: pd.DataFrame | None,
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 9: national T1 PFAS (PFOS) risk map.

    Draws the frozen slim T1-PFOS prediction surface (``predictions_slim.json``);
    the loader guarantees single-task input so the "PFAS" title is accurate.
    Clips to CONUS bounds, adds US state boundaries, and uses quantile-based
    color normalization to avoid nearly-all-pale-yellow issue when risk scores
    are dominated by near-zero values.
    """
    try:
        from paper.generate_figures_geo import fig_national_risk_map_pygmt

        return fig_national_risk_map_pygmt(predictions_df, output_dir)
    except ImportError:
        pass

    from matplotlib.colors import BoundaryNorm

    _load_style()
    proj = _get_conus_projection()
    if proj is not None:
        import cartopy.crs as ccrs

        fig, ax = plt.subplots(1, 1, figsize=(7.087, 4.0), subplot_kw={"projection": proj})
        data_crs = ccrs.PlateCarree()
        ax.set_extent([-125, -66, 24, 50], crs=data_crs)  # type: ignore[attr-defined]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.087, 4.0))
        data_crs = None

    if predictions_df is not None and not predictions_df.empty:
        # Clip to CONUS bounds before plotting
        conus = predictions_df[
            (predictions_df.get("latitude", pd.Series()) >= 24)
            & (predictions_df.get("latitude", pd.Series()) <= 50)
            & (predictions_df.get("longitude", pd.Series()) >= -130)
            & (predictions_df.get("longitude", pd.Series()) <= -65)
        ]
        if not conus.empty:
            _add_conus_basemap(ax)

            risk_col = conus.get("risk_score", np.zeros(len(conus)))
            risk_vals = np.asarray(risk_col, dtype=float)

            # Quantile-based normalization to spread color across percentiles
            percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
            boundaries = np.percentile(risk_vals[np.isfinite(risk_vals)], percentiles)
            # Ensure strictly increasing boundaries
            boundaries = np.unique(boundaries)
            if len(boundaries) < 3:
                # Fallback to linear if too few unique values
                boundaries = np.linspace(np.nanmin(risk_vals), np.nanmax(risk_vals), 10)

            cmap = plt.cm.YlOrRd  # type: ignore[attr-defined]
            norm = BoundaryNorm(boundaries, cmap.N, clip=True)

            scatter_kw: dict[str, Any] = {}
            if data_crs is not None:
                scatter_kw["transform"] = data_crs
            sc = ax.scatter(
                conus["longitude"],
                conus["latitude"],
                c=risk_vals,
                s=1,
                cmap=cmap,
                norm=norm,
                alpha=0.6,
                zorder=1,
                **scatter_kw,
            )
            cbar = plt.colorbar(sc, ax=ax, label="Predicted risk", shrink=0.85)
            cbar.ax.tick_params(labelsize=6)
        else:
            _placeholder(ax, "No CONUS predictions", figure="fig_supp_national_risk_map")
    else:
        _placeholder(ax, "No predictions", figure="fig_supp_national_risk_map")

    ax.set_title("National PFAS Risk Surface", loc="left")
    if data_crs is None:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-130, -65)
        ax.set_ylim(24, 50)
    fig.tight_layout()

    path = output_dir / "fig_supp_national_risk_map.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


# Human-readable labels for equity analysis demographic dimensions
EQUITY_DISPLAY_NAMES: dict[str, str] = {
    "pct_people_of_color": "People of color",
    "pct_low_income": "Low income",
    "pct_less_hs": "Less than HS education",
    "pct_limited_english": "Limited English",
    "people_of_color": "People of color",
    "low_income": "Low income",
    "less_hs": "Less than HS education",
    "limited_english": "Limited English",
    "education": "Education attainment",
    "income": "Income",
    "linguistic_isolation": "Linguistic isolation",
}


def fig_equity_analysis(
    equity_df: pd.DataFrame | None,
    output_dir: Path,
    burden_ci: dict[str, dict] | None = None,
) -> Path:
    """Supplementary asset: equity analysis — burden ratios and demographics.

    When ``burden_ci`` (from group_burden_ci.json) is supplied, each burden-ratio bar
    carries a bootstrap 95% CI error bar, so the figure shows uncertainty, not
    only significance stars.
    """
    _load_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.087, 3.5))

    if equity_df is not None and not equity_df.empty:
        if "group" in equity_df.columns and "burden_ratio" in equity_df.columns:
            # Map raw group names to readable labels
            display_groups = [
                EQUITY_DISPLAY_NAMES.get(g, g.replace("_", " ").title())
                for g in equity_df["group"]
            ]
            colors = ["#CC79A7" if br > 1.0 else "#999999" for br in equity_df["burden_ratio"]]
            # Asymmetric bootstrap-CI error bars where a CI is available.
            xerr = None
            if burden_ci:
                lo, hi = [], []
                any_ci = False
                for g, br in zip(equity_df["group"], equity_df["burden_ratio"], strict=False):
                    ci = (burden_ci.get(g) or {}).get("burden_ratio_ci")
                    if ci and len(ci) == 2:
                        lo.append(max(0.0, float(br) - float(ci[0])))
                        hi.append(max(0.0, float(ci[1]) - float(br)))
                        any_ci = True
                    else:
                        lo.append(0.0)
                        hi.append(0.0)
                if any_ci:
                    xerr = [lo, hi]
            bars = ax.barh(
                display_groups,
                equity_df["burden_ratio"],
                color=colors,
                xerr=xerr,
                error_kw={"ecolor": "#333333", "elinewidth": 0.8, "capsize": 3},
            )
            ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.5, label="Parity")
            ax.set_xlabel("Burden Ratio (high-burden / reference)")

            # Add significance annotations if p-value column exists
            p_col = None
            for col in ["p_fdr", "p_value_fdr", "p_adjusted"]:
                if col in equity_df.columns:
                    p_col = col
                    break
            if p_col is not None:
                for bar, p_val in zip(bars, equity_df[p_col], strict=False):
                    if pd.notna(p_val):
                        stars = (
                            "***"
                            if p_val < 0.001
                            else "**"
                            if p_val < 0.01
                            else "*"
                            if p_val < 0.05
                            else ""
                        )
                        if stars:
                            ax.text(
                                bar.get_width() + 0.05,
                                bar.get_y() + bar.get_height() / 2,
                                stars,
                                ha="left",
                                va="center",
                                fontsize=10,
                                fontweight="bold",
                            )

            # Future: add monitoring intensity annotation if available
    else:
        _placeholder(
            ax,
            "EJScreen demographic data required to populate",
            figure="fig_supp_equity_analysis",
        )

    ax.set_title("Environmental Justice Analysis", loc="left")
    fig.tight_layout()

    path = output_dir / "fig_supp_equity_analysis.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_transfer_learning(
    results: list[dict[str, Any]] | None,
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 1: transfer learning — T5 grouped bar chart.

    Shows source→target AUROC and AUPRC across model families.
    """
    _load_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.087, 3.0))

    if not results:
        for ax in axes:
            _placeholder(ax, "No data", figure="fig_ext6_transfer_learning")
        fig.tight_layout()
        path = output_dir / "fig_ext6_transfer_learning.pdf"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"))
        plt.close(fig)
        return path

    df = pd.DataFrame(results)
    df["display_name"] = df["model"].map(_pub_name)
    display_models = list(dict.fromkeys(df["display_name"]))
    x = np.arange(len(display_models))

    for panel_idx, metric_key in enumerate(["auroc", "auprc"]):
        ax = axes[panel_idx]
        values = []
        for display_name in display_models:
            model_rows = df[df["display_name"] == display_name]
            if not model_rows.empty:
                metrics = model_rows.iloc[0].get("metrics", {})
                values.append(metrics.get(metric_key, 0.0))
            else:
                values.append(0.0)
        bars = ax.bar(
            x,
            values,
            color=[MODEL_COLORS.get(m, "#666666") for m in display_models],
            hatch=[MODEL_HATCHES.get(m, "") for m in display_models],
            width=0.6,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(display_models, rotation=30, ha="right", fontsize=7)
        label = "a" if panel_idx == 0 else "b"
        metric_label = "AUROC" if metric_key == "auroc" else "AUPRC"
        ax.set_title(f"{label}  T5 Transfer {metric_label}", loc="left")
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1.0)
        # Add value labels on bars
        for bar, val in zip(bars, values, strict=False):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    fig.tight_layout()
    path = output_dir / "fig_ext6_transfer_learning.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_temporal_prediction(
    results: list[dict[str, Any]] | None,
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 2: temporal prediction — T7 grouped bar chart.

    Shows test/persistent/new system AUROC across model families.
    """
    _load_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.087, 3.5))

    if not results:
        _placeholder(ax, "No data", figure="fig_ext7_temporal_prediction")
        fig.tight_layout()
        path = output_dir / "fig_ext7_temporal_prediction.pdf"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"))
        plt.close(fig)
        return path

    df = pd.DataFrame(results)
    df["display_name"] = df["model"].map(_pub_name)
    display_models = list(dict.fromkeys(df["display_name"]))
    colors = ["#0072B2", "#D55E00", "#009E73"]
    # Map: (split_metrics_key, metric_name, display_label)
    split_keys = [
        ("test", "auroc", "Test"),
        ("test_persistent", "auroc", "Persistent"),
        ("test_new", "auroc", "New Systems"),
    ]

    x = np.arange(len(display_models))
    width = 0.25

    for i, (split_name, metric_key, split_label) in enumerate(split_keys):
        values = []
        for display_name in display_models:
            model_rows = df[df["display_name"] == display_name]
            if not model_rows.empty:
                row = model_rows.iloc[0]
                val = 0.0
                # Primary approach: look in split_metrics dict
                split_metrics = row.get("split_metrics", {})
                if isinstance(split_metrics, dict) and split_name in split_metrics:
                    val = split_metrics[split_name].get(metric_key, 0.0)
                elif split_name == "test":
                    # Fallback: top-level metrics for the "test" key
                    val = row.get("metrics", {}).get(metric_key, 0.0)
                values.append(val)
            else:
                values.append(0.0)
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=split_label,
            color=colors[i],
        )
        for bar, val in zip(bars, values, strict=False):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels(display_models, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_title("T7 Temporal Prediction (UCMR3 → UCMR5)", loc="left")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = output_dir / "fig_ext7_temporal_prediction.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_ext_correlation_matrix(
    feature_df: pd.DataFrame | None,
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 3: feature correlation matrix (file fig_ext1 is historical)."""
    _load_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.087, 6.5))

    if feature_df is not None and not feature_df.empty:
        # Select numeric columns and limit to top 30 by variance
        numeric = feature_df.select_dtypes(include=[np.number])
        if numeric.shape[1] > 30:
            top_cols = numeric.var().nlargest(30).index
            numeric = numeric[top_cols]
        corr = numeric.corr()
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=4)
        ax.set_yticklabels(corr.columns, fontsize=4)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    else:
        _placeholder(ax, "No feature data", figure="fig_ext1_correlation_matrix")

    # Figure numbering lives in the manuscript captions, never baked into the image.
    ax.set_title("Feature Correlation Matrix", loc="left", fontsize=8)
    fig.tight_layout()

    path = output_dir / "fig_ext1_correlation_matrix.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_ext_per_analyte_roc(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 4: per-analyte AUROC bars for the T3 multilabel task.

    The historical ``fig_ext2`` filename is kept to avoid churning committed
    figure references. Analytes with no evaluable AUROC (all models NaN, e.g.
    HFPO-DA with zero detections in the test regions) are annotated as not
    evaluable rather than drawn as zero-height bars.
    """
    _load_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.087, 4.5))

    # Extract per-analyte AUROC from T3 results
    t3_results = [r for r in results if r.get("task") == "T3"]
    if t3_results:
        analytes = ["PFOS", "PFOA", "PFBS", "PFHxS", "HFPO-DA"]
        display_models = list(dict.fromkeys(_pub_name(r["model"]) for r in t3_results))
        x = np.arange(len(analytes))
        width = 0.8 / max(len(display_models), 1)

        seen_values: dict[str, list[float]] = {a: [] for a in analytes}
        for i, display_name in enumerate(display_models):
            model_rows = [r for r in t3_results if _pub_name(r["model"]) == display_name]
            if not model_rows:
                continue
            metrics = model_rows[0].get("metrics", {})
            values = [metrics.get(f"auroc_{a}", float("nan")) for a in analytes]
            for analyte, v in zip(analytes, values, strict=True):
                seen_values[analyte].append(v)
            color = MODEL_COLORS.get(display_name, "#666666")
            hatch = MODEL_HATCHES.get(display_name, "")
            ax.bar(
                x + i * width,
                [v if not np.isnan(v) else 0.0 for v in values],
                width,
                label=display_name,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.3,
            )

        group_center = x + width * (len(display_models) - 1) / 2
        for j, analyte in enumerate(analytes):
            vals = seen_values[analyte]
            if vals and all(np.isnan(v) for v in vals):
                # e.g. HFPO-DA: zero detections in the test regions, AUROC undefined
                ax.text(
                    group_center[j],
                    0.05,
                    "not evaluable\n(no test-region detections)",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                    rotation=90,
                )

        ax.set_xticks(group_center)
        ax.set_xticklabels(analytes, fontsize=8)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7)
    else:
        # Fallback: show per-task AUROC across models
        if results:
            classification_results = [r for r in results if r.get("task") in ("T1", "T4", "T5")]
            if classification_results:
                rdf = pd.DataFrame(classification_results)
                rdf["display_name"] = rdf["model"].map(_pub_name)
                tasks = sorted(rdf["task"].unique())
                display_models = list(dict.fromkeys(rdf["display_name"]))
                x = np.arange(len(tasks))
                width = 0.8 / max(len(display_models), 1)
                for i, display_name in enumerate(display_models):
                    values = []
                    for task in tasks:
                        rows = rdf[(rdf["display_name"] == display_name) & (rdf["task"] == task)]
                        if not rows.empty:
                            values.append(rows.iloc[0]["metrics"].get("auroc", 0.0))
                        else:
                            values.append(0.0)
                    color = MODEL_COLORS.get(display_name, "#666666")
                    hatch = MODEL_HATCHES.get(display_name, "")
                    ax.bar(
                        x + i * width,
                        values,
                        width,
                        label=display_name,
                        color=color,
                        hatch=hatch,
                        edgecolor="black",
                        linewidth=0.3,
                    )
                ax.set_xticks(x + width * (len(display_models) - 1) / 2)
                ax.set_xticklabels(tasks)
                ax.set_ylabel("AUROC")
                ax.set_ylim(0, 1.0)
                ax.legend(fontsize=7)
            else:
                _placeholder(ax, "No data", figure="fig_ext2_per_analyte_roc")
        else:
            _placeholder(ax, "No data", figure="fig_ext2_per_analyte_roc")

    # Figure numbering lives in the manuscript captions, never baked into the image.
    ax.set_title("Per-Analyte AUROC (T3)", loc="left", fontsize=8)
    fig.tight_layout()

    path = output_dir / "fig_ext2_per_analyte_roc.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_ext_geographic_splits(
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
    prefix and is unaffected.

    Renderer pinned to matplotlib/cartopy (2026-07-12): every committed
    version of this figure was produced by this path (PDF Creator metadata),
    and the pygmt variant renders a visibly different layout, so the pygmt
    short-circuit used by the other geographic figures is deliberately absent
    here. ``generate_figures_geo.fig_ext_geographic_splits_pygmt`` remains
    available but is not the renderer of record.
    """
    _load_style()
    proj = _get_conus_projection()
    if proj is not None:
        import cartopy.crs as ccrs

        fig, ax = plt.subplots(1, 1, figsize=(7.087, 4.5), subplot_kw={"projection": proj})
        data_crs = ccrs.PlateCarree()
        ax.set_extent([-125, -66, 24, 50], crs=data_crs)  # type: ignore[attr-defined]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.087, 4.5))
        data_crs = None

    _add_conus_basemap(ax)

    split_config = {
        "train": {"regions": [1, 3, 4, 5, 6], "color": "#0072B2", "label": "Train"},
        "val": {"regions": [2, 7], "color": "#E69F00", "label": "Validation"},
        "test": {"regions": [8, 9, 10], "color": "#D55E00", "label": "Test"},
    }

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

            # Prefix-derived region (split truth), polygon fallback for non-state prefixes
            coords_reset = coords.reset_index()
            regions = figure_plot_regions(coords_reset)
            regions.index = coords_reset["pwsid"]

            for _split_name, cfg in split_config.items():
                mask = regions.isin(cfg["regions"])
                subset = coords[mask]
                if not subset.empty:
                    scatter_kw: dict[str, Any] = {}
                    if data_crs is not None:
                        scatter_kw["transform"] = data_crs
                    ax.scatter(
                        subset["longitude"],
                        subset["latitude"],
                        s=2,
                        alpha=0.4,
                        color=cfg["color"],
                        label=f"{cfg['label']} (R{','.join(str(r) for r in cfg['regions'])})",
                        zorder=2,
                        **scatter_kw,
                    )

            ax.legend(loc="lower left", markerscale=3, fontsize=7)
    else:
        _placeholder(ax, "No coordinate data", figure="fig_ext3_geographic_splits")

    # Figure numbering lives in the manuscript captions, never baked into the image.
    ax.set_title("Geographic Split (EPA Regions)", loc="left", fontsize=8)
    if data_crs is None:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-130, -65)
        ax.set_ylim(24, 50)
    fig.tight_layout()

    path = output_dir / "fig_ext3_geographic_splits.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_ext_roc_pr_curves(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 7: AUROC and AUPRC by model for T1 and T4.

    Per-model AUROC/AUPRC summary bars from the frozen benchmark metrics.
    The archive deliberately strips prediction arrays, so ROC/PR curve
    reconstruction is not possible from frozen inputs (Code Availability);
    the figure is therefore defined as the metric summary, and the
    historical ``fig_ext4_roc_pr_curves`` filename is kept to avoid
    churning committed figure references.
    """
    _load_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.087, 6.0))

    tasks_to_plot = ["T1", "T4"]

    for row_idx, task in enumerate(tasks_to_plot):
        task_results = [r for r in results if r.get("task") == task]
        if not task_results:
            for col_idx in range(2):
                _placeholder(
                    axes[row_idx, col_idx],
                    f"No {task} data",
                    figure="fig_ext4_roc_pr_curves",
                )
            continue

        for col_idx, metric_key in enumerate(["auroc", "auprc"]):
            ax = axes[row_idx, col_idx]
            display_names = []
            values = []
            colors = []
            for r in task_results:
                dn = _pub_name(r.get("model", ""))
                val = r.get("metrics", {}).get(metric_key, 0.0)
                if val == val:  # NaN guard
                    display_names.append(dn)
                    values.append(float(val))
                    colors.append(MODEL_COLORS.get(dn, "#666666"))

            if values:
                ax.barh(range(len(values)), values, color=colors)
                ax.set_yticks(range(len(values)))
                ax.set_yticklabels(display_names, fontsize=7)
                for i, v in enumerate(values):
                    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=6)

            metric_label = "AUROC" if metric_key == "auroc" else "AUPRC"
            ax.set_xlabel(metric_label)
            ax.set_title(f"{task} {metric_label}", loc="left", fontsize=8)
            ax.set_xlim(0, 1.0)

    # Figure numbering lives in the manuscript captions, never baked into the image.
    fig.suptitle(
        "AUROC and AUPRC by Model (T1 PFAS, T4 Heavy Metals)",
        fontsize=9,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = output_dir / "fig_ext4_roc_pr_curves.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def _plot_shap_panel(ax: Any, shap_data: dict[str, Any], color: str, top_n: int = 15) -> None:
    """Horizontal mean-|SHAP| bars for one model variant, descending."""
    mean_abs = shap_data.get("mean_abs_shap", {})
    top = sorted(mean_abs.items(), key=lambda kv: -kv[1])[:top_n]
    labels = [_readable_feature_name(name) for name, _ in top]
    values = [v for _, v in top]
    y_pos = range(len(top))
    ax.barh(y_pos, values, color=color)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")


def fig_shap_summary(
    shap_data: dict[str, Any] | None,
    shap_pf_data: dict[str, Any] | None,
    output_dir: Path,
) -> Path:
    """Figure 3: two-panel SHAP comparison for T1 XGBoost.

    Panel (a) is the full model (``shap_T1.json``), where data-source
    provenance proxies dominate -- the evidence that models learn the
    monitoring process. Panel (b) is the environment-only model
    (``shap_T1_provenance_free.json``), showing the genuine environmental
    drivers. Falls back to a single panel when the provenance-free SHAP
    export is unavailable.
    """
    _load_style()
    two_panel = bool(shap_data) and bool(shap_pf_data)
    n_cols = 2 if two_panel else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7.087, 4.5), squeeze=False)

    ax_a = axes[0][0]
    if shap_data and shap_data.get("mean_abs_shap"):
        _plot_shap_panel(ax_a, shap_data, color="#0072B2")
        title_a = "a  With provenance features" if two_panel else "Feature importance (SHAP)"
        ax_a.set_title(title_a, loc="left")
    else:
        _placeholder(ax_a, "SHAP data not available", figure="fig3_shap_summary")
        ax_a.set_title("Feature Importance (SHAP)", loc="left")

    if two_panel:
        ax_b = axes[0][1]
        _plot_shap_panel(ax_b, shap_pf_data, color="#009E73")  # type: ignore[arg-type]
        ax_b.set_title("b  Environment-only (provenance-free)", loc="left")

    fig.tight_layout()

    path = output_dir / "fig3_shap_summary.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_confound_decomposition(
    split_data: dict[str, Any] | None,
    detection_ablation: list[dict[str, Any]] | None,
    feature_ablation: list[dict[str, Any]] | None,
    output_dir: Path,
) -> Path:
    """Main figure: the two compounding confounds in reported model skill.

    Panel (a): spatial leakage -- random-split vs geographic-split AUPRC on the
    baseline feature set (``split_comparison.json``). Panel (b): ascertainment
    -- T1 AUPRC with vs without detection-only sources
    (``detection_only_ablation.json``) and T4 AUPRC with vs without monitoring
    intensity features (``feature_ablation.json``).
    """
    _load_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.087, 3.5))

    # Panel (a): spatial leakage.
    ax = axes[0]
    simple = (split_data or {}).get("simple") or {}
    if "random_split_metrics" in simple:
        rand_auprc = simple["random_split_metrics"].get("auprc", float("nan"))
        geo_auprc = simple["geographic_split_metrics"].get("auprc", float("nan"))
        inflation = simple.get("inflation_ratio", {}).get("auprc", float("nan"))
        bars = ax.bar(
            ["Random split", "Geographic split"],
            [rand_auprc, geo_auprc],
            color=["#D55E00", "#0072B2"],
            width=0.5,
        )
        for bar, val in zip(bars, [rand_auprc, geo_auprc], strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.3f}",
                ha="center",
                fontsize=7,
            )
        ax.text(
            0.5,
            0.92,
            f"{inflation:.2f}× inflation",  # noqa: RUF001
            ha="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_ylabel("AUPRC")
        ax.set_ylim(0, max(rand_auprc, geo_auprc) * 1.25)
    else:
        _placeholder(ax, "No data", figure="fig2_confound_decomposition")
    ax.set_title("a  Spatial leakage (evaluation protocol)", loc="left")

    # Panel (b): ascertainment / monitoring confounding.
    ax = axes[1]
    pairs: list[tuple[str, float, float]] = []
    for entry in detection_ablation or []:
        if entry.get("task") == "T1" and entry.get("model") == "xgboost_classifier":
            pairs.append(
                (
                    "T1: detection-only\nsources removed",
                    entry.get("all_sources", {}).get("auprc", float("nan")),
                    entry.get("excluding_detection_only", {}).get("auprc", float("nan")),
                )
            )
            break
    for entry in feature_ablation or []:
        if entry.get("task") == "T4" and entry.get("category") == "monitoring_intensity":
            pairs.append(
                (
                    "T4: monitoring\nfeatures ablated",
                    entry.get("baseline_score", float("nan")),
                    entry.get("ablated_score", float("nan")),
                )
            )
            break
    if pairs:
        x = np.arange(len(pairs))
        width = 0.35
        before = [p[1] for p in pairs]
        after = [p[2] for p in pairs]
        ax.bar(x - width / 2, before, width, color="#D55E00", label="With")
        ax.bar(x + width / 2, after, width, color="#0072B2", label="Without")
        for xi, (b, a) in enumerate(zip(before, after, strict=False)):
            ax.text(xi - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=7)
            ax.text(xi + width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in pairs], fontsize=7)
        ax.set_ylabel("AUPRC")
        ax.set_ylim(0, max(before) * 1.25)
        ax.legend(fontsize=6, loc="upper right")
    else:
        _placeholder(ax, "No data", figure="fig2_confound_decomposition")
    ax.set_title("b  Ascertainment (who is monitored)", loc="left")

    fig.tight_layout()
    path = output_dir / "fig2_confound_decomposition.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


#: Display names for the monitoring-inequity demographic dimensions.
_INEQUITY_GROUP_LABELS = {
    "pct_people_of_color": "% People of color",
    "pct_low_income": "% Low income",
    "pct_limited_english": "% Limited English",
    "pct_less_hs_education": "% Less than HS education",
}


def fig_monitoring_inequity(
    inequity_data: dict[str, Any] | None,
    output_dir: Path,
) -> Path:
    """Main figure: monitoring effort is inequitably distributed.

    Consumes ``monitoring_inequity.json``. Panel (a): mean samples per system
    in high- vs low-share systems per demographic dimension, annotated with the
    monitoring-intensity ratio (an infinite ratio -- no low-share systems --
    is annotated rather than plotted). Panel (b): population-size-adjusted
    regression coefficients with 95% CIs.
    """
    import math

    _load_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.087, 3.5))

    groups = [g for g in _INEQUITY_GROUP_LABELS if g in (inequity_data or {})]

    ax = axes[0]
    if groups:
        x = np.arange(len(groups))
        width = 0.35
        high = [inequity_data[g].get("mean_high", float("nan")) for g in groups]  # type: ignore[index]
        low = [inequity_data[g].get("mean_low", float("nan")) for g in groups]  # type: ignore[index]
        ax.bar(x - width / 2, high, width, color="#D55E00", label="High share")
        ax.bar(x + width / 2, low, width, color="#0072B2", label="Low share")
        for xi, g in enumerate(groups):
            ratio = inequity_data[g].get("monitoring_ratio", float("nan"))  # type: ignore[index]
            note = "no low-share\nsystems" if math.isinf(ratio) else f"{ratio:.2f}×"  # noqa: RUF001
            ax.text(xi, max(high[xi], low[xi]) + 0.4, note, ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([_INEQUITY_GROUP_LABELS[g] for g in groups], fontsize=6, rotation=20)
        ax.set_ylabel("Mean samples per system")
        ax.legend(fontsize=6, loc="upper left")
        # Headroom so the ratio / "no low-share systems" notes clear the title.
        _top = max([v for v in high + low if v == v] or [1.0])
        ax.set_ylim(top=_top * 1.35)
    else:
        _placeholder(ax, "No data", figure="fig4_monitoring_inequity")
    ax.set_title("a  Monitoring intensity by group", loc="left")

    ax = axes[1]
    if groups:
        coefs = [inequity_data[g].get("size_adjusted_coef", float("nan")) for g in groups]  # type: ignore[index]
        ses = [inequity_data[g].get("size_adjusted_se", float("nan")) for g in groups]  # type: ignore[index]
        x = np.arange(len(groups))
        ax.errorbar(
            x,
            coefs,
            yerr=[1.96 * s for s in ses],
            fmt="o",
            color="#0072B2",
            capsize=3,
            markersize=4,
        )
        ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
        for xi, g in enumerate(groups):
            p = inequity_data[g].get("size_adjusted_p", float("nan"))  # type: ignore[index]
            p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            # Place the label clear of the upper error-bar cap so the decimal
            # point is not obscured.
            cap_top = coefs[xi] + 1.96 * ses[xi]
            ax.annotate(
                p_str,
                (xi, cap_top),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.margins(y=0.18)
        ax.set_xticks(x)
        ax.set_xticklabels([_INEQUITY_GROUP_LABELS[g] for g in groups], fontsize=6, rotation=20)
        ax.set_ylabel("Size-adjusted coefficient\n(log samples per 1-SD)")
    else:
        _placeholder(ax, "No data", figure="fig4_monitoring_inequity")
    ax.set_title("b  Population-size-adjusted association", loc="left")

    fig.tight_layout()
    path = output_dir / "fig4_monitoring_inequity.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_reliability_diagram(
    calibration_data: list[dict[str, Any]] | None,
    output_dir: Path,
) -> Path:
    """Reliability diagram — uncalibrated calibration curves for T1 and T4.

    Consumes ``calibration_analysis.json`` from the calibration pipeline stage.
    """
    _load_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.087, 3.5))

    if not calibration_data:
        # Without data the loop below silently renders diagonal-only panels.
        _placeholder(axes[0], "No calibration data", figure="fig_reliability_diagram")

    for ax_idx, task in enumerate(["T1", "T4"]):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5, label="Perfect")

        if calibration_data:
            task_entries = [e for e in calibration_data if e.get("task") == task]
            for entry in task_entries:
                if entry.get("model", "") == "dummy_classifier":
                    continue  # constant predictor collapses to one point; clutters legend
                model_name = _pub_name(entry.get("model", ""))
                curve = entry.get("calibration_curve", {})
                frac_pos = curve.get("fraction_positive", [])
                mean_pred = curve.get("mean_predicted", [])
                ece = entry.get("ece", float("nan"))
                if frac_pos and mean_pred:
                    color = MODEL_COLORS.get(model_name, "#666666")
                    ax.plot(
                        mean_pred,
                        frac_pos,
                        "o-",
                        color=color,
                        markersize=3,
                        linewidth=1.0,
                        label=f"{model_name} (ECE={ece:.3f})",
                    )

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{'a' if ax_idx == 0 else 'b'}  {task} Calibration", loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=5, loc="upper left")

    fig.tight_layout()
    path = output_dir / "fig_reliability_diagram.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_split_comparison(
    split_data: dict[str, Any] | None,
    output_dir: Path,
) -> Path:
    """Supplementary Fig. 6: split strategy comparison — AUPRC bars."""
    _load_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.087, 3.5))

    # split_comparison.json nests the headline pair under "simple".
    if split_data and "simple" in split_data:
        split_data = split_data["simple"]
    if split_data and "random_split_metrics" in split_data:
        categories = ["Baseline Features\n(Random Split)", "Baseline Features\n(Geo Split)"]
        rand_auprc = split_data["random_split_metrics"].get("auprc", 0)
        geo_auprc = split_data["geographic_split_metrics"].get("auprc", 0)
        values = [rand_auprc, geo_auprc]
        colors = ["#D55E00", "#0072B2"]
        bars = ax.bar(categories, values, color=colors, width=0.5)
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        inflation = split_data.get("inflation_ratio", {}).get("auprc")
        if inflation:
            ax.text(
                0.95,
                0.95,
                f"Inflation ratio: {inflation:.2f}x",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
    else:
        _placeholder(ax, "Split comparison data not available", figure="fig_ext_split_comparison")

    ax.set_ylabel("AUPRC")
    ax.set_title("Evaluation Protocol Comparison", loc="left")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    path = output_dir / "fig_ext_split_comparison.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_causal_comparison(
    causal_data: list[dict[str, Any]] | None,
    output_dir: Path,
) -> Path:
    """Extended Data Fig. 2: SHAP importance vs DML causal effect comparison.

    The historical ``fig_ext8`` filename is kept to avoid churning committed
    figure references.

    Parameters
    ----------
    causal_data : list[dict] | None
        Loaded from ``causal_deconfounding.json`` (adjusted association data).
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    _load_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.087, 5.0))

    if causal_data:
        # Filter to features with meaningful sample size and sort by |causal_effect|
        sorted_data = sorted(
            causal_data, key=lambda x: abs(x.get("causal_effect") or 0), reverse=True
        )
        top = sorted_data[:12]
        top = list(reversed(top))  # bottom-to-top for horizontal bars

        features = [d["feature"] for d in top]
        shap_vals = [d.get("shap_importance") or 0 for d in top]
        causal_vals = [d.get("causal_effect") or 0 for d in top]

        y = np.arange(len(features))
        height = 0.35

        ax.barh(y - height / 2, shap_vals, height, label="SHAP Importance", color="#0072B2")
        ax.barh(y + height / 2, causal_vals, height, label="DML Causal Effect", color="#D55E00")

        ax.set_yticks(y)
        ax.set_yticklabels(features, fontsize=6)
        ax.set_xlabel("Effect Size")
        ax.legend(fontsize=7, loc="lower right")
        # The aggregate Spearman rank correlation is stated (pn-gated) in the
        # Extended Data Fig. 2 caption; never bake a copy into the image.
    else:
        _placeholder(
            ax,
            "Causal adjusted association data not available",
            figure="fig_ext8_causal_comparison",
        )

    # Figure numbering lives in the manuscript captions, never baked into the image.
    ax.set_title(
        "SHAP vs Causal Effect Comparison",
        loc="left",
        fontsize=8,
    )
    fig.tight_layout()

    path = output_dir / "fig_ext8_causal_comparison.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_group_conformal(
    conformal_data: list[dict[str, Any]] | None,
    output_dir: Path,
) -> Path:
    """Extended Data Fig. 3: group conformal prediction per-region coverage.

    The historical ``fig_ext9`` filename is kept to avoid churning committed
    figure references.

    Parameters
    ----------
    conformal_data : list[dict] | None
        Loaded from ``group_conformal_results.json``.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    _load_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.087, 3.5))

    if conformal_data:
        # Use alpha=0.05 entry
        entry = next((e for e in conformal_data if e.get("alpha") == 0.05), conformal_data[0])
        per_region = entry.get("per_region", {})
        target = entry.get("target_coverage", 0.95)

        regions = sorted(per_region.keys(), key=int)
        coverages = [per_region[r]["coverage"] for r in regions]
        labels = [f"Region {r}\n(n={per_region[r]['n']})" for r in regions]

        colors = ["#0072B2" if c >= target else "#D55E00" for c in coverages]
        ax.bar(labels, coverages, color=colors, width=0.6, edgecolor="black", linewidth=0.3)
        ax.axhline(
            y=target, color="black", linestyle="--", linewidth=1.0, label=f"Target ({target})"
        )

        # Add overall coverage
        overall = entry.get("coverage", 0)
        ax.axhline(
            y=overall,
            color="#009E73",
            linestyle=":",
            linewidth=1.0,
            label=f"Overall ({overall:.3f})",
        )

        ax.set_ylabel("Coverage")
        ax.set_ylim(0.85, 1.0)
        ax.legend(fontsize=7, loc="lower left")
    else:
        _placeholder(ax, "Group conformal data not available", figure="fig_ext9_group_conformal")

    # Figure numbering lives in the manuscript captions, never baked into the image.
    ax.set_title(
        "Group Conformal Per-Region Coverage (alpha=0.05)",
        loc="left",
        fontsize=8,
    )
    fig.tight_layout()

    path = output_dir / "fig_ext9_group_conformal.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def fig_icp_analysis(
    icp_data: dict[str, Any] | None,
    output_dir: Path,
) -> Path:
    """Extended Data Fig. 4: ICP training dynamics + per-task metrics.

    The historical ``fig_ext10`` filename is kept to avoid churning committed
    figure references. Panel (a) draws the frozen ``history`` block (seeded
    CPU canonical-config re-fit; see ``icp_diagnostics.json`` ``history_meta``);
    panel (b) draws the frozen benchmark metrics.

    Parameters
    ----------
    icp_data : dict | None
        ICP diagnostics loaded from ``icp_diagnostics.json``.
    output_dir : Path
        Directory to save figure.

    Returns
    -------
    Path
        Path to saved figure.
    """
    _load_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.087, 3.5))

    # Panel a: Adversary convergence
    ax_a = axes[0]
    if icp_data and "history" in icp_data:
        # Use first available task history (prefer T1)
        history_key = "T1" if "T1" in icp_data["history"] else next(iter(icp_data["history"]))
        history = icp_data["history"][history_key]
        epochs = [h["epoch"] for h in history]
        task_losses = [h["task_loss"] for h in history]
        adv_losses = [h["adv_loss"] for h in history]
        lambda_advs = [h["lambda_adv"] for h in history]

        ax_a.plot(epochs, task_losses, color="#0072B2", label="Task loss", linewidth=1.2)
        ax_a.plot(epochs, adv_losses, color="#D55E00", label="Adversary loss", linewidth=1.2)
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Loss")
        ax_a.legend(loc="upper right", fontsize=7)

        # Right y-axis for lambda schedule
        ax_r = ax_a.twinx()
        ax_r.plot(
            epochs,
            lambda_advs,
            color="#009E73",
            linestyle="--",
            alpha=0.7,
            label=r"$\lambda_{adv}$",
            linewidth=1.0,
        )
        ax_r.set_ylabel(r"$\lambda_{adv}$", color="#009E73")
        ax_r.tick_params(axis="y", labelcolor="#009E73")

        # Mark warmup boundary
        warmup_end = next((e for e, lv in zip(epochs, lambda_advs, strict=True) if lv > 0), None)
        if warmup_end is not None:
            ax_a.axvline(warmup_end, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
            ax_a.text(
                warmup_end + 1,
                ax_a.get_ylim()[1] * 0.95,
                "warmup end",
                fontsize=6,
                color="gray",
                va="top",
            )
    else:
        _placeholder(ax_a, "No ICP training history available", figure="fig_ext10_icp_analysis")
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=10)

    # Panel b: Feature importance comparison (ICP permutation importance)
    ax_b = axes[1]
    if icp_data and "results" in icp_data:
        # Show ICP metrics as a summary bar chart
        icp_metrics = []
        for r in icp_data["results"]:
            task = r.get("task", "")
            metrics = r.get("metrics", {})
            auroc = metrics.get("auroc")
            auprc = metrics.get("auprc")
            if auroc is not None:
                icp_metrics.append({"task": task, "metric": "AUROC", "value": auroc})
            if auprc is not None:
                icp_metrics.append({"task": task, "metric": "AUPRC", "value": auprc})

        if icp_metrics:
            df_m = pd.DataFrame(icp_metrics)
            tasks = df_m["task"].unique()
            x = np.arange(len(tasks))
            width = 0.35

            auroc_vals = [
                df_m[(df_m["task"] == t) & (df_m["metric"] == "AUROC")]["value"].to_numpy()
                for t in tasks
            ]
            auprc_vals = [
                df_m[(df_m["task"] == t) & (df_m["metric"] == "AUPRC")]["value"].to_numpy()
                for t in tasks
            ]

            auroc_vals = [v[0] if len(v) > 0 else 0 for v in auroc_vals]
            auprc_vals = [v[0] if len(v) > 0 else 0 for v in auprc_vals]

            ax_b.bar(x - width / 2, auroc_vals, width, label="AUROC", color="#0072B2")
            ax_b.bar(x + width / 2, auprc_vals, width, label="AUPRC", color="#D55E00")
            ax_b.set_xticks(x)
            ax_b.set_xticklabels(tasks, fontsize=8)
            ax_b.set_ylabel("Score")
            ax_b.set_ylim(0, 1)
            ax_b.legend(fontsize=7)
        else:
            _placeholder(
                ax_b, "No ICP classification metrics available", figure="fig_ext10_icp_analysis"
            )
    else:
        _placeholder(ax_b, "No ICP data available", figure="fig_ext10_icp_analysis")
    ax_b.set_title("b", loc="left", fontweight="bold", fontsize=10)

    fig.tight_layout()
    path = output_dir / "fig_ext10_icp_analysis.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


ALL_FIGURE_FUNCTIONS = [
    fig_dataset_overview,
    fig_data_distribution,
    fig_benchmark_results,
    fig_shap_summary,
    fig_confound_decomposition,
    fig_monitoring_inequity,
    fig_national_risk_map,
    fig_equity_analysis,
    fig_transfer_learning,
    fig_temporal_prediction,
    fig_ext_correlation_matrix,
    fig_ext_per_analyte_roc,
    fig_ext_geographic_splits,
    fig_ext_roc_pr_curves,
    fig_reliability_diagram,
    fig_split_comparison,
    fig_causal_comparison,
    fig_group_conformal,
    fig_icp_analysis,
]


def _load_pipeline_data(
    results_dir: Path,
    data_dir: Path,
    pf_dir: Path | None = None,
) -> dict[str, Any]:
    """Load pipeline outputs for figure generation.

    Parameters
    ----------
    results_dir : Path
        Results directory from reproduce.py.
    data_dir : Path
        Root data directory.
    pf_dir : Path or None
        Environment-only (``--provenance-free``) run directory; fallback
        location for the provenance-free SHAP export when
        ``shap_T1_provenance_free.json`` is absent from ``results_dir``.

    Returns
    -------
    dict[str, Any]
        Keys: ``results``, ``wq_df``, ``importance``, ``predictions``,
        ``equity``, ``t5_results``, ``t7_results``, ``shap_data``,
        ``shap_pf_data``, ``inequity_data``, ``detection_ablation``,
        ``feature_ablation``, ...
    """
    import json

    out: dict[str, Any] = {}

    # Benchmark results
    results_json = results_dir / "results.json"
    results_data: list[dict[str, Any]] = []
    if results_json.exists():
        results_data = json.loads(results_json.read_text())
    out["results"] = results_data

    # Water quality data
    merged_path = data_dir / "interim" / "merged_wq.parquet"
    if merged_path.exists():
        out["wq_df"] = pd.read_parquet(merged_path)
        logger.info("Loaded WQ data: %d rows", len(out["wq_df"]))
    else:
        # Synthetic fallback
        logger.warning(
            "FIGURES GENERATED FROM SYNTHETIC DATA — not for publication. "
            "Run the full pipeline (scripts/reproduce.py --all) to generate "
            "figures from real data."
        )
        rng = np.random.RandomState(42)
        out["wq_df"] = pd.DataFrame(
            {
                "pwsid": [f"SYS{i:05d}" for i in range(100)],
                "analyte": rng.choice(["PFOS", "PFOA", "PFHxS"], 100),
                "concentration": rng.exponential(10.0, 100),
                "censored": rng.random(100) < 0.7,
                "latitude": rng.uniform(25, 48, 100),
                "longitude": rng.uniform(-125, -67, 100),
            }
        )
        out["_synthetic_fallback"] = True
        logger.info("Using synthetic WQ data (merged_wq.parquet not found)")

    # Feature importance
    importance_path = results_dir / "feature_importance.json"
    if importance_path.exists():
        imp_data = json.loads(importance_path.read_text())
        out["importance"] = pd.Series(imp_data, name="importance").sort_values(ascending=False)
        logger.info("Loaded feature importance: %d features", len(out["importance"]))
    else:
        out["importance"] = None

    # Predictions for the national risk map. Canonical source is the frozen
    # slim T1-PFOS surface (predictions_slim.json); the parquet fallback
    # (live results/predictions/) concatenates every task, so it is filtered
    # to the same T1/PFOS slice to keep the "PFAS risk" title truthful.
    slim_path = results_dir / "predictions_slim.json"
    predictions_dir = results_dir / "predictions"
    if slim_path.exists():
        slim = json.loads(slim_path.read_text())
        out["predictions"] = pd.DataFrame(slim["rows"])
        slim_meta = slim.get("_meta", {})
        logger.info(
            "Loaded frozen slim predictions: %d rows (task=%s, analyte=%s)",
            len(out["predictions"]),
            slim_meta.get("task"),
            slim_meta.get("analyte"),
        )
    elif predictions_dir.exists():
        pred_files = list(predictions_dir.rglob("*.parquet"))
        if pred_files:
            dfs = [pd.read_parquet(p) for p in pred_files]
            preds = pd.concat(dfs, ignore_index=True)
            if "task" in preds.columns:
                preds = preds[preds["task"] == "T1"]
                if "analyte" in preds.columns and (preds["analyte"] == "PFOS").any():
                    preds = preds[preds["analyte"] == "PFOS"]
            out["predictions"] = preds.reset_index(drop=True)
            logger.info("Loaded predictions (T1 slice): %d rows", len(out["predictions"]))
        else:
            out["predictions"] = None
    else:
        out["predictions"] = None

    # Equity analysis
    equity_path = results_dir / "equity_analysis.json"
    if equity_path.exists():
        eq_data = json.loads(equity_path.read_text())
        if isinstance(eq_data, list):
            out["equity"] = pd.DataFrame(eq_data)
        else:
            out["equity"] = pd.DataFrame([eq_data])
        logger.info("Loaded equity analysis")
    else:
        out["equity"] = None

    # Bootstrap CIs per burden ratio for the equity figure's error bars.
    burden_ci_path = results_dir / "group_burden_ci.json"
    if burden_ci_path.exists():
        out["burden_ci"] = json.loads(burden_ci_path.read_text()).get("burden_cis", {})
    else:
        out["burden_ci"] = None

    # T5 / T7 filtered results
    out["t5_results"] = [r for r in results_data if r.get("task") == "T5"] or None
    out["t7_results"] = [r for r in results_data if r.get("task") == "T7"] or None

    # Calibration analysis
    calibration_path = results_dir / "calibration_analysis.json"
    if calibration_path.exists():
        out["calibration_data"] = json.loads(calibration_path.read_text())
        logger.info("Loaded calibration analysis: %d entries", len(out["calibration_data"]))
    else:
        out["calibration_data"] = None

    # Split strategy comparison
    split_comp_path = results_dir / "split_comparison.json"
    if split_comp_path.exists():
        out["split_data"] = json.loads(split_comp_path.read_text())
        logger.info("Loaded split comparison data")
    else:
        out["split_data"] = None

    # Causal adjusted association
    causal_path = results_dir / "causal_deconfounding.json"
    if causal_path.exists():
        out["causal_data"] = json.loads(causal_path.read_text())
        logger.info("Loaded causal adjusted associations: %d features", len(out["causal_data"]))
    else:
        out["causal_data"] = None

    # Group conformal prediction
    conformal_path = results_dir / "group_conformal_results.json"
    if conformal_path.exists():
        out["conformal_data"] = json.loads(conformal_path.read_text())
        logger.info("Loaded group conformal results: %d entries", len(out["conformal_data"]))
    else:
        out["conformal_data"] = None

    # ICP diagnostics
    icp_path = results_dir / "icp_diagnostics.json"
    if icp_path.exists():
        out["icp_data"] = json.loads(icp_path.read_text())
        logger.info("Loaded ICP diagnostics")
    else:
        out["icp_data"] = None

    # SHAP exports: full model + environment-only (provenance-free) variant
    shap_path = results_dir / "shap_T1.json"
    out["shap_data"] = json.loads(shap_path.read_text()) if shap_path.exists() else None
    out["shap_pf_data"] = None
    pf_candidates = [results_dir / "shap_T1_provenance_free.json"]
    if pf_dir is not None:
        pf_candidates.append(pf_dir / "shap_T1.json")
    for cand in pf_candidates:
        if cand.exists():
            out["shap_pf_data"] = json.loads(cand.read_text())
            logger.info("Loaded provenance-free SHAP: %s", cand)
            break
    if out["shap_pf_data"] is None:
        logger.warning("Provenance-free SHAP not found — Fig 3 will be single-panel")

    # Monitoring inequity (allocation analysis)
    inequity_path = results_dir / "monitoring_inequity.json"
    out["inequity_data"] = (
        json.loads(inequity_path.read_text()) if inequity_path.exists() else None
    )

    # Two-confound decomposition inputs
    det_path = results_dir / "detection_only_ablation.json"
    out["detection_ablation"] = json.loads(det_path.read_text()) if det_path.exists() else None
    abl_path = results_dir / "feature_ablation.json"
    out["feature_ablation"] = json.loads(abl_path.read_text()) if abl_path.exists() else None

    # Feature matrix for correlation plot (Extended Data Fig 1)
    features_dir = data_dir / "interim" / "features"
    if features_dir.exists():
        feat_dfs = []
        for p in sorted(features_dir.glob("*.parquet")):
            feat_dfs.append(pd.read_parquet(p))
        if feat_dfs:
            out["feature_df"] = pd.concat(feat_dfs, axis=1)
            logger.info("Loaded feature data: %d columns", out["feature_df"].shape[1])
        else:
            out["feature_df"] = None
    else:
        out["feature_df"] = None

    return out


@click.command()
@click.option("--results", default="results", help="Results directory from reproduce.py.")
@click.option("--output", "-o", default="paper/figures", help="Output directory.")
@click.option("--data-dir", default="data", help="Root data directory.")
@click.option(
    "--provenance-free-results",
    default="results_provenance_free",
    help=(
        "Environment-only (--provenance-free) run directory; fallback source for "
        "the Fig 3 provenance-free SHAP panel."
    ),
)
@click.option(
    "--allow-placeholders",
    is_flag=True,
    default=False,
    help=(
        "Render 'No data' placeholder panels instead of raising "
        "MissingFigureDataError. Never use for canonical (frozen) regeneration."
    ),
)
def main(
    results: str,
    output: str,
    data_dir: str,
    provenance_free_results: str,
    allow_placeholders: bool,
) -> None:
    """Generate all paper figures."""
    global ALLOW_PLACEHOLDERS
    ALLOW_PLACEHOLDERS = allow_placeholders
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_pipeline_data(Path(results), Path(data_dir), pf_dir=Path(provenance_free_results))

    click.echo(f"Generating {len(ALL_FIGURE_FUNCTIONS)} figures to {output_dir}")

    paths = []
    paths.append(fig_dataset_overview(data["wq_df"], output_dir))
    paths.append(fig_data_distribution(data["wq_df"], output_dir))
    paths.append(fig_benchmark_results(data["results"], output_dir))
    paths.append(fig_feature_importance(data["importance"], output_dir))
    paths.append(fig_shap_summary(data.get("shap_data"), data.get("shap_pf_data"), output_dir))
    paths.append(
        fig_confound_decomposition(
            data.get("split_data"),
            data.get("detection_ablation"),
            data.get("feature_ablation"),
            output_dir,
        )
    )
    paths.append(fig_monitoring_inequity(data.get("inequity_data"), output_dir))
    paths.append(fig_national_risk_map(data["predictions"], output_dir))
    paths.append(fig_equity_analysis(data["equity"], output_dir, burden_ci=data.get("burden_ci")))
    paths.append(fig_transfer_learning(data["t5_results"], output_dir))
    paths.append(fig_temporal_prediction(data["t7_results"], output_dir))
    paths.append(fig_ext_correlation_matrix(data.get("feature_df"), output_dir))
    paths.append(fig_ext_per_analyte_roc(data["results"], output_dir))
    paths.append(fig_ext_geographic_splits(data["wq_df"], output_dir))
    paths.append(fig_ext_roc_pr_curves(data["results"], output_dir))
    paths.append(fig_reliability_diagram(data.get("calibration_data"), output_dir))
    paths.append(fig_split_comparison(data.get("split_data"), output_dir))
    paths.append(fig_causal_comparison(data.get("causal_data"), output_dir))
    paths.append(fig_group_conformal(data.get("conformal_data"), output_dir))
    paths.append(fig_icp_analysis(data.get("icp_data"), output_dir))

    for p in paths:
        click.echo(f"  {p}")

    click.echo("Done!")


if __name__ == "__main__":
    main()
