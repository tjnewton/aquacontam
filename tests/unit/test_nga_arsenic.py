"""Tests for the USGS NGA arsenic loader (T6 data source)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aquacontam.data.nga_arsenic import ARSENIC_MCL_UGL, load_nga_arsenic, nga_metadata

# EPSG:5070 (CONUS Albers) coordinates that reproject inside the CONUS bbox.
_SITES = [
    # SITE_ID, DataSource, State, XCoord, YCoord, WATER_USE, Depth_Value, Aquifer
    ("S_DOM_1", "NWIS", "KS", "0", "1900000", "Domestic", "120", "High Plains aquifer"),
    ("S_DOM_2", "GWAM", "IL", "600000", "2100000", "Domestic", "80", "Glacial aquifer system"),
    ("S_PUB_1", "SDWIS", "TX", "-200000", "1500000", "Public supply", "300", "Edwards-Trinity"),
    ("S_PUB_2", "NWIS", "CO", "-700000", "1950000", "Public supply", "250", "Denver Basin"),
    ("S_IRR_1", "NWIS", "NE", "100000", "2200000", "Irrigation", "150", "High Plains aquifer"),
]
# SITE_ID, As_Remark, As_Value
_CHEM = [
    ("S_DOM_1", "", "15.0"),  # detected, exceeds MCL
    ("S_DOM_1", "", "8.0"),  # detected, below
    ("S_DOM_2", "<", "1.0"),  # censored non-detect
    ("S_PUB_1", "", "25.0"),  # detected, exceeds
    ("S_PUB_2", "<", "0.5"),  # censored
    ("S_IRR_1", "", "2.0"),  # detected, below
]


def _write_mock_nga(raw_dir: Path) -> None:
    dest = raw_dir / "nga_arsenic"
    dest.mkdir(parents=True, exist_ok=True)
    site = pd.DataFrame(
        [
            {
                "DataSource": ds,
                "State": st,
                "SITE_ID": sid,
                "XCoord": x,
                "YCoord": y,
                "WATER_USE": wu,
                "Depth_pcode": "72008",
                "Depth_Value": depth,
                "Open_Interval_Length": "",
                "Aquifer": aq,
                "Confidence": "High",
            }
            for sid, ds, st, x, y, wu, depth, aq in _SITES
        ]
    )
    site.to_csv(dest / "Site_Information.csv", index=False)
    chem = pd.DataFrame(
        [
            {
                "SITE_ID": sid,
                "Sample_Date": "2015-06-01",
                "Sample_Time": "",
                "SampleNumID": f"N{i}",
                "As_pcode": "01000",
                "As_Remark": rem,
                "As_Value": val,
                "Mn_pcode": "",
                "Mn_Remark": "",
                "Mn_Value": "",
                "pH_pcode": "",
                "pH_Remark": "",
                "pH_Value": "",
            }
            for i, (sid, rem, val) in enumerate(_CHEM)
        ]
    )
    chem.to_csv(dest / "Water_Chemistry.csv", index=False)


def test_load_schema(tmp_path: Path) -> None:
    _write_mock_nga(tmp_path)
    gdf = load_nga_arsenic(tmp_path)
    for col in (
        "pwsid",
        "analyte",
        "concentration",
        "censored",
        "detection_limit",
        "sample_date",
        "latitude",
        "longitude",
        "water_use",
        "well_depth_ft",
    ):
        assert col in gdf.columns
    assert (gdf["analyte"] == "arsenic").all()
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326


def test_water_use_filter(tmp_path: Path) -> None:
    _write_mock_nga(tmp_path)
    dom = load_nga_arsenic(tmp_path, water_use="Domestic")
    assert set(dom["water_use"].unique()) == {"Domestic"}
    assert set(dom["pwsid"].unique()) == {"S_DOM_1", "S_DOM_2"}
    pub = load_nga_arsenic(tmp_path, water_use="Public supply")
    assert set(pub["pwsid"].unique()) == {"S_PUB_1", "S_PUB_2"}


def test_censoring(tmp_path: Path) -> None:
    _write_mock_nga(tmp_path)
    gdf = load_nga_arsenic(tmp_path)
    censored = gdf[gdf["pwsid"] == "S_DOM_2"].iloc[0]
    assert bool(censored["censored"]) is True
    assert censored["concentration"] == 0.0
    assert censored["detection_limit"] == 1.0
    detected = gdf[(gdf["pwsid"] == "S_DOM_1") & (gdf["concentration"] == 15.0)].iloc[0]
    assert bool(detected["censored"]) is False


def test_reprojection_conus(tmp_path: Path) -> None:
    _write_mock_nga(tmp_path)
    gdf = load_nga_arsenic(tmp_path)
    assert gdf["latitude"].between(24.0, 50.0).all()
    assert gdf["longitude"].between(-125.0, -66.0).all()


def test_arsenic_exceedance_target(tmp_path: Path) -> None:
    """aggregate_to_system_level supports arsenic action_level (MCL 10 µg/L)."""
    from aquacontam.features.assembly import aggregate_to_system_level

    _write_mock_nga(tmp_path)
    dom = load_nga_arsenic(tmp_path, water_use="Domestic")
    agg = aggregate_to_system_level(dom, "arsenic", target="action_level")
    assert agg.loc["S_DOM_1", "target"] == 1  # max 15 >= 10
    assert agg.loc["S_DOM_2", "target"] == 0  # only a censored non-detect


def test_metadata() -> None:
    meta = nga_metadata()
    assert meta["doi"] == "10.5066/P9JMUAPY"
    assert meta["mcl_ugl"] == ARSENIC_MCL_UGL == 10.0
