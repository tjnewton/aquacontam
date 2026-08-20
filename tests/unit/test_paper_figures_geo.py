"""Tests for PyGMT geographic paper figures (split from test_paper_figures.py).

PyGMT needs the GMT C library, so this module guards on pygmt before anything
else; the matplotlib guard protects the ``paper.generate_figures`` import that
``generate_figures_geo`` and the shared fixtures rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pygmt = pytest.importorskip("pygmt")
pytest.importorskip("matplotlib")

import paper.generate_figures as gf  # noqa: E402
from paper.generate_figures import MissingFigureDataError  # noqa: E402
from paper.generate_figures_geo import (  # noqa: E402
    fig_dataset_overview_pygmt,
    fig_ext_geographic_splits_pygmt,
    fig_national_risk_map_pygmt,
)


@pytest.fixture()
def allow_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt in to placeholder panels (canonical runs loud-fail instead)."""
    monkeypatch.setattr(gf, "ALLOW_PLACEHOLDERS", True)


@pytest.fixture()
def synthetic_wq_for_fig() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "pwsid": [f"SYS{i:05d}" for i in range(50)],
            "analyte": rng.choice(["PFOS", "PFOA"], 50),
            "concentration": rng.exponential(10.0, 50),
            "censored": rng.random(50) < 0.7,
            "latitude": rng.uniform(25, 48, 50),
            "longitude": rng.uniform(-125, -67, 50),
        }
    )


@pytest.fixture()
def synthetic_wq_with_sources() -> pd.DataFrame:
    """Water quality DataFrame with source column for 11 data sources."""
    rng = np.random.RandomState(42)
    sources = [
        "ucmr5",
        "ucmr3",
        "sdwis",
        "wqp",
        "mi_mpart",
        "ca_geotracker",
        "mn_mdh",
        "oh_epa",
        "wa_doh",
        "mo_dnr",
        "nj_dep",
        "nc_deq",
    ]
    n = 120
    return pd.DataFrame(
        {
            "pwsid": [f"SYS{i:05d}" for i in range(n)],
            "source": [sources[i % len(sources)] for i in range(n)],
            "latitude": rng.uniform(25, 48, n),
            "longitude": rng.uniform(-125, -67, n),
        }
    )


class TestPyGMTFigures:
    """Test PyGMT geographic figure generation."""

    def test_dataset_overview(
        self, tmp_path: Path, synthetic_wq_with_sources: pd.DataFrame
    ) -> None:
        path = fig_dataset_overview_pygmt(synthetic_wq_with_sources, tmp_path)
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.with_suffix(".png").exists()

    def test_dataset_overview_no_source(
        self, tmp_path: Path, synthetic_wq_for_fig: pd.DataFrame
    ) -> None:
        """Falls back gracefully when source column is absent."""
        df = synthetic_wq_for_fig.drop(columns=["source"], errors="ignore")
        path = fig_dataset_overview_pygmt(df, tmp_path)
        assert path.exists()

    def test_national_risk_map_with_data(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "latitude": rng.uniform(25, 48, 200),
                "longitude": rng.uniform(-125, -67, 200),
                "risk_score": rng.random(200),
            }
        )
        path = fig_national_risk_map_pygmt(df, tmp_path)
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.with_suffix(".png").exists()

    def test_national_risk_map_none(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_national_risk_map_pygmt(None, tmp_path)
        assert path.exists()
        assert path.name == "fig_supp_national_risk_map.pdf"

    def test_national_risk_map_empty(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_national_risk_map_pygmt(pd.DataFrame(), tmp_path)
        assert path.exists()

    def test_national_risk_map_none_raises_by_default(self, tmp_path: Path) -> None:
        with pytest.raises(MissingFigureDataError):
            fig_national_risk_map_pygmt(None, tmp_path)

    def test_mn_mdh_display_label(self) -> None:
        """mn_mdh must render as 'MN MDH' in the Fig. 1 legend, not raw code."""
        from paper.generate_figures_geo import _SOURCE_LABELS

        assert _SOURCE_LABELS["mn_mdh"] == "MN MDH"

    def test_geographic_splits(
        self, tmp_path: Path, synthetic_wq_with_sources: pd.DataFrame
    ) -> None:
        path = fig_ext_geographic_splits_pygmt(synthetic_wq_with_sources, tmp_path)
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.with_suffix(".png").exists()
