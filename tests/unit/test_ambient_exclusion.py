"""Tests for ambient-source exclusion from the main benchmark splits."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from aquacontam._constants import AMBIENT_WQ_SOURCES
from aquacontam.pipeline.training import _exclude_ambient_sources


def _wq(sources: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pwsid": [f"CA010100{i}" for i in range(len(sources))],
            "analyte": ["PFOS"] * len(sources),
            "concentration": [1.0] * len(sources),
            "source": sources,
        }
    )


def test_ambient_sources_constant() -> None:
    assert frozenset({"ca_geotracker", "wqp"}) == AMBIENT_WQ_SOURCES


def test_excludes_ambient_keeps_pws() -> None:
    df = _wq(["ucmr5", "sdwis", "nj_dep", "ca_geotracker", "wqp", "mo_dnr"])
    out = _exclude_ambient_sources(df)
    assert set(out["source"]) == {"ucmr5", "sdwis", "nj_dep", "mo_dnr"}
    assert "ca_geotracker" not in set(out["source"])
    assert "wqp" not in set(out["source"])
    assert len(out) == 4


def test_logs_what_was_dropped(caplog: pytest.LogCaptureFixture) -> None:
    df = _wq(["ucmr5", "ca_geotracker", "wqp"])
    with caplog.at_level(logging.INFO, logger="aquacontam.pipeline.training"):
        _exclude_ambient_sources(df)
    assert any("ambient-source" in r.getMessage() for r in caplog.records)


def test_noop_when_no_source_column() -> None:
    df = pd.DataFrame({"pwsid": ["CA0101001"], "analyte": ["PFOS"]})
    out = _exclude_ambient_sources(df)
    assert out.equals(df)


def test_noop_when_no_ambient_sources() -> None:
    df = _wq(["ucmr5", "sdwis", "nj_dep"])
    out = _exclude_ambient_sources(df)
    assert len(out) == 3
    assert set(out["source"]) == {"ucmr5", "sdwis", "nj_dep"}
