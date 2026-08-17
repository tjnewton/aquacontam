"""Tests for the Minnesota MDH PFAS data source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquacontam._constants import PPT_TO_UGL
from aquacontam.data.mn_mdh import MnMdhSource

_COLS = [
    "PWS_ID",
    "PWS_TYPE",
    "SAMPLE_ID",
    "ANALYTE_GROUP",
    "ANALYTE",
    "RESULT",
    "UNIT_MEASURE_CODE",
    "RESULT_CODE",
    "REPORTING_LIMIT",
    "COLLECTION_DATE",
]


def _row(
    pws: int,
    sample: str,
    group: str,
    analyte: str,
    result: float,
    unit: str,
    rl: float,
    date: str = "2023-06-15",
    ptype: str = "Community",
) -> list:
    return [pws, ptype, sample, group, analyte, result, unit, np.nan, rl, date]


def _write_xlsx(path: Path, rows: list[list]) -> None:
    pd.DataFrame(rows, columns=_COLS).to_excel(path, index=False)


@pytest.fixture()
def mn_source(tmp_data_dirs: dict[str, Path]) -> MnMdhSource:
    return MnMdhSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_xlsx(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a representative MDH export covering every parse branch."""
    rows = [
        # PFOS detect, replicated across all three ANALYTE_GROUP panels -> 1 row
        _row(1010001, "S1", "PFAS_533", "Perfluorooctanesulfonate (PFOS)", 10.0, "ng/L", 2.0),
        _row(1010001, "S1", "PFC_EXPANDED", "Perfluorooctanesulfonate (PFOS)", 10.0, "ng/L", 2.0),
        _row(
            1010001, "S1", "PFAS_Regulated", "Perfluorooctanesulfonate (PFOS)", 10.0, "ng/L", 2.0
        ),
        # ug/L detect with a whitespace-padded unit (real data is padded)
        _row(1010001, "S1", "PFAS_533", "Perfluorooctanoic acid (PFOA)", 0.07, "ug/L  ", 0.05),
        # non-detect reported at the reporting limit (ng/L)
        _row(1010002, "S2", "PFAS_533", "Perfluorobutanesulfonate (PFBS)", 1.8, "ng/L", 1.8),
        # below-RL estimate (ng/L) -> censored
        _row(1010002, "S2", "PFAS_533", "Perfluorohexanesulfonate (PFHxS)", 1.0, "ng/L", 1.8),
        # FOSA -> the one MN-only canonical name
        _row(
            1010002, "S2", "PFC_EXPANDED", "Perfluorooctane sulfonamide (FOSA)", 5.0, "ng/L", 2.0
        ),
        # renamed analyte: source PFUDA -> canonical PFUnA
        _row(1010002, "S2", "PFC_EXPANDED", "Perfluoroundecanoic acid (PFUDA)", 3.0, "ng/L", 2.0),
        # truncated unclosed-paren analyte -> 11Cl-PF3OUdS
        _row(
            1010002,
            "S2",
            "PFC_EXPANDED",
            "11-chloroeicosafluoro-3-oxandecane-1-sulfonic acid (11CI-PF3",
            2.5,
            "ng/L",
            2.0,
        ),
    ]
    path = tmp_data_dirs["raw"] / "mn_mdh_pfas.xlsx"
    _write_xlsx(path, rows)
    return path


class TestMnMdhSource:
    def test_name_property(self, mn_source: MnMdhSource) -> None:
        assert mn_source.name == "mn_mdh"

    def test_config_loaded(self, mn_source: MnMdhSource) -> None:
        assert "expected_files" in mn_source._config
        # The manual-export config must NOT be marked unavailable, or download.py
        # would purge the parquet as a stale cache.
        assert not mn_source._config.get("unavailable")

    def test_parse_schema_columns(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        for col in (
            "pwsid",
            "analyte",
            "concentration",
            "unit",
            "censored",
            "detection_limit",
            "sample_date",
            "latitude",
            "longitude",
        ):
            assert col in df.columns

    def test_analyte_group_dedup(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        """The 3 replicated PFOS panel rows collapse to a single measurement."""
        df = mn_source.parse()
        assert len(df) == 7  # 9 raw rows, 3 PFOS panels -> 1
        assert (df["analyte"] == "PFOS").sum() == 1

    def test_analyte_mapping(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        analytes = set(df["analyte"])
        assert analytes == {"PFOS", "PFOA", "PFBS", "PFHxS", "FOSA", "PFUnA", "11Cl-PF3OUdS"}

    def test_ngl_detect_conversion(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        pfos = df[df["analyte"] == "PFOS"].iloc[0]
        assert bool(pfos["censored"]) is False
        assert abs(pfos["concentration"] - 10.0 * PPT_TO_UGL) < 1e-12

    def test_ugl_detect_conversion_and_strip(
        self, mn_source: MnMdhSource, sample_xlsx: Path
    ) -> None:
        """A padded 'ug/L  ' unit must strip and pass through (factor 1.0)."""
        df = mn_source.parse()
        pfoa = df[df["analyte"] == "PFOA"].iloc[0]
        assert bool(pfoa["censored"]) is False
        assert abs(pfoa["concentration"] - 0.07) < 1e-12
        assert (df["unit"] == "ug/L").all()

    def test_nondetect_at_rl(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        pfbs = df[df["analyte"] == "PFBS"].iloc[0]
        assert bool(pfbs["censored"]) is True
        assert pfbs["concentration"] == 0.0
        assert abs(pfbs["detection_limit"] - 1.8 * PPT_TO_UGL) < 1e-12

    def test_below_rl_is_censored(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        pfhxs = df[df["analyte"] == "PFHxS"].iloc[0]
        assert bool(pfhxs["censored"]) is True
        assert pfhxs["concentration"] == 0.0

    def test_coordinates_nan(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        assert df["latitude"].isna().all()
        assert df["longitude"].isna().all()

    def test_sample_date_parsed(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        assert df["sample_date"].notna().all()
        assert pd.api.types.is_datetime64_any_dtype(df["sample_date"])

    def test_pwsid_format(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        for pwsid in df["pwsid"]:
            assert pwsid.startswith("MN")
            assert len(pwsid) == 9
            assert " " not in pwsid

    def test_pwsid_zero_pad(self, mn_source: MnMdhSource, tmp_data_dirs: dict[str, Path]) -> None:
        rows = [_row(123, "S1", "PFAS_533", "Perfluorooctanesulfonate (PFOS)", 10.0, "ng/L", 2.0)]
        _write_xlsx(tmp_data_dirs["raw"] / "mn_mdh_pfas.xlsx", rows)
        df = mn_source.parse()
        assert df.iloc[0]["pwsid"] == "MN0000123"

    def test_dedup_conflict_keeps_most_sensitive(
        self, mn_source: MnMdhSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Conflicting redundant rows resolve to the lowest-normalized-RL reading."""
        rows = [
            # ng/L detect (normalized RL 0.002 ug/L) — should win
            _row(1010003, "S3", "PFAS_533", "Perfluorooctanesulfonate (PFOS)", 10.0, "ng/L", 2.0),
            # ug/L non-detect (normalized RL 0.05 ug/L) — less sensitive
            _row(
                1010003,
                "S3",
                "PFC_EXPANDED",
                "Perfluorooctanesulfonate (PFOS)",
                0.001,
                "ug/L",
                0.05,
            ),
        ]
        _write_xlsx(tmp_data_dirs["raw"] / "mn_mdh_pfas.xlsx", rows)
        df = mn_source.parse()
        assert len(df) == 1
        assert bool(df.iloc[0]["censored"]) is False
        assert abs(df.iloc[0]["concentration"] - 10.0 * PPT_TO_UGL) < 1e-12

    def test_unmapped_unit_raises(
        self, mn_source: MnMdhSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        rows = [
            _row(1010001, "S1", "PFAS_533", "Perfluorooctanesulfonate (PFOS)", 1.0, "mg/L", 0.5)
        ]
        _write_xlsx(tmp_data_dirs["raw"] / "mn_mdh_pfas.xlsx", rows)
        with pytest.raises(ValueError, match="Unrecognized MN MDH unit"):
            mn_source.parse()

    def test_validate_retains_nan_coords(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        """validate() keeps rows despite NaN coordinates (skip_coord_check)."""
        df = mn_source.validate(mn_source.parse())
        assert len(df) == 7

    def test_validate_warns_unexpected_analyte(
        self, mn_source: MnMdhSource, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["MN1010001"],
                "analyte": ["UNKNOWN_PFAS"],
                "concentration": [1.0],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [0.5],
                "sample_date": pd.to_datetime(["2023-06-15"]),
                "latitude": [np.nan],
                "longitude": [np.nan],
            }
        )
        with caplog.at_level("WARNING"):
            mn_source.validate(df)
        assert "Unexpected MN MDH analytes" in caplog.text

    def test_to_parquet_roundtrip(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        df = mn_source.parse()
        path = mn_source.to_parquet(df)
        assert path.exists()
        assert len(pd.read_parquet(path)) == len(df)

    def test_download_skip_existing(self, mn_source: MnMdhSource, sample_xlsx: Path) -> None:
        result = mn_source.download()
        assert len(result) == 1
        assert result[0].exists()

    def test_download_missing_raises(self, mn_source: MnMdhSource) -> None:
        with pytest.raises(FileNotFoundError, match="MN MDH export not found"):
            mn_source.download()
