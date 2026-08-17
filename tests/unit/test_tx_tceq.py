"""Tests for Texas TCEQ data source (unavailable placeholder)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam.data.tx_tceq import TxTceqSource


@pytest.fixture()
def tx_source(tmp_data_dirs: dict[str, Path]) -> TxTceqSource:
    return TxTceqSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


class TestTxTceqSource:
    """Tests for TxTceqSource."""

    def test_name_property(self, tx_source: TxTceqSource) -> None:
        assert tx_source.name == "tx_tceq"

    def test_config_loaded(self, tx_source: TxTceqSource) -> None:
        assert tx_source._config.get("unavailable") is True
        assert "expected_files" in tx_source._config

    def test_download_raises_runtime_error(self, tx_source: TxTceqSource) -> None:
        """Download should raise RuntimeError since data is unavailable."""
        with pytest.raises(RuntimeError, match="TX TCEQ download unavailable"):
            tx_source.download()

    def test_parse_raises_file_not_found(self, tx_source: TxTceqSource) -> None:
        """Parse should raise FileNotFoundError when no data file exists."""
        with pytest.raises(FileNotFoundError, match="TX TCEQ data file not found"):
            tx_source.parse()

    def test_parse_with_manual_data(
        self, tx_source: TxTceqSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Parse should work if a manually obtained CSV is present."""
        data = pd.DataFrame(
            {
                "PWSID": ["TX1010001", "TX2020002"],
                "analyte": ["PFOS", "PFOA"],
                "concentration": [0.015, 0.0],
                "censored": ["False", "True"],
                "detection_limit": [0.004, 0.004],
                "sample_date": ["2023-05-10", "2023-06-15"],
                "latitude": [30.27, 32.78],
                "longitude": [-97.74, -96.80],
            }
        )
        path = tmp_data_dirs["raw"] / "tx_tceq_pfas.csv"
        data.to_csv(path, index=False)

        df = tx_source.parse()
        assert len(df) == 2
        assert "pwsid" in df.columns
        assert df.iloc[0]["pwsid"] == "TX1010001"

    def test_pwsid_normalization(
        self, tx_source: TxTceqSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Short PWSIDs should be zero-padded with TX prefix."""
        data = pd.DataFrame(
            {
                "PWSID": ["123"],
                "analyte": ["PFOS"],
                "concentration": [0.015],
                "censored": ["False"],
                "detection_limit": [0.004],
                "sample_date": ["2023-05-10"],
                "latitude": [30.27],
                "longitude": [-97.74],
            }
        )
        path = tmp_data_dirs["raw"] / "tx_tceq_pfas.csv"
        data.to_csv(path, index=False)

        df = tx_source.parse()
        assert df.iloc[0]["pwsid"] == "TX0000123"

    def test_validate(self, tx_source: TxTceqSource) -> None:
        """Validate should pass for well-formed data."""
        df = pd.DataFrame(
            {
                "pwsid": ["TX1010001"],
                "analyte": ["PFOS"],
                "concentration": [0.015],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [0.004],
                "sample_date": pd.to_datetime(["2023-05-10"]),
                "latitude": [30.27],
                "longitude": [-97.74],
            }
        )
        result = tx_source.validate(df)
        assert len(result) == 1
