"""UCMR3 data source — 38 contaminants incl. 6 PFAS (~1.1M samples)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import UCMR3_ANALYTES
from aquacontam.data._ucmr_common import download_and_extract_ucmr_zip, parse_ucmr_txt
from aquacontam.data.base import DataSource

logger = logging.getLogger(__name__)


class UCMR3Source(DataSource):
    """EPA Unregulated Contaminant Monitoring Rule 3 data loader.

    UCMR3 (2013-2015) monitored 6 PFAS compounds at public water systems
    serving ≥ 10,000 people. Used for temporal prediction (benchmark T7:
    predict UCMR5 contamination from historical UCMR3 trends).

    Parameters
    ----------
    raw_dir : Path
        Directory for raw downloaded files.
    interim_dir : Path
        Directory for intermediate artifacts.
    processed_dir : Path
        Directory for final Parquet output.
    config_path : Path | str | None
        Path to the data config YAML. Defaults to ``configs/data.yaml``.
    """

    def __init__(
        self,
        raw_dir: Path,
        interim_dir: Path,
        processed_dir: Path,
        config_path: Path | str | None = None,
    ) -> None:
        super().__init__(raw_dir, interim_dir, processed_dir)
        self._config = load_data_config(config_path, source_name="ucmr3")

    @property
    def name(self) -> str:
        return "ucmr3"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download UCMR3 occurrence data ZIP from EPA."""
        return download_and_extract_ucmr_zip(
            url=self._config["url"],
            raw_dir=self.raw_dir,
            expected_files=self._config["expected_files"],
            force=force,
        )

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse UCMR3 tab-delimited occurrence file."""
        txt_file = self.raw_dir / self._config["expected_files"][0]
        return parse_ucmr_txt(txt_file, self._config)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the parsed UCMR3 DataFrame."""
        return self._validate_standard(
            df, set(UCMR3_ANALYTES), skip_coord_check=True, check_duplicates=False
        )
