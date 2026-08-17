"""Abstract base class for all data source ingestion pipelines."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Base class that every data source module must implement.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw downloaded files.
    interim_dir : Path
        Directory for intermediate processing artifacts.
    processed_dir : Path
        Directory for final ML-ready outputs.
    """

    def __init__(
        self,
        raw_dir: Path,
        interim_dir: Path,
        processed_dir: Path,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this data source (e.g., 'ucmr5', 'sdwis')."""

    @abstractmethod
    def download(self, **kwargs: Any) -> list[Path]:
        """Download raw data files from the source.

        Returns
        -------
        list[Path]
            Paths to the downloaded files.
        """

    @abstractmethod
    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse raw files into a standardized DataFrame.

        Returns
        -------
        pd.DataFrame
            Parsed data with standardized column names and types.
        """

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the parsed DataFrame against the source schema.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame (may drop invalid rows with warnings).

        Raises
        ------
        ValueError
            If critical validation errors are found.
        """

    def to_parquet(self, df: pd.DataFrame, **kwargs: Any) -> Path:
        """Write the validated DataFrame to Parquet format.

        Parameters
        ----------
        df : pd.DataFrame
            Validated DataFrame to persist.

        Returns
        -------
        Path
            Path to the written Parquet file.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        out = self.processed_dir / f"{self.name}.parquet"
        df.to_parquet(out, compression="snappy", index=False)
        logger.info("Wrote %d rows to %s", len(df), out)
        return out

    def _validate_standard(
        self,
        df: pd.DataFrame,
        expected_analytes: set[str],
        *,
        skip_coord_check: bool = False,
        check_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Run standard validation: schema, analyte check, duplicate check.

        Parameters
        ----------
        df : pd.DataFrame
            Parsed DataFrame.
        expected_analytes : set[str]
            Expected analyte names for this source.
        skip_coord_check : bool
            Skip coordinate validation in schema check.
        check_duplicates : bool
            Check for duplicate (pwsid, analyte, sample_date) rows.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame.
        """
        from aquacontam.data.schema import validate_schema

        df = validate_schema(df, skip_coord_check=skip_coord_check, drop_invalid_rows=True)

        if "analyte" in df.columns:
            unique_analytes = set(df["analyte"].unique())
            unexpected = unique_analytes - expected_analytes
            if unexpected:
                logger.warning("Unexpected analytes in %s data: %s", self.name, sorted(unexpected))
            missing = expected_analytes - unique_analytes
            if missing:
                logger.info(
                    "Expected analytes not found in %s data: %s", self.name, sorted(missing)
                )

        if check_duplicates:
            dup_cols = ["pwsid", "analyte", "sample_date"]
            if all(c in df.columns for c in dup_cols):
                n_dups = df.duplicated(subset=dup_cols).sum()
                if n_dups:
                    logger.warning(
                        "%d duplicate samples in %s (pwsid + analyte + date)",
                        n_dups,
                        self.name,
                    )

        return df

    def run(self, **kwargs: Any) -> Path:
        """Execute the full pipeline: download → parse → validate → save.

        Returns
        -------
        Path
            Path to the final Parquet file.
        """
        self.download(**kwargs)
        df = self.parse(**kwargs)
        df = self.validate(df)
        return self.to_parquet(df, **kwargs)
