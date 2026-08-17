"""Tests for the DataSource abstract base class contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from aquacontam.data.base import DataSource


class ConcreteSource(DataSource):
    """Minimal concrete subclass for testing."""

    @property
    def name(self) -> str:
        return "test_source"

    def download(self, **kwargs: Any) -> list[Path]:
        return [self.raw_dir / "test.csv"]

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"pwsid": ["CA0101001"], "analyte": ["PFOS"]})

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def to_parquet(self, df: pd.DataFrame, **kwargs: Any) -> Path:
        return self.processed_dir / "test.parquet"


class TestDataSourceABC:
    """Tests for DataSource ABC enforcement."""

    def test_cannot_instantiate_directly(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="abstract"):
            DataSource(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self, tmp_path: Path) -> None:
        source = ConcreteSource(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
        assert source.name == "test_source"

    def test_dirs_stored_as_paths(self, tmp_path: Path) -> None:
        source = ConcreteSource(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
        assert isinstance(source.raw_dir, Path)
        assert isinstance(source.interim_dir, Path)
        assert isinstance(source.processed_dir, Path)


class TestDataSourceRun:
    """Tests for the run() pipeline orchestration."""

    def test_run_calls_steps_in_order(self, tmp_path: Path) -> None:
        source = ConcreteSource(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
        call_order: list[str] = []

        original_download = source.download
        original_parse = source.parse
        original_validate = source.validate
        original_to_parquet = source.to_parquet

        def mock_download(**kwargs: Any) -> list[Path]:
            call_order.append("download")
            return original_download(**kwargs)

        def mock_parse(**kwargs: Any) -> pd.DataFrame:
            call_order.append("parse")
            return original_parse(**kwargs)

        def mock_validate(df: pd.DataFrame) -> pd.DataFrame:
            call_order.append("validate")
            return original_validate(df)

        def mock_to_parquet(df: pd.DataFrame, **kwargs: Any) -> Path:
            call_order.append("to_parquet")
            return original_to_parquet(df, **kwargs)

        with (
            patch.object(source, "download", side_effect=mock_download),
            patch.object(source, "parse", side_effect=mock_parse),
            patch.object(source, "validate", side_effect=mock_validate),
            patch.object(source, "to_parquet", side_effect=mock_to_parquet),
        ):
            source.run()

        assert call_order == ["download", "parse", "validate", "to_parquet"]

    def test_run_returns_parquet_path(self, tmp_path: Path) -> None:
        source = ConcreteSource(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
        result = source.run()
        assert result == source.processed_dir / "test.parquet"

    def test_run_propagates_download_error(self, tmp_path: Path) -> None:
        source = ConcreteSource(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
        with (
            patch.object(source, "download", side_effect=RuntimeError("download failed")),
            pytest.raises(RuntimeError, match="download failed"),
        ):
            source.run()

    def test_run_propagates_parse_error(self, tmp_path: Path) -> None:
        source = ConcreteSource(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
        with (
            patch.object(source, "parse", side_effect=ValueError("bad data")),
            pytest.raises(ValueError, match="bad data"),
        ):
            source.run()

    def test_run_propagates_validate_error(self, tmp_path: Path) -> None:
        source = ConcreteSource(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
        with (
            patch.object(source, "validate", side_effect=ValueError("validation failed")),
            pytest.raises(ValueError, match="validation failed"),
        ):
            source.run()
