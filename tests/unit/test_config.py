"""Tests for data configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aquacontam._config import load_data_config


class TestLoadDataConfig:
    """Verify config loading from YAML."""

    def test_loads_full_config(self, tmp_path: Path) -> None:
        cfg = {"defaults": {"crs": "EPSG:4326"}, "ucmr5": {"url": "https://example.com"}}
        path = tmp_path / "data.yaml"
        path.write_text(yaml.dump(cfg))

        result = load_data_config(path)
        assert "defaults" in result
        assert "ucmr5" in result
        assert result["ucmr5"]["url"] == "https://example.com"

    def test_loads_source_specific_section(self, tmp_path: Path) -> None:
        cfg = {"defaults": {"crs": "EPSG:4326"}, "ucmr5": {"url": "https://example.com"}}
        path = tmp_path / "data.yaml"
        path.write_text(yaml.dump(cfg))

        result = load_data_config(path, source_name="ucmr5")
        assert result["url"] == "https://example.com"
        assert "defaults" not in result

    def test_missing_source_raises_key_error(self, tmp_path: Path) -> None:
        cfg = {"defaults": {}, "ucmr5": {}}
        path = tmp_path / "data.yaml"
        path.write_text(yaml.dump(cfg))

        with pytest.raises(KeyError, match="nosuch"):
            load_data_config(path, source_name="nosuch")

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_data_config("/nonexistent/path.yaml")

    def test_default_config_loads(self) -> None:
        """The bundled configs/data.yaml should load without errors."""
        result = load_data_config()
        assert "ucmr5" in result
        assert "ucmr3" in result
