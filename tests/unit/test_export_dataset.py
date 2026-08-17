"""Tests for dataset export pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aquacontam.export.dataset import (
    _sha256_file,
    _write_manifest,
    _write_split_assignments,
    export_dataset,
)


@pytest.fixture()
def sample_data() -> pd.DataFrame:
    """Minimal water quality DataFrame for export tests."""
    return pd.DataFrame(
        {
            "pwsid": [
                "CT0000001",
                "CT0000002",  # Region 1 (train)
                "NJ0000001",
                "NJ0000002",  # Region 2 (val)
                "CA0000001",
                "CA0000002",  # Region 9 (test)
            ],
            "analyte": ["PFOS"] * 6,
            "concentration": [10.0, 0.0, 5.0, 0.0, 8.0, 0.0],
            "censored": [False, True, False, True, False, True],
            "detection_limit": [2.0] * 6,
            "unit": ["ug/L"] * 6,
            "sample_date": pd.Timestamp("2023-01-01"),
            "latitude": [41.0, 41.5, 40.0, 40.5, 34.0, 34.5],
            "longitude": [-73.0, -73.5, -74.0, -74.5, -118.0, -118.5],
        }
    )


class TestSha256File:
    """Tests for _sha256_file."""

    def test_computes_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = _sha256_file(f)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert _sha256_file(f1) != _sha256_file(f2)


class TestWriteManifest:
    """Tests for _write_manifest."""

    def test_creates_manifest_json(self, tmp_path: Path) -> None:
        (tmp_path / "data.parquet").write_bytes(b"fake")
        result = _write_manifest(tmp_path)
        assert result.name == "manifest.json"
        assert result.exists()

    def test_manifest_contains_checksums(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        _write_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            data = json.load(f)
        assert "checksums" in data
        assert len(data["checksums"]) == 2

    def test_manifest_has_version(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, version="2.0")
        with open(tmp_path / "manifest.json") as f:
            data = json.load(f)
        assert data["version"] == "2.0"

    def test_extra_metadata_included(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, extra_metadata={"custom_key": "value"})
        with open(tmp_path / "manifest.json") as f:
            data = json.load(f)
        assert data["custom_key"] == "value"


class TestWriteSplitAssignments:
    """Tests for _write_split_assignments."""

    def test_writes_csv(self, tmp_path: Path) -> None:
        train = pd.DataFrame({"pwsid": ["A"], "epa_region": [1]})
        val = pd.DataFrame({"pwsid": ["B"], "epa_region": [2]})
        test = pd.DataFrame({"pwsid": ["C"], "epa_region": [9]})
        path = _write_split_assignments(train, val, test, tmp_path)
        assert path.exists()
        result = pd.read_csv(path)
        assert len(result) == 3
        assert set(result["split"]) == {"train", "val", "test"}

    def test_handles_empty_splits(self, tmp_path: Path) -> None:
        train = pd.DataFrame({"pwsid": ["A"], "epa_region": [1]})
        val = pd.DataFrame(columns=["pwsid", "epa_region"])
        test = pd.DataFrame(columns=["pwsid", "epa_region"])
        path = _write_split_assignments(train, val, test, tmp_path)
        result = pd.read_csv(path)
        assert len(result) == 1


class TestExportDataset:
    """Tests for export_dataset."""

    def test_creates_output_directory(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        out = tmp_path / "dataset"
        export_dataset(sample_data, output_dir=out)
        assert out.is_dir()

    def test_writes_split_parquets(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        out = tmp_path / "dataset"
        export_dataset(sample_data, output_dir=out)
        assert (out / "data" / "train.parquet").exists()
        assert (out / "data" / "val.parquet").exists()
        assert (out / "data" / "test.parquet").exists()

    def test_writes_feature_files(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        features = {
            "proximity": pd.DataFrame(
                {"dist_industrial": [100.0, 200.0]},
                index=pd.Index(["CT0000001", "CT0000002"], name="pwsid"),
            )
        }
        out = tmp_path / "dataset"
        export_dataset(sample_data, feature_dfs=features, output_dir=out)
        assert (out / "features" / "proximity.parquet").exists()

    def test_writes_manifest(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        out = tmp_path / "dataset"
        export_dataset(sample_data, output_dir=out)
        assert (out / "manifest.json").exists()
        with open(out / "manifest.json") as f:
            manifest = json.load(f)
        assert "checksums" in manifest
        assert "version" in manifest

    def test_writes_split_assignments(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        out = tmp_path / "dataset"
        export_dataset(sample_data, output_dir=out)
        assert (out / "metadata" / "split_assignments.csv").exists()

    def test_writes_baseline_predictions(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        preds = pd.DataFrame({"pwsid": ["A"], "prediction": [1]})
        out = tmp_path / "dataset"
        export_dataset(sample_data, baseline_predictions=preds, output_dir=out)
        assert (out / "predictions" / "baselines.parquet").exists()

    def test_returns_output_dir(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        out = tmp_path / "dataset"
        result = export_dataset(sample_data, output_dir=out)
        assert result == out


class TestExportExclusionsAndArchiveExtras:
    """Redistribution exclusions, POSIX manifest, prediction tree, README."""

    def test_exclude_sources_drops_rows(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        data = sample_data.copy()
        data["source"] = ["ucmr5", "mn_mdh", "ucmr5", "mn_mdh", "ucmr5", "mn_mdh"]
        out = tmp_path / "dataset"
        export_dataset(data, output_dir=out, exclude_sources=("mn_mdh",))
        parts = [
            pd.read_parquet(out / "data" / f"{name}.parquet") for name in ("train", "val", "test")
        ]
        combined = pd.concat(parts, ignore_index=True)
        assert "mn_mdh" not in set(combined["source"])
        assert len(combined) == 3

    def test_exclude_sources_noop_without_source_column(
        self, tmp_path: Path, sample_data: pd.DataFrame
    ) -> None:
        out = tmp_path / "dataset"
        export_dataset(sample_data, output_dir=out, exclude_sources=("mn_mdh",))
        assert (out / "manifest.json").exists()

    def test_manifest_paths_are_posix(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        out = tmp_path / "dataset"
        export_dataset(sample_data, output_dir=out)
        with open(out / "manifest.json") as f:
            manifest = json.load(f)
        nested = [k for k in manifest["checksums"] if "/" in k]
        assert nested, "expected nested archive paths in the manifest"
        assert not any("\\" in k for k in manifest["checksums"])

    def test_predictions_tree_copied_and_checksummed(
        self, tmp_path: Path, sample_data: pd.DataFrame
    ) -> None:
        src = tmp_path / "preds" / "task=T1" / "analyte=PFOS"
        src.mkdir(parents=True)
        (src / "part-0.parquet").write_bytes(b"stub")
        out = tmp_path / "dataset"
        export_dataset(sample_data, output_dir=out, predictions_dir=tmp_path / "preds")
        assert (out / "predictions" / "task=T1" / "analyte=PFOS" / "part-0.parquet").exists()
        with open(out / "manifest.json") as f:
            manifest = json.load(f)
        assert "predictions/task=T1/analyte=PFOS/part-0.parquet" in manifest["checksums"]

    def test_readme_written(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        out = tmp_path / "dataset"
        export_dataset(sample_data, output_dir=out, readme_text="# Archive\n")
        assert (out / "README.md").read_text(encoding="utf-8").startswith("# Archive")

    def test_zenodo_pipeline_manifest_covers_final_metadata(
        self, tmp_path: Path, sample_data: pd.DataFrame
    ) -> None:
        """Regression: the manifest must hash the FINAL metadata files (stale-card bug)."""
        import hashlib

        from aquacontam.pipeline.export import export_zenodo_dataset

        data = sample_data.copy()
        data["source"] = ["ucmr5", "mn_mdh"] * 3
        export_zenodo_dataset(data, [], tmp_path / "data", tmp_path / "results")
        zen = tmp_path / "results" / "zenodo-dataset"
        with open(zen / "manifest.json") as f:
            manifest = json.load(f)
        assert "dataset_card.json" in manifest["checksums"]
        for rel, expected in manifest["checksums"].items():
            actual = hashlib.sha256((zen / rel).read_bytes()).hexdigest()
            assert actual == expected, f"stale checksum for {rel}"
        combined = pd.concat(
            [pd.read_parquet(zen / "data" / f"{n}.parquet") for n in ("train", "val", "test")],
            ignore_index=True,
        )
        assert "mn_mdh" not in set(combined["source"])
