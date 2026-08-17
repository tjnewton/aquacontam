"""Tests for T6: public-supply → domestic arsenic transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.benchmark.private_wells import run_t6
from aquacontam.benchmark.registry import TaskResult, get_task, list_tasks
from aquacontam.models.random_forest import RandomForestClassifier

# Ten states across distinct EPA regions so the geographic split is non-empty.
_STATE_COORDS = {
    "CT": (41.6, -72.7),
    "NJ": (40.2, -74.7),
    "PA": (40.3, -76.9),
    "GA": (33.7, -84.4),
    "IL": (40.6, -89.4),
    "TX": (31.0, -97.0),
    "KS": (38.5, -98.8),
    "CO": (39.5, -105.0),
    "CA": (36.8, -119.4),
    "WA": (47.4, -120.5),
}


def _arsenic_wells(states, n_per_state, seed, prefix):
    rng = np.random.RandomState(seed)
    rows = []
    ids = []
    for state in states:
        lat, lon = _STATE_COORDS[state]
        for i in range(n_per_state):
            pwsid = f"{prefix}{state}{i:05d}"
            ids.append(pwsid)
            for _ in range(2):
                exceed = rng.random() < 0.18
                censored = (not exceed) and rng.random() < 0.4
                conc = (
                    0.0
                    if censored
                    else (12.0 + rng.exponential(8.0) if exceed else rng.uniform(0.5, 9.0))
                )
                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": "arsenic",
                        "concentration": conc,
                        "unit": "ug/L",
                        "censored": censored,
                        "detection_limit": 1.0,
                        "sample_date": pd.Timestamp("2015-06-01"),
                        "latitude": lat + rng.normal(0, 0.4),
                        "longitude": lon + rng.normal(0, 0.4),
                    }
                )
    return pd.DataFrame(rows), ids


@pytest.fixture()
def public_arsenic_data():
    df, ids = _arsenic_wells(list(_STATE_COORDS), 8, 42, "PUB")
    df.attrs["ids"] = ids
    return df


@pytest.fixture()
def domestic_arsenic_data():
    df, ids = _arsenic_wells(["NJ", "PA", "TX", "CA"], 6, 7, "DOM")
    df.attrs["ids"] = ids
    return df


@pytest.fixture()
def nga_feature_df(public_arsenic_data, domestic_arsenic_data):
    """Synthetic environmental features for every well (public + domestic)."""
    all_ids = sorted(set(public_arsenic_data["pwsid"]) | set(domestic_arsenic_data["pwsid"]))
    rng = np.random.RandomState(99)
    return pd.DataFrame(
        {
            "depth_proxy": rng.uniform(20, 400, len(all_ids)),
            "geo_index": rng.normal(0, 1, len(all_ids)),
        },
        index=pd.Index(all_ids, name="pwsid"),
    )


def _rf():
    return RandomForestClassifier(config={"n_estimators": 15, "random_state": 42})


class TestT6Registration:
    def test_t6_registered(self) -> None:
        info, _ = get_task("T6")
        assert info.task_type == "classification"
        assert info.primary_metric == "auprc"

    def test_expected_tasks_registered(self) -> None:
        names = [t.name for t in list_tasks()]
        for expected in ["T1", "T2", "T4", "T5", "T6", "T7"]:
            assert expected in names, f"Missing task: {expected}"


class TestT6Task:
    def test_runs_geographic(
        self, public_arsenic_data, domestic_arsenic_data, nga_feature_df
    ) -> None:
        result = run_t6(
            model=_rf(),
            data=public_arsenic_data,
            well_data=domestic_arsenic_data,
            feature_dfs=[nga_feature_df],
        )
        assert isinstance(result, TaskResult)
        assert result.task_name == "T6"
        assert "train" in result.split_metrics
        assert "holdout" in result.split_metrics  # zero-shot domestic eval

    def test_primary_is_domestic_holdout(
        self, public_arsenic_data, domestic_arsenic_data, nga_feature_df
    ) -> None:
        result = run_t6(
            model=_rf(),
            data=public_arsenic_data,
            well_data=domestic_arsenic_data,
            feature_dfs=[nga_feature_df],
        )
        assert result.metrics == result.split_metrics["holdout"]

    def test_random_split_runs(
        self, public_arsenic_data, domestic_arsenic_data, nga_feature_df
    ) -> None:
        result = run_t6(
            model=_rf(),
            data=public_arsenic_data,
            well_data=domestic_arsenic_data,
            feature_dfs=[nga_feature_df],
            split_strategy="random",
        )
        assert result.metadata["split_strategy"] == "random"
        assert "holdout" in result.split_metrics

    def test_metadata(self, public_arsenic_data, domestic_arsenic_data, nga_feature_df) -> None:
        result = run_t6(
            model=_rf(),
            data=public_arsenic_data,
            well_data=domestic_arsenic_data,
            feature_dfs=[nga_feature_df],
        )
        assert result.metadata["analyte"] == "arsenic"
        assert result.metadata["target"] == "action_level"
        assert "n_train" in result.metadata
        assert result.metadata["n_domestic_holdout"] > 0

    def test_default_analyte_is_arsenic(
        self, public_arsenic_data, domestic_arsenic_data, nga_feature_df
    ) -> None:
        result = run_t6(
            model=_rf(),
            data=public_arsenic_data,
            well_data=domestic_arsenic_data,
            feature_dfs=[nga_feature_df],
        )
        assert result.metadata["analyte"] == "arsenic"

    def test_empty_training_data(self, domestic_arsenic_data, nga_feature_df) -> None:
        empty_df = pd.DataFrame(
            columns=[
                "pwsid",
                "analyte",
                "concentration",
                "unit",
                "censored",
                "detection_limit",
                "sample_date",
                "latitude",
                "longitude",
            ]
        )
        result = run_t6(
            model=_rf(),
            data=empty_df,
            well_data=domestic_arsenic_data,
            feature_dfs=[nga_feature_df],
        )
        assert "error" in result.metadata


def test_run_t6_experiment_forwards_use_tuned_params(tmp_path, monkeypatch) -> None:
    """run_t6_experiment threads use_tuned_params into get_model_instances.

    CI-safe: mocks NGA loading + feature build + model factory, so no
    NGA/network/GPU. Returning [] models skips the inner loop.
    """
    import aquacontam.pipeline.t6_arsenic as t6mod

    fake_wells = pd.DataFrame({"pwsid": ["A", "B"], "concentration": [1.0, 12.0]})
    monkeypatch.setattr(t6mod, "load_nga_arsenic", lambda *a, **k: fake_wells)
    monkeypatch.setattr(t6mod, "build_nga_feature_dfs", lambda *a, **k: [])
    captured: dict = {}

    def fake_get_model_instances(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(t6mod, "get_model_instances", fake_get_model_instances)

    t6mod.run_t6_experiment(
        tmp_path, out_path=tmp_path / "t6.json", download=False, use_tuned_params=True
    )
    assert captured.get("use_tuned_params") is True
    assert captured.get("task_name") == "T6"
