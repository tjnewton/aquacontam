"""Tests for HPO + GPU plumbing introduced for the comprehensive sweep.

Covers:
- ``get_device(gpu_id=...)`` selects the requested CUDA device.
- Tree-library ``_apply_gpu_params`` helpers route ``use_gpu``/``gpu_id``
  to the right library-specific kwargs.
- ``SEARCH_SPACES_BY_TASK`` registry is well-formed and samples produce
  configs the model classes accept.
- SQLite study persistence allows resuming an interrupted study.
- ``load_best_config`` round-trips JSON (incl. legacy entries that lack
  the ``task`` field).
- ``get_model_instances(use_tuned_params=True)`` prefers the tuned config
  over the YAML default.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# get_device(gpu_id=...)
# ---------------------------------------------------------------------------


class TestGetDevice:
    @pytest.fixture(autouse=True)
    def _require_torch(self) -> None:
        pytest.importorskip("torch")

    def test_gpu_id_arg_returns_specific_cuda_device(self) -> None:
        from aquacontam.models import _torch_utils

        with patch.object(_torch_utils.torch.cuda, "is_available", return_value=True):
            dev = _torch_utils.get_device(gpu_id=2)
        assert str(dev) == "cuda:2"

    def test_env_var_used_when_no_gpu_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aquacontam.models import _torch_utils

        monkeypatch.setenv("AQUACONTAM_GPU_ID", "1")
        with patch.object(_torch_utils.torch.cuda, "is_available", return_value=True):
            dev = _torch_utils.get_device()
        assert str(dev) == "cuda:1"

    def test_default_cuda_when_no_args_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aquacontam.models import _torch_utils

        monkeypatch.delenv("AQUACONTAM_GPU_ID", raising=False)
        with patch.object(_torch_utils.torch.cuda, "is_available", return_value=True):
            dev = _torch_utils.get_device()
        # Honors CUDA_VISIBLE_DEVICES via the bare 'cuda' device.
        assert str(dev) == "cuda"

    def test_cpu_fallback_when_no_cuda(self) -> None:
        from aquacontam.models import _torch_utils

        with (
            patch.object(_torch_utils.torch.cuda, "is_available", return_value=False),
            patch.object(_torch_utils.torch.backends, "mps", create=True) as mps_mod,
        ):
            mps_mod.is_available = lambda: False
            dev = _torch_utils.get_device(gpu_id=2)
        assert str(dev) == "cpu"


# ---------------------------------------------------------------------------
# Tree-library GPU passthrough
# ---------------------------------------------------------------------------


class TestTreeLibGpuPassthrough:
    def _make_xy(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 5)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_xgboost_routes_use_gpu_and_gpu_id(self) -> None:
        from aquacontam.models.xgboost import _apply_gpu_params

        params: dict[str, Any] = {"n_estimators": 10, "use_gpu": True, "gpu_id": 0}
        _apply_gpu_params(params)
        assert params["device"] == "cuda:0"
        assert params["tree_method"] == "hist"
        assert "use_gpu" not in params and "gpu_id" not in params

    def test_xgboost_no_gpu_when_use_gpu_false(self) -> None:
        from aquacontam.models.xgboost import _apply_gpu_params

        params: dict[str, Any] = {"n_estimators": 10, "use_gpu": False}
        _apply_gpu_params(params)
        assert "device" not in params
        assert "tree_method" not in params

    def test_lightgbm_routes_use_gpu_and_gpu_id(self) -> None:
        pytest.importorskip("lightgbm")
        from aquacontam.models.lightgbm import _apply_gpu_params

        params: dict[str, Any] = {"n_estimators": 10, "use_gpu": True, "gpu_id": 0}
        _apply_gpu_params(params)
        assert params["device"] == "gpu"
        assert params["gpu_platform_id"] == 0
        assert params["gpu_device_id"] == 0
        assert "use_gpu" not in params

    def test_catboost_strips_auto_class_weights_when_gpu(self) -> None:
        pytest.importorskip("catboost")
        from aquacontam.models.catboost import _apply_gpu_params

        params: dict[str, Any] = {
            "iterations": 10,
            "use_gpu": True,
            "gpu_id": 0,
            "auto_class_weights": "Balanced",
        }
        _apply_gpu_params(params)
        assert params["task_type"] == "GPU"
        assert params["devices"] == "0"
        assert "auto_class_weights" not in params

    def test_catboost_no_gpu_keeps_auto_class_weights(self) -> None:
        pytest.importorskip("catboost")
        from aquacontam.models.catboost import _apply_gpu_params

        params: dict[str, Any] = {
            "iterations": 10,
            "use_gpu": False,
            "auto_class_weights": "Balanced",
        }
        _apply_gpu_params(params)
        assert "task_type" not in params
        # auto_class_weights stays — caller still passes it to constructor.
        assert params["auto_class_weights"] == "Balanced"


# ---------------------------------------------------------------------------
# SEARCH_SPACES_BY_TASK registry
# ---------------------------------------------------------------------------


class TestSearchSpaceRegistry:
    def test_all_seven_tasks_present(self) -> None:
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        assert set(SEARCH_SPACES_BY_TASK.keys()) == {
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6",
            "T7",
        }

    def test_search_spaces_alias_is_t1(self) -> None:
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES, SEARCH_SPACES_BY_TASK

        assert SEARCH_SPACES is SEARCH_SPACES_BY_TASK["T1"]

    def test_t1_has_classifiers(self) -> None:
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        t1 = SEARCH_SPACES_BY_TASK["T1"]
        for name in [
            "xgboost_classifier",
            "lightgbm_classifier",
            "catboost_classifier",
            "random_forest_classifier",
            "mlp_classifier",
            "cnn1d_classifier",
            "logistic_regression",
            "tabpfn_classifier",
        ]:
            assert name in t1, f"T1 missing {name}"

    def test_t2_has_regressors(self) -> None:
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        t2 = SEARCH_SPACES_BY_TASK["T2"]
        for name in [
            "xgboost_regressor",
            "lightgbm_regressor",
            "catboost_regressor",
            "random_forest_regressor",
            "tobit_regressor",
            "aft_regressor",
            "xgboost_aft_regressor",
            "hurdle_regressor",
        ]:
            assert name in t2, f"T2 missing {name}"

    def test_t4_gnn_spaces_narrowed(self) -> None:
        """T4 GNN search space is narrower than T1's to avoid CUDA wedges.

        Root cause: T4's spatial graph is 5.68x larger than T1's. With T1's
        wider bounds (k=30, n_hidden=256) the GCN forward+backward pass
        wedged the CUDA context. See PR #41 round 3.
        """
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        space = SEARCH_SPACES_BY_TASK["T4"]["gnn_gcn_classifier"]
        assert space["k"] == ("int", "3", "10"), f"T4 gnn_gcn k bound stale: {space['k']!r}"
        # n_hidden upper bound dropped from 256 -> 128
        assert "256" not in space["n_hidden"], (
            f"T4 gnn_gcn still allows 256: {space['n_hidden']!r}"
        )
        # epochs cap reduced from 400 -> 200
        assert space["epochs"] == ("int", "100", "200")

    def test_t4_gnn_sage_matches_gnn_gcn(self) -> None:
        """gnn_sage_classifier uses the same torch_geometric path; same narrowing."""
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        gcn = SEARCH_SPACES_BY_TASK["T4"]["gnn_gcn_classifier"]
        sage = SEARCH_SPACES_BY_TASK["T4"]["gnn_sage_classifier"]
        assert sage == gcn

    def test_t4_other_classifiers_unchanged(self) -> None:
        """The narrowing is GNN-only; other T4 classifiers still mirror T1."""
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        for model in ("xgboost_classifier", "catboost_classifier", "mlp_classifier"):
            assert SEARCH_SPACES_BY_TASK["T4"][model] == SEARCH_SPACES_BY_TASK["T1"][model], (
                f"{model}: T4 unexpectedly differs from T1"
            )

    def test_t4_excludes_tabpfn(self) -> None:
        """TabPFN rejects training sets > max_train_size (default 10000) at
        ``src/aquacontam/models/tabpfn.py:118``. T4 has ~47k training rows, so
        every TabPFN trial fails with ValueError before any modeling happens.
        Dropping the entry from the T4 registry avoids 300 wasted trials.
        See PR #41 round 3 follow-up.
        """
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        assert "tabpfn_classifier" not in SEARCH_SPACES_BY_TASK["T4"], (
            "T4 must not include tabpfn_classifier: train set ~47k exceeds the 10k cap"
        )
        # Sanity: tabpfn is still in T1 (training rows ~8k, under the cap)
        assert "tabpfn_classifier" in SEARCH_SPACES_BY_TASK["T1"]

    def test_logistic_regression_in_derived_task_spaces(self) -> None:
        """logistic_regression must participate in T3/T4 (and T5-T7) HPO.

        Its registry key lacks the ``classifier`` suffix, so the original
        substring filter ``"classifier" in k`` silently dropped it from every
        derived task — it then trained at the default ``C=1.0``. See PR #41
        item 3.
        """
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        for task in ("T3", "T4", "T5", "T6", "T7"):
            assert "logistic_regression" in SEARCH_SPACES_BY_TASK[task], (
                f"{task} must include logistic_regression in its search space"
            )
        assert "logistic_regression" in SEARCH_SPACES_BY_TASK["T1"]
        # The filter must not have leaked any regressor into T4.
        assert not any(k.endswith("_regressor") for k in SEARCH_SPACES_BY_TASK["T4"]), (
            "T4 search space unexpectedly contains a regressor"
        )

    def test_t6_space_curated_to_tabular_models(self) -> None:
        """T6 (arsenic public-supply -> domestic transfer) tunes only the tabular
        tree/linear/MLP models that suit it — not GNN/TabPFN/deep models (TabPFN
        exceeds its 10k cap on ~26k public-supply rows; GNN builds a large graph
        per trial). Matches the standalone runner's model set."""
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        t6 = SEARCH_SPACES_BY_TASK["T6"]
        for model in (
            "logistic_regression",
            "random_forest_classifier",
            "xgboost_classifier",
            "lightgbm_classifier",
            "catboost_classifier",
            "mlp_classifier",
        ):
            assert model in t6, f"T6 must tune {model}"
        for model in (
            "gnn_gcn_classifier",
            "gnn_sage_classifier",
            "tabpfn_classifier",
            "cnn1d_classifier",
            "deep_tobit_classifier",
        ):
            assert model not in t6, f"T6 must not tune {model} (unsuited to the ~26k-well task)"
        # logistic narrowed (L2, no saga) — T6's wide one-hot (aquifer) makes the
        # wide saga/L1 space slow, same as T4.
        assert "saga" not in t6["logistic_regression"]["solver"], (
            "T6 logistic must drop saga (wide one-hot matrix, like T4)"
        )
        assert t6["logistic_regression"]["penalty"] == ("categorical", "l2"), (
            "T6 logistic must fix penalty to L2"
        )

    def test_prepare_t6_data_implemented_returns_none_without_nga(self) -> None:
        """T6 HPO prep is implemented (no longer a NotImplementedError stub) and
        returns None gracefully when the NGA arsenic source is absent (e.g. CI)."""
        from unittest.mock import patch

        import pandas as pd

        from aquacontam.benchmark._prep_data import prepare_t6_data

        with patch(
            "aquacontam.data.nga_arsenic.load_nga_arsenic",
            side_effect=FileNotFoundError("no NGA data"),
        ):
            # model_cls is unused on the early-None path, so any object works.
            result = prepare_t6_data(object, pd.DataFrame(), [])
        assert result is None

    def test_t4_logistic_space_narrowed(self) -> None:
        """T4's logistic space drops saga + L1 and caps max_iter.

        On T4's ~47k rows with wide one-hot features, saga and L1-regularised
        fits (liblinear coordinate descent at ~13s/iteration) ran 50+ min,
        while L2 fits finish in ~2-3s. T4 fixes the penalty to L2 and drops
        saga so the in-process sweep stays fast. Mirrors the GNN T4 narrowing.
        See PR #41 item 3 follow-up.
        """
        from aquacontam.benchmark.optuna_hpo import SEARCH_SPACES_BY_TASK

        space = SEARCH_SPACES_BY_TASK["T4"]["logistic_regression"]
        assert "saga" not in space["solver"], "T4 logistic must drop the slow saga solver"
        assert space["penalty"] == ("categorical", "l2"), (
            f"T4 logistic must fix penalty to L2 (the slow path is L1): {space['penalty']!r}"
        )
        assert space["max_iter"] == ("int", "200", "1000"), (
            f"T4 logistic max_iter not narrowed: {space['max_iter']!r}"
        )
        # T1 keeps the wider space (saga + L1 present).
        t1 = SEARCH_SPACES_BY_TASK["T1"]["logistic_regression"]
        assert "saga" in t1["solver"] and "l1" in t1["penalty"]

    def test_sample_valid_for_xgb_classifier(self) -> None:
        """A sampled trial config should be acceptable to the model class."""
        from aquacontam.benchmark.optuna_hpo import (
            SEARCH_SPACES_BY_TASK,
            _apply_param_transforms,
            _suggest_param,
        )
        from aquacontam.models.xgboost import XGBoostClassifier

        space = SEARCH_SPACES_BY_TASK["T1"]["xgboost_classifier"]
        trial = MagicMock()
        # Seed the mock to return the lower bound for each suggest.
        trial.suggest_int = lambda name, lo, hi: lo
        trial.suggest_float = lambda name, lo, hi, log=False: lo
        trial.suggest_categorical = lambda name, choices: choices[0]

        config = {name: _suggest_param(trial, name, spec) for name, spec in space.items()}
        _apply_param_transforms(config, "xgboost_classifier")
        # Must instantiate without error.
        XGBoostClassifier(config=config)


# ---------------------------------------------------------------------------
# SQLite study resume + load_best_config round-trip
# ---------------------------------------------------------------------------


class TestStudyPersistence:
    def test_sqlite_study_resume_accumulates_trials(self) -> None:
        pytest.importorskip("optuna")
        import optuna

        from aquacontam.benchmark.optuna_hpo import optuna_search
        from aquacontam.models.xgboost import XGBoostClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Windows: optuna's SQLite handle stays open after the study object
        # falls out of scope; ignore_cleanup_errors avoids a teardown race.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = Path(tmp) / "study.db"
            url = f"sqlite:///{db_path.as_posix()}"

            optuna_search(
                XGBoostClassifier,
                X,
                y,
                X,
                y,
                n_trials=3,
                metric="auprc",
                seed=42,
                storage=url,
                study_name="resume_test",
                sampler="tpe",
                pruner="median",
            )
            optuna_search(
                XGBoostClassifier,
                X,
                y,
                X,
                y,
                n_trials=2,
                metric="auprc",
                seed=42,
                storage=url,
                study_name="resume_test",
                sampler="tpe",
                pruner="median",
            )
            study = optuna.load_study(study_name="resume_test", storage=url)
            assert len(study.trials) == 5

    def test_load_best_config_round_trip(self) -> None:
        from aquacontam.benchmark.optuna_hpo import load_best_config

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                [
                    {
                        "model": "xgboost_classifier",
                        "task": "T1",
                        "best_config": {"n_estimators": 999, "random_state": 42},
                    },
                    {
                        "model": "xgboost_regressor",
                        "task": "T2",
                        "best_config": {"n_estimators": 1234},
                    },
                    # Legacy entry with no task field — should be readable as T1.
                    {
                        "model": "legacy_model",
                        "best_config": {"foo": "bar"},
                    },
                ],
                f,
            )
            p = Path(f.name)

        cfg_t1 = load_best_config("xgboost_classifier", "T1", p)
        cfg_t2 = load_best_config("xgboost_regressor", "T2", p)
        cfg_legacy = load_best_config("legacy_model", "T1", p)
        cfg_missing = load_best_config("xgboost_classifier", "T2", p)

        assert cfg_t1 == {"n_estimators": 999, "random_state": 42}
        assert cfg_t2 == {"n_estimators": 1234}
        assert cfg_legacy == {"foo": "bar"}
        assert cfg_missing is None

    def test_load_best_config_missing_file_returns_none(self) -> None:
        from aquacontam.benchmark.optuna_hpo import load_best_config

        cfg = load_best_config("xgboost_classifier", "T1", "/nonexistent/path.json")
        assert cfg is None


# ---------------------------------------------------------------------------
# get_model_instances(use_tuned_params=True)
# ---------------------------------------------------------------------------


class TestUseTunedParams:
    def test_prefers_tuned_over_yaml(self) -> None:
        from aquacontam.pipeline.models import get_model_instances

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                [
                    {
                        "model": "xgboost_classifier",
                        "task": "T1",
                        "best_config": {
                            "n_estimators": 4242,
                            "max_depth": 5,
                            "random_state": 42,
                        },
                    }
                ],
                f,
            )
            tuned_path = f.name

        models = get_model_instances(
            ["xgboost"],
            "classification",
            seed=42,
            task_name="T1",
            use_tuned_params=True,
            tuned_params_path=tuned_path,
        )
        xgb = next(m for m in models if "xgboost" in m.name)
        assert xgb.config.get("n_estimators") == 4242

    def test_falls_back_to_yaml_when_tuned_missing(self) -> None:
        from aquacontam.pipeline.models import get_model_instances

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                [
                    {
                        "model": "xgboost_classifier",
                        "task": "T1",
                        "best_config": {"n_estimators": 4242},
                    }
                ],
                f,
            )
            tuned_path = f.name

        # T4 has no tuned entry → should pick YAML defaults.
        models = get_model_instances(
            ["xgboost"],
            "classification",
            seed=42,
            task_name="T4",
            use_tuned_params=True,
            tuned_params_path=tuned_path,
        )
        xgb = next(m for m in models if "xgboost" in m.name)
        # YAML default for xgboost_classifier is 500 in experiment.yaml.
        assert xgb.config.get("n_estimators") != 4242

    def test_no_tuned_when_flag_off(self) -> None:
        """Passing use_tuned_params=False keeps YAML defaults even if JSON exists."""
        from aquacontam.pipeline.models import get_model_instances

        models_default = get_model_instances(["xgboost"], "classification", seed=42)
        xgb = next(m for m in models_default if "xgboost" in m.name)
        n_yaml = xgb.config.get("n_estimators")

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                [
                    {
                        "model": "xgboost_classifier",
                        "task": "T1",
                        "best_config": {"n_estimators": 7777},
                    }
                ],
                f,
            )
            tuned_path = f.name

        models_no_tune = get_model_instances(
            ["xgboost"],
            "classification",
            seed=42,
            task_name="T1",
            use_tuned_params=False,
            tuned_params_path=tuned_path,
        )
        xgb_no = next(m for m in models_no_tune if "xgboost" in m.name)
        assert xgb_no.config.get("n_estimators") == n_yaml
        assert xgb_no.config.get("n_estimators") != 7777


# ---------------------------------------------------------------------------
# Registry-key threading + GPU gate (round-2 bug fixes)
# ---------------------------------------------------------------------------


class TestRegistryKeyThreading:
    def test_apply_param_transforms_uses_registry_key_for_ensembles(self) -> None:
        """Voting/stacking ensembles' transforms key on the registry name, not
        the short class.name. The fix passes the registry key through.
        """
        from aquacontam.benchmark.optuna_hpo import _apply_param_transforms

        # Voting: "xgb_rf" preset should expand to a concrete list.
        cfg = {"base_models": "xgb_rf", "voting": "soft"}
        _apply_param_transforms(cfg, "voting_ensemble_classifier")
        assert cfg["base_models"] == ["xgboost", "random_forest"]
        assert cfg["voting"] == "soft"  # untouched

        # Stacking: same transform mapping.
        cfg = {"base_models": "all_four", "cv": 5}
        _apply_param_transforms(cfg, "stacking_ensemble_classifier")
        assert cfg["base_models"] == ["xgboost", "random_forest", "lightgbm", "catboost"]

        # Short class-name lookup misses (this was the bug).
        cfg = {"base_models": "xgb_rf"}
        _apply_param_transforms(cfg, "voting_ensemble")
        assert cfg["base_models"] == "xgb_rf"  # unchanged — no entry for short name

    def test_optuna_search_honors_explicit_model_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the caller supplies model_name, that key drives PARAM_TRANSFORMS
        lookup — not the instance's ``.name`` property.
        """
        pytest.importorskip("optuna")

        from aquacontam.benchmark import optuna_hpo
        from aquacontam.benchmark.optuna_hpo import optuna_search

        # Inject a fake transform under a registry-key-style name. Maps the
        # categorical preset string "alpha" to a concrete list, the same way
        # _ENSEMBLE_BASE_PRESETS maps "xgb_rf" to ["xgboost", "random_forest"].
        fake_transforms = {"full_registry_name": {"preset_key": {"alpha": [1, 2, 3]}}}
        monkeypatch.setattr(optuna_hpo, "PARAM_TRANSFORMS", fake_transforms)

        seen_configs: list[dict[str, Any]] = []

        class _FakeModel:
            """Stub model whose .name returns a SHORT name (the bug-triggering pattern)."""

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            @property
            def name(self) -> str:
                return "short_name"

            def fit(self, X, y, **_kwargs):
                seen_configs.append(dict(self.config))

            def evaluate(self, X, y, **_kwargs):
                return {"auprc": 0.5}

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        # search_space suggests preset_key="alpha"; the transform should
        # replace that string with the list [1, 2, 3].
        optuna_search(
            _FakeModel,
            X,
            y,
            X,
            y,
            search_space={"preset_key": ("categorical", "alpha")},
            n_trials=1,
            metric="auprc",
            seed=42,
            sampler="tpe",
            pruner="median",
            model_name="full_registry_name",  # the registry key, not "short_name"
            inproc=True,  # nested _FakeModel isn't importable from a spawn child
        )

        assert seen_configs, "Model fit was never called"
        assert seen_configs[0]["preset_key"] == [1, 2, 3]


class TestEffectiveGpuId:
    """The _effective_gpu_id helper gates GPU routing per model_key.

    Tree-lib models receive the gpu_id; sklearn / torch models get None
    (they don't consume use_gpu/gpu_id kwargs). Models known to segfault
    on GPU (e.g. catboost_regressor) are forced to None even though they
    match a tree-lib prefix.
    """

    def test_xgboost_classifier_gets_gpu(self) -> None:
        from aquacontam.pipeline.analysis._tuning import _effective_gpu_id

        assert _effective_gpu_id("xgboost_classifier", 0) == 0

    def test_random_forest_classifier_gets_cpu(self) -> None:
        from aquacontam.pipeline.analysis._tuning import _effective_gpu_id

        # sklearn-backed; rejects use_gpu kwarg → must stay on CPU
        assert _effective_gpu_id("random_forest_classifier", 0) is None

    def test_catboost_regressor_forced_to_cpu(self) -> None:
        """catboost_regressor is in _GPU_UNSTABLE_MODELS due to T2 RMSE segfault."""
        from aquacontam.pipeline.analysis._tuning import _effective_gpu_id

        assert _effective_gpu_id("catboost_regressor", 0) is None

    def test_catboost_classifier_stays_on_gpu(self) -> None:
        """Classifier path was verified working 8h on T1 — must still get GPU."""
        from aquacontam.pipeline.analysis._tuning import _effective_gpu_id

        assert _effective_gpu_id("catboost_classifier", 0) == 0

    def test_none_gpu_id_stays_none(self) -> None:
        from aquacontam.pipeline.analysis._tuning import _effective_gpu_id

        assert _effective_gpu_id("xgboost_classifier", None) is None


# ---------------------------------------------------------------------------
# T4 prep target string (round-3 bug fix)
# ---------------------------------------------------------------------------


class TestT4PrepTarget:
    """Regression guard for the T4 data-prep target string.

    aggregate_to_system_level() accepts only ``"detected"``,
    ``"max_concentration"``, ``"action_level"`` — earlier code passed
    ``"action_level_exceedance"`` and every T4 study in the full sweep
    failed in data prep with a ValueError.
    """

    def test_prepare_t4_forwards_valid_target(self) -> None:
        from aquacontam.features.assembly import aggregate_to_system_level

        # Sanity-check the set of valid targets to anchor the test against
        # the assembly contract — if assembly grows a new target name, this
        # test continues to pass.
        valid_targets = {"detected", "max_concentration", "action_level"}

        # Inspect the prepare_t4_data source to confirm the literal it passes.
        import inspect

        from aquacontam.benchmark import _prep_data

        src = inspect.getsource(_prep_data.prepare_t4_data)
        # Extract the target=... literal passed in the prepare_train_val_test call.
        import re

        m = re.search(r'target=["\']([a-z_]+)["\']', src)
        assert m is not None, "prepare_t4_data must pass an explicit target= kwarg"
        target_str = m.group(1)
        assert target_str in valid_targets, (
            f"prepare_t4_data passes target={target_str!r}, which is not "
            f"in the set accepted by aggregate_to_system_level: {valid_targets}"
        )

        # Belt-and-suspenders: directly call aggregate_to_system_level with the
        # extracted target string against a tiny synthetic frame. If the string
        # is wrong, this raises ValueError. No analyte data is needed — passing
        # an empty filtered frame is fine.
        import pandas as pd

        empty_df = pd.DataFrame(
            {
                "pwsid": pd.Series(dtype="object"),
                "analyte": pd.Series(dtype="object"),
                "concentration": pd.Series(dtype="float64"),
                "censored": pd.Series(dtype="bool"),
                "detection_limit": pd.Series(dtype="float64"),
            }
        )
        # Should not raise for the analyte we won't find — the function
        # logs a warning and returns an empty frame.
        result = aggregate_to_system_level(empty_df, "lead", target=target_str)
        assert result is not None


class TestDefaultMetrics:
    """The DEFAULT_METRICS dict drives which metric each (task) tunes against.

    The metric string must be one that ``compute_classification_metrics`` (or
    its regression sibling) actually returns — otherwise ``optuna_search``
    sees ``None`` for the metric value and returns ``fail_value`` (``-inf``)
    on every trial.

    Regression guard: T4 was previously set to ``"mean_auprc"``, which the
    metrics helper didn't know about, so every T4 trial returned ``-inf``.
    """

    def test_t1_t4_use_classification_aware_metric(self) -> None:
        from aquacontam.benchmark.metrics import compute_classification_metrics
        from aquacontam.benchmark.optuna_hpo import DEFAULT_METRICS

        # Sanity-check the metric helper actually emits these keys.
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([0.2, 0.9, 0.6, 0.8])
        emitted = compute_classification_metrics(y_true, y_pred, y_proba)

        for task in ("T1", "T3", "T4", "T5", "T6", "T7"):
            metric = DEFAULT_METRICS[task]
            assert metric in emitted, (
                f"DEFAULT_METRICS[{task!r}] = {metric!r} not emitted by "
                f"compute_classification_metrics. Emitted keys: {sorted(emitted)}"
            )

    def test_t2_uses_regression_metric(self) -> None:
        from aquacontam.benchmark.optuna_hpo import DEFAULT_METRICS

        # T2 must be a regression metric — these aren't in classification's output.
        assert DEFAULT_METRICS["T2"] in {"rmse", "mae", "r2"}


class TestHpoTopUpResume:
    """Resume passes only the remaining trial budget to ``optuna_search``.

    Optuna's ``study.optimize(n_trials=N)`` runs N *new* trials. Without the
    top-up logic in ``_run_optuna_tuning``, resuming a 226/300 study with
    ``--n-trials 300`` overshoots to 526 trials (the historical gnn_gcn
    590-trial overshoot); passing ``--n-trials 74`` instead trips the
    skip-guard (226 >= 74) and does nothing.
    """

    @staticmethod
    def _seed_study(storage_dir: Path, n_complete: int) -> None:
        import optuna

        storage_dir.mkdir(parents=True, exist_ok=True)
        db = storage_dir / "xgboost_classifier_T1.db"
        study = optuna.create_study(
            study_name="xgboost_classifier_T1_seed42",
            storage=f"sqlite:///{db.as_posix()}",
            direction="maximize",
        )
        for _ in range(n_complete):
            study.add_trial(
                optuna.trial.create_trial(
                    params={},
                    distributions={},
                    value=0.5,
                    state=optuna.trial.TrialState.COMPLETE,
                )
            )

    @staticmethod
    def _run_tuning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_trials: int) -> MagicMock:
        """Drive ``_run_optuna_tuning`` with a stubbed prep fn and mocked search."""
        from types import SimpleNamespace

        import pandas as pd

        from aquacontam.benchmark import _prep_data, optuna_hpo
        from aquacontam.pipeline.analysis._tuning import _run_optuna_tuning

        task_data = _prep_data.TaskData(
            X_train=pd.DataFrame({"a": [0.0, 1.0]}),
            y_train=pd.Series([0, 1]),
            X_val=pd.DataFrame({"a": [0.0, 1.0]}),
            y_val=pd.Series([0, 1]),
        )
        monkeypatch.setitem(_prep_data.PREP_FUNCTIONS, "T1", lambda *a, **k: task_data)
        mock_search = MagicMock(
            return_value=({}, SimpleNamespace(trials=[], best_value=0.5, best_params={}))
        )
        # _run_optuna_tuning lazily re-imports optuna_search at call time, so
        # patching the source module attribute is sufficient (no leak risk).
        monkeypatch.setattr(optuna_hpo, "optuna_search", mock_search)
        _run_optuna_tuning(
            pd.DataFrame(),
            [],
            tmp_path,
            n_trials=n_trials,
            tasks=["T1"],
            models=["xgboost_classifier"],
            storage_dir=tmp_path / "studies",
            inproc=True,
        )
        return mock_search

    def test_resume_passes_only_remaining_trials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("optuna")
        self._seed_study(tmp_path / "studies", n_complete=3)
        mock_search = self._run_tuning(tmp_path, monkeypatch, n_trials=5)
        assert mock_search.call_count == 1
        assert mock_search.call_args.kwargs["n_trials"] == 2

    def test_skip_when_target_reached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("optuna")
        self._seed_study(tmp_path / "studies", n_complete=5)
        mock_search = self._run_tuning(tmp_path, monkeypatch, n_trials=5)
        assert mock_search.call_count == 0

    def test_fresh_study_gets_full_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("optuna")
        mock_search = self._run_tuning(tmp_path, monkeypatch, n_trials=5)
        assert mock_search.call_args.kwargs["n_trials"] == 5
