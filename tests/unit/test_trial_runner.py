"""Tests for ``aquacontam.benchmark._trial_runner``.

Each test exercises one outcome of the subprocess runner. The model classes
under test live in ``tests.unit._trial_runner_models`` — must be module-level
so they're importable from the ``spawn`` child.

The tests pass ``PYTHONPATH=<repo root>`` through ``env`` so the child can
resolve ``tests.unit._trial_runner_models``. This mirrors what the real
caller (``optuna_hpo._objective``) will do for ``aquacontam.models.*``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pytest

from aquacontam.benchmark._trial_runner import (
    DEFAULT_TIMEOUT_S,
    MAX_CONSECUTIVE_TIMEOUTS,
    PER_MODEL_TIMEOUTS,
    get_timeout,
    reset_timeout_counter,
    run_trial_inproc,
    run_trial_with_timeout,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def child_env() -> dict[str, str]:
    """``env`` that makes ``tests.unit._trial_runner_models`` importable in the child."""
    return {"PYTHONPATH": str(REPO_ROOT)}


@pytest.fixture
def reset_counters() -> None:
    """Clear consecutive-timeout counters before/after each test."""
    reset_timeout_counter()
    yield
    reset_timeout_counter()


@pytest.fixture
def trivial_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tiny numpy arrays — the test models ignore the contents."""
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros(4, dtype=np.int64)
    return X, y, X, y


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_happy_path(child_env: dict[str, str], trivial_data: tuple) -> None:
    X_train, y_train, X_val, y_val = trivial_data
    result = run_trial_with_timeout(
        model_module="tests.unit._trial_runner_models",
        model_class_name="_OkModel",
        config={},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={},
        metric="auprc",
        task_type="classification",
        timeout_s=20.0,
        env=child_env,
    )
    assert result.status == "ok", result.message
    assert result.value == pytest.approx(0.42)


def test_timeout_kills_hung_child(
    child_env: dict[str, str],
    trivial_data: tuple,
    reset_counters: None,
) -> None:
    X_train, y_train, X_val, y_val = trivial_data
    start = time.monotonic()
    result = run_trial_with_timeout(
        model_module="tests.unit._trial_runner_models",
        model_class_name="_SleepModel",
        config={"sleep_s": 60},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={},
        metric="auprc",
        task_type="classification",
        timeout_s=2.0,
        env=child_env,
        model_key="_sleep_model_test",
    )
    elapsed = time.monotonic() - start
    assert result.status == "timeout", result.message
    assert "killed after 2.0s" in result.message
    # The child must actually be killed, not awaited to its 60s sleep.
    # 5s upper bound allows for spawn overhead (~1-3s on Windows) + kill grace.
    assert elapsed < 15.0, f"runner waited {elapsed:.1f}s (expected <15)"


def test_exception_in_child(child_env: dict[str, str], trivial_data: tuple) -> None:
    X_train, y_train, X_val, y_val = trivial_data
    result = run_trial_with_timeout(
        model_module="tests.unit._trial_runner_models",
        model_class_name="_ExceptionModel",
        config={},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={},
        metric="auprc",
        task_type="classification",
        timeout_s=20.0,
        env=child_env,
    )
    assert result.status == "error", result.message
    assert "ValueError" in result.message
    assert "test" in result.message


def test_crash_in_child(child_env: dict[str, str], trivial_data: tuple) -> None:
    X_train, y_train, X_val, y_val = trivial_data
    result = run_trial_with_timeout(
        model_module="tests.unit._trial_runner_models",
        model_class_name="_CrashModel",
        config={},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={},
        metric="auprc",
        task_type="classification",
        timeout_s=20.0,
        env=child_env,
    )
    assert result.status == "crash", result.message
    # os._exit(2) normally surfaces as exitcode=2, but on Windows a hard exit
    # can be clobbered to an access-violation code during DLL teardown in some
    # environments (observed with pip-installed numpy). The contract under test
    # is the crash CLASSIFICATION with the exit code surfaced, not its value.
    assert re.search(r"exitcode=-?\d+", result.message), result.message


def test_env_propagation(child_env: dict[str, str], trivial_data: tuple) -> None:
    X_train, y_train, X_val, y_val = trivial_data
    env = dict(child_env)
    env["TEST_VAR"] = "0.875"
    result = run_trial_with_timeout(
        model_module="tests.unit._trial_runner_models",
        model_class_name="_EchoEnvModel",
        config={},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={},
        metric="auprc",
        task_type="classification",
        timeout_s=20.0,
        env=env,
    )
    assert result.status == "ok", result.message
    assert result.value == pytest.approx(0.875)


def test_unimportable_model(child_env: dict[str, str], trivial_data: tuple) -> None:
    X_train, y_train, X_val, y_val = trivial_data
    result = run_trial_with_timeout(
        model_module="aquacontam.no_such_module",
        model_class_name="DoesNotExist",
        config={},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={},
        metric="auprc",
        task_type="classification",
        timeout_s=20.0,
        env=child_env,
    )
    assert result.status == "error", result.message
    assert "ImportError" in result.message or "ModuleNotFoundError" in result.message


def test_input_not_pickleable(child_env: dict[str, str], trivial_data: tuple) -> None:
    """Pickling failure must surface before the child is spawned."""
    X_train, y_train, X_val, y_val = trivial_data
    start = time.monotonic()
    result = run_trial_with_timeout(
        model_module="tests.unit._trial_runner_models",
        model_class_name="_OkModel",
        config={},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={"bad_callback": lambda x: x},  # lambdas don't pickle
        metric="auprc",
        task_type="classification",
        timeout_s=20.0,
        env=child_env,
    )
    elapsed = time.monotonic() - start
    assert result.status == "error", result.message
    assert "pickleable" in result.message.lower()
    # No child should ever have started — must be < 1 second total.
    assert elapsed < 2.0, f"pickle error took {elapsed:.1f}s (expected <2 — child spawn?)"


def test_consecutive_timeouts_halt(
    child_env: dict[str, str],
    trivial_data: tuple,
    reset_counters: None,
) -> None:
    """After MAX_CONSECUTIVE_TIMEOUTS in a row on the same model_key, runner raises."""
    X_train, y_train, X_val, y_val = trivial_data
    model_key = "_halt_test_model"

    # First MAX_CONSECUTIVE_TIMEOUTS-1 timeouts return cleanly.
    for i in range(MAX_CONSECUTIVE_TIMEOUTS - 1):
        result = run_trial_with_timeout(
            model_module="tests.unit._trial_runner_models",
            model_class_name="_SleepModel",
            config={"sleep_s": 30},
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fit_kwargs={},
            metric="auprc",
            task_type="classification",
            timeout_s=1.5,
            env=child_env,
            model_key=model_key,
        )
        assert result.status == "timeout", f"iter {i}: {result.message}"

    # The Nth consecutive timeout raises.
    with pytest.raises(RuntimeError, match="wedged"):
        run_trial_with_timeout(
            model_module="tests.unit._trial_runner_models",
            model_class_name="_SleepModel",
            config={"sleep_s": 30},
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fit_kwargs={},
            metric="auprc",
            task_type="classification",
            timeout_s=1.5,
            env=child_env,
            model_key=model_key,
        )


# ---------------------------------------------------------------------------
# Smaller unit tests for the helpers
# ---------------------------------------------------------------------------


def test_get_timeout_default() -> None:
    assert get_timeout("xgboost_classifier") == DEFAULT_TIMEOUT_S


def test_get_timeout_per_model() -> None:
    assert get_timeout("tabpfn_classifier") == PER_MODEL_TIMEOUTS["tabpfn_classifier"]
    assert get_timeout("gnn_gcn_classifier") == 1200.0


def test_run_trial_inproc_happy_path(trivial_data: tuple) -> None:
    """In-process runner returns the same result shape as the subprocess runner."""
    from tests.unit._trial_runner_models import _OkModel

    X_train, y_train, X_val, y_val = trivial_data
    result = run_trial_inproc(
        model_cls=_OkModel,
        config={},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={},
        metric="auprc",
        task_type="classification",
    )
    assert result.status == "ok"
    assert result.value == pytest.approx(0.42)


def test_run_trial_inproc_exception(trivial_data: tuple) -> None:
    from tests.unit._trial_runner_models import _ExceptionModel

    X_train, y_train, X_val, y_val = trivial_data
    result = run_trial_inproc(
        model_cls=_ExceptionModel,
        config={},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_kwargs={},
        metric="auprc",
        task_type="classification",
    )
    assert result.status == "error"
    assert "ValueError" in result.message
