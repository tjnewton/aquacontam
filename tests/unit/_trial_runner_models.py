"""Test-only model classes for ``test_trial_runner.py``.

These classes are module-level so they're importable from a ``spawn``
multiprocessing child. They sidestep ``BaseModel.evaluate`` by overriding it
directly — the runner only cares about the ``evaluate`` return value.

Not for production use. Lives under ``tests/unit/`` so the wheel doesn't
ship it.
"""

from __future__ import annotations

import os
import time
from typing import Any


class _OkModel:
    """Returns a fixed value from evaluate(); used for the happy path."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def fit(self, X_train: Any, y_train: Any, **kwargs: Any) -> None:
        pass

    def evaluate(self, X_test: Any, y_test: Any, **kwargs: Any) -> dict[str, float]:
        return {"auprc": 0.42}


class _SleepModel:
    """Sleeps in fit() — used to trigger wall-clock timeout."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def fit(self, X_train: Any, y_train: Any, **kwargs: Any) -> None:
        time.sleep(self.config.get("sleep_s", 60))

    def evaluate(self, X_test: Any, y_test: Any, **kwargs: Any) -> dict[str, float]:
        return {"auprc": 0.0}


class _ExceptionModel:
    """Raises ValueError in fit() — used for the exception path."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def fit(self, X_train: Any, y_train: Any, **kwargs: Any) -> None:
        raise ValueError("test")

    def evaluate(self, X_test: Any, y_test: Any, **kwargs: Any) -> dict[str, float]:
        return {"auprc": 0.0}


class _CrashModel:
    """Calls os._exit(2) in fit() — used for the crash path."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def fit(self, X_train: Any, y_train: Any, **kwargs: Any) -> None:
        os._exit(2)

    def evaluate(self, X_test: Any, y_test: Any, **kwargs: Any) -> dict[str, float]:
        return {"auprc": 0.0}


class _EchoEnvModel:
    """Echoes os.environ['TEST_VAR'] as a metric value — env-propagation check."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def fit(self, X_train: Any, y_train: Any, **kwargs: Any) -> None:
        pass

    def evaluate(self, X_test: Any, y_test: Any, **kwargs: Any) -> dict[str, float]:
        return {"auprc": float(os.environ.get("TEST_VAR", "-1"))}
