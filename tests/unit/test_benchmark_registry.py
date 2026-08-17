"""Tests for the benchmark task registry."""

from __future__ import annotations

import pytest

from aquacontam.benchmark.registry import (
    _REGISTRY,
    TaskInfo,
    TaskResult,
    clear_registry,
    get_task,
    list_tasks,
    register_task,
    run_task,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _dummy_task(**kwargs) -> TaskResult:
    """Dummy task function for testing."""
    return TaskResult(
        task_name="test",
        model_name="dummy",
        metrics={"accuracy": 0.9},
    )


class TestRegisterTask:
    """Tests for @register_task decorator."""

    def test_register_and_retrieve(self) -> None:
        @register_task(
            name="T_test",
            description="Test task",
            task_type="classification",
            primary_metric="auprc",
            analytes=["PFOS"],
        )
        def my_task(**kwargs) -> TaskResult:
            return _dummy_task(**kwargs)

        info, _func = get_task("T_test")
        assert isinstance(info, TaskInfo)
        assert info.name == "T_test"
        assert info.task_type == "classification"
        assert info.analytes == ("PFOS",)

    def test_duplicate_registration_raises(self) -> None:
        register_task("T_dup", "desc", "classification", "auprc", ["PFOS"])(_dummy_task)

        with pytest.raises(ValueError, match="already registered"):
            register_task("T_dup", "desc2", "classification", "auprc", ["PFOA"])(_dummy_task)

    def test_decorator_returns_original_function(self) -> None:
        original = _dummy_task
        decorated = register_task("T_orig", "desc", "classification", "auprc", ["PFOS"])(original)
        assert decorated is original


class TestGetTask:
    """Tests for get_task."""

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown task"):
            get_task("nonexistent")

    def test_get_registered_task(self) -> None:
        register_task("T_get", "desc", "classification", "auprc", ["PFOS"])(_dummy_task)
        info, _func = get_task("T_get")
        assert info.name == "T_get"


class TestListTasks:
    """Tests for list_tasks."""

    def test_empty_registry(self) -> None:
        assert list_tasks() == []

    def test_list_sorted_by_name(self) -> None:
        register_task("T_B", "B", "classification", "auprc", ["PFOS"])(_dummy_task)
        register_task("T_A", "A", "classification", "auprc", ["PFOS"])(_dummy_task)

        tasks = list_tasks()
        assert len(tasks) == 2
        assert tasks[0].name == "T_A"
        assert tasks[1].name == "T_B"


class TestRunTask:
    """Tests for run_task."""

    def test_run_returns_result(self) -> None:
        register_task("T_run", "desc", "classification", "auprc", ["PFOS"])(_dummy_task)
        result = run_task("T_run")
        assert isinstance(result, TaskResult)
        assert result.metrics["accuracy"] == 0.9

    def test_run_unknown_task_raises(self) -> None:
        with pytest.raises(KeyError):
            run_task("nonexistent")


class TestClearRegistry:
    """Tests for clear_registry."""

    def test_clears_all_tasks(self) -> None:
        register_task("T_clear", "desc", "classification", "auprc", ["PFOS"])(_dummy_task)
        assert len(list_tasks()) == 1
        clear_registry()
        assert len(list_tasks()) == 0


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_default_fields(self) -> None:
        result = TaskResult(
            task_name="T1",
            model_name="xgboost",
            metrics={"auroc": 0.85},
        )
        assert result.split_metrics == {}
        assert result.metadata == {}

    def test_with_all_fields(self) -> None:
        result = TaskResult(
            task_name="T1",
            model_name="xgboost",
            metrics={"auroc": 0.85},
            split_metrics={"test": {"auroc": 0.85}},
            metadata={"analyte": "PFOS"},
        )
        assert result.split_metrics["test"]["auroc"] == 0.85
        assert result.metadata["analyte"] == "PFOS"
