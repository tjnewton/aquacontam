"""Benchmark task definitions (T1-T7), metrics, splits, and leaderboard."""

# Import tasks to trigger @register_task decorators
import aquacontam.benchmark.multilabel as _multilabel  # noqa: F401
import aquacontam.benchmark.private_wells as _private_wells  # noqa: F401
import aquacontam.benchmark.tasks as _tasks  # noqa: F401
import aquacontam.benchmark.temporal as _temporal  # noqa: F401
import aquacontam.benchmark.transfer as _transfer  # noqa: F401
from aquacontam.benchmark.metrics import (
    compute_censoring_aware_metrics,
    compute_classification_metrics,
    compute_multilabel_metrics,
    compute_regression_metrics,
)
from aquacontam.benchmark.registry import (
    TaskInfo,
    TaskResult,
    clear_registry,
    get_task,
    list_tasks,
    register_task,
    run_all_tasks,
    run_task,
)

__all__ = [
    "TaskInfo",
    "TaskResult",
    "clear_registry",
    "compute_censoring_aware_metrics",
    "compute_classification_metrics",
    "compute_multilabel_metrics",
    "compute_regression_metrics",
    "get_task",
    "list_tasks",
    "register_task",
    "run_all_tasks",
    "run_task",
]
