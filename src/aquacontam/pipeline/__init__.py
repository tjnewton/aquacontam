"""AquaContam reproducibility pipeline package.

Provides the building blocks for the full reproducibility pipeline
(``scripts/reproduce.py``).  Core stages are re-exported here for
convenience::

    from aquacontam.pipeline import download_data, preprocess_data
"""

from __future__ import annotations

from aquacontam.pipeline._strict import is_strict, set_strict
from aquacontam.pipeline.assembly import assemble_with_split_imputation
from aquacontam.pipeline.checksum import generate_results_checksums, verify_results_checksums
from aquacontam.pipeline.download import download_data
from aquacontam.pipeline.export import (
    export_feature_importances,
    export_icp_diagnostics,
    export_results,
    export_zenodo_dataset,
    generate_paper_assets,
    generate_webapp_predictions,
)
from aquacontam.pipeline.features import extract_features
from aquacontam.pipeline.models import get_model_instances, load_model_configs
from aquacontam.pipeline.preprocess import preprocess_data
from aquacontam.pipeline.training import build_task_kwargs, train_and_evaluate
from aquacontam.pipeline.utils import (
    clean_caches,
    collect_software_versions,
    print_results_summary,
    save_software_versions,
)

__all__ = [
    "assemble_with_split_imputation",
    "build_task_kwargs",
    "clean_caches",
    "collect_software_versions",
    "download_data",
    "export_feature_importances",
    "export_icp_diagnostics",
    "export_results",
    "export_zenodo_dataset",
    "extract_features",
    "generate_paper_assets",
    "generate_results_checksums",
    "generate_webapp_predictions",
    "get_model_instances",
    "is_strict",
    "load_model_configs",
    "preprocess_data",
    "print_results_summary",
    "save_software_versions",
    "set_strict",
    "train_and_evaluate",
    "verify_results_checksums",
]
