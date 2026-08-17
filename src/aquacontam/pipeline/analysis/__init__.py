"""Analysis sub-package for the reproducibility pipeline.

Re-exports every ``_run_*`` function so existing imports such as::

    from aquacontam.pipeline.analysis import _run_loro_cv

continue to work unchanged.
"""

from __future__ import annotations

from aquacontam.pipeline.analysis._equity_transfer import (
    _run_causal_deconfounding,
    _run_equity_analysis,
    _run_external_validation,
    _run_group_conformal_analysis,
    _run_lift_analysis,
    _run_loro_equity_analysis,
    _run_monitoring_inequity,
    _run_transfer_ablation,
)
from aquacontam.pipeline.analysis._evaluation import (
    _run_bootstrap_cis,
    _run_calibration_analysis,
    _run_conformal_analysis,
    _run_detection_only_ablation,
    _run_model_comparison,
    _run_power_analysis,
    _run_seed_inflation_check,
    _run_sensitivity_bounds,
    _run_shap_analysis,
    _run_shap_interaction_analysis,
    _run_spatial_autocorrelation,
)
from aquacontam.pipeline.analysis._robustness import (
    _run_buffered_loro,
    _run_cnn1d_feature_ordering,
    _run_coordinate_sensitivity,
    _run_dedup_comparison,
    _run_feature_ablation,
    _run_icp_recoverability,
    _run_loro_cv,
    _run_mcl_exceedance_analysis,
    _run_multi_seed_analysis,
    _run_region_heterogeneity,
    _run_sensitivity_analysis,
    _run_spatial_block_bootstrap,
    _run_split_comparison,
    _run_val_reuse_bias,
)
from aquacontam.pipeline.analysis._tuning import (
    _run_fair_tuning,
    _run_hyperparam_sensitivity,
    _run_optuna_tuning,
)

__all__ = [
    "_run_bootstrap_cis",
    "_run_buffered_loro",
    "_run_calibration_analysis",
    "_run_causal_deconfounding",
    "_run_cnn1d_feature_ordering",
    "_run_conformal_analysis",
    "_run_coordinate_sensitivity",
    "_run_dedup_comparison",
    "_run_detection_only_ablation",
    "_run_equity_analysis",
    "_run_external_validation",
    "_run_fair_tuning",
    "_run_feature_ablation",
    "_run_group_conformal_analysis",
    "_run_hyperparam_sensitivity",
    "_run_icp_recoverability",
    "_run_lift_analysis",
    "_run_loro_cv",
    "_run_loro_equity_analysis",
    "_run_mcl_exceedance_analysis",
    "_run_model_comparison",
    "_run_monitoring_inequity",
    "_run_multi_seed_analysis",
    "_run_optuna_tuning",
    "_run_power_analysis",
    "_run_region_heterogeneity",
    "_run_seed_inflation_check",
    "_run_sensitivity_analysis",
    "_run_sensitivity_bounds",
    "_run_shap_analysis",
    "_run_shap_interaction_analysis",
    "_run_spatial_autocorrelation",
    "_run_spatial_block_bootstrap",
    "_run_split_comparison",
    "_run_transfer_ablation",
    "_run_val_reuse_bias",
]
