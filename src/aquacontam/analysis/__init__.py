"""Post-hoc analysis tools — equity, disparity, fairness, spatial diagnostics, and interpretability."""

from aquacontam.analysis.equity import (
    DisparityReport,
    analyze_equity,
    compute_burden_ratio,
    compute_group_metrics,
)
from aquacontam.analysis.regional import (
    analyze_regional_heterogeneity,
    compare_feature_distributions,
    compute_missing_data_patterns,
    compute_region_detection_rates,
)
from aquacontam.analysis.spatial_autocorrelation import (
    MoranResult,
    analyze_spatial_autocorrelation,
    morans_i,
)

__all__ = [
    "DisparityReport",
    "MoranResult",
    "analyze_equity",
    "analyze_regional_heterogeneity",
    "analyze_spatial_autocorrelation",
    "compare_feature_distributions",
    "compute_burden_ratio",
    "compute_group_metrics",
    "compute_missing_data_patterns",
    "compute_region_detection_rates",
    "morans_i",
]
