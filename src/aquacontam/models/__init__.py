"""Baseline model implementations (XGBoost, RF, LightGBM, MLP, CNN)."""

from aquacontam.models.base import BaseModel
from aquacontam.models.hurdle import HurdleRegressor
from aquacontam.models.random_forest import RandomForestClassifier, RandomForestRegressor

try:
    from aquacontam.models.xgboost import XGBoostClassifier, XGBoostRegressor

    _has_xgboost = True
except (ImportError, OSError, RuntimeError):
    _has_xgboost = False  # XGBoost native library unavailable (e.g., missing libomp)

try:
    from aquacontam.models.xgboost_aft import XGBoostAFTRegressor

    _has_xgboost_aft = True
except (ImportError, OSError, RuntimeError):
    _has_xgboost_aft = False

try:
    from aquacontam.models.lightgbm import LightGBMClassifier, LightGBMRegressor

    _has_lightgbm = True
except (ImportError, OSError, RuntimeError):
    _has_lightgbm = False  # LightGBM unavailable

try:
    from aquacontam.models.ensemble import StackingEnsembleClassifier, VotingEnsembleClassifier

    _has_ensemble = True
except (ImportError, OSError, RuntimeError):
    _has_ensemble = False

__all__ = [
    "BaseModel",
    "HurdleRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
]

if _has_xgboost:
    __all__ += ["XGBoostClassifier", "XGBoostRegressor"]

if _has_xgboost_aft:
    __all__ += ["XGBoostAFTRegressor"]

if _has_lightgbm:
    __all__ += ["LightGBMClassifier", "LightGBMRegressor"]

if _has_ensemble:
    __all__ += ["StackingEnsembleClassifier", "VotingEnsembleClassifier"]

# Lazy-load torch-dependent models (MLP, CNN1D) to avoid importing torch
# at module level. This prevents NumPy/torch ABI mismatch warnings when
# only non-torch models (xgboost, random_forest) are requested.

_LAZY_TORCH_MODELS: dict[str, tuple[str, str]] = {
    "MLPClassifier": ("aquacontam.models.mlp", "MLPClassifier"),
    "MLPRegressor": ("aquacontam.models.mlp", "MLPRegressor"),
    "CNN1DClassifier": ("aquacontam.models.cnn1d", "CNN1DClassifier"),
    "CNN1DRegressor": ("aquacontam.models.cnn1d", "CNN1DRegressor"),
    "GNNClassifier": ("aquacontam.models.gnn", "GNNClassifier"),
    "DeepTobitClassifier": ("aquacontam.models.deep_tobit", "DeepTobitClassifier"),
    "DeepTobitRegressor": ("aquacontam.models.deep_tobit", "DeepTobitRegressor"),
    "ZeroInflatedTobitClassifier": (
        "aquacontam.models.zero_inflated_tobit",
        "ZeroInflatedTobitClassifier",
    ),
    "ZeroInflatedTobitRegressor": (
        "aquacontam.models.zero_inflated_tobit",
        "ZeroInflatedTobitRegressor",
    ),
    "ICPClassifier": ("aquacontam.models.icp", "ICPClassifier"),
    "ICPRegressor": ("aquacontam.models.icp", "ICPRegressor"),
}


def __getattr__(name: str) -> type:
    if name in _LAZY_TORCH_MODELS:
        module_path, cls_name = _LAZY_TORCH_MODELS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)  # type: ignore[no-any-return]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
