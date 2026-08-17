"""AquaContam: ML-ready benchmark for drinking water contamination prediction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aquacontam")
except PackageNotFoundError:
    __version__ = "0.0.0"
