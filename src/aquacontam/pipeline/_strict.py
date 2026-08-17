"""Strict mode flag for pipeline error handling."""

from __future__ import annotations

_strict: bool = False


def set_strict(value: bool) -> None:
    global _strict
    _strict = value


def is_strict() -> bool:
    return _strict
