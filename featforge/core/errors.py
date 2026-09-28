"""FeatForge error taxonomy (E100-E500), one family per subsystem.

Every subsystem raises the matching family so the pipeline can branch on cause
instead of string-matching messages.
"""
from __future__ import annotations


class ForgeError(Exception):
    """Base class for all FeatForge errors."""


class ConfigError(ForgeError):
    """E100 — invalid configuration / environment."""


class DataError(ForgeError):
    """E200 — malformed input data or unsupported loader."""


class SynthError(ForgeError):
    """E300 — feature synthesis failure (bad transform / domain guard)."""


class SelectError(ForgeError):
    """E400 — feature selection failure (empty pool / degenerate CV)."""


class EvalError(ForgeError):
    """E500 — downstream evaluation failure (metric / model error)."""
