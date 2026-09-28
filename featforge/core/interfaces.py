"""Protocol contracts for synthesizers and selectors.

`pipeline` depends only on these interfaces, never on concrete classes, so the
direction of calls stays `pipeline -> {data, synth, select, eval} -> core`.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseSynthesizer(Protocol):
    name: str

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "object":  # returns FeatureFrame
        ...


@runtime_checkable
class BaseSelector(Protocol):
    name: str

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> "object":  # returns SelectionResult
        ...
