"""Core data types for FeatForge (AutoML feature engineering).

Immutable-friendly dataclasses shared across every module so the pipeline can
compare synthesizers / selectors fairly under one contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Dataset:
    """A feature matrix plus optional ground-truth labels.

    `task` is 'classification' or 'regression'. `feature_names` aligns 1:1 with
    `X` columns. All downstream gain is measured against this labelled frame.
    """

    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    task: str = "classification"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got {self.X.ndim}-D")
        n, d = self.X.shape
        if self.y is not None:
            self.y = np.asarray(self.y)
            if self.y.shape[0] != n:
                raise ValueError("X and y must share the same number of rows")
        if self.feature_names is None:
            self.feature_names = [f"x{i}" for i in range(d)]
        else:
            if len(self.feature_names) != d:
                raise ValueError("feature_names must match X columns")

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@dataclass
class FeatureFrame:
    """A matrix of candidate features (raw + synthesized) with provenance."""

    X: np.ndarray
    feature_names: List[str]
    source: Dict[str, str] = field(default_factory=dict)  # name -> 'raw' | 'synth'

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names must match X columns")


@dataclass
class Result:
    """Outcome of one method on one dataset (baseline / synth / selected)."""

    name: str
    feature_names: List[str]
    n_features: int
    downstream_score: float
    cv_scores: List[float] = field(default_factory=list)
    runtime_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
