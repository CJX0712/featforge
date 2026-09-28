"""Synthetic tabular datasets engineered so feature synthesis earns its keep.

Every generator bakes in an effect a *linear* raw-feature model cannot capture
(interaction, nonlinearity, or redundancy), which is exactly where FeatForge's
synthesizer + selector should show downstream gain.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.types import Dataset


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def interaction_dataset(
    n: int = 600, d: int = 6, noise: float = 0.15, seed: int = 42
) -> Dataset:
    """y depends on the product x0*x1 plus a small linear term.

    A raw linear model leaves the interaction on the table; the product feature
    x0*x1 recovers it.
    """
    rng = _rng(seed)
    X = rng.standard_normal((n, d))
    score = X[:, 0] * X[:, 1] + 0.3 * X[:, 2]
    y = (score + noise * rng.standard_normal(n) > 0).astype(int)
    return Dataset(X, y, [f"x{i}" for i in range(d)], task="classification")


def nonlinear_dataset(n: int = 600, d: int = 5, noise: float = 0.15, seed: int = 7) -> Dataset:
    """y depends on sin(x0) and a quadratic x1^2 — needs nonlinear transforms."""
    rng = _rng(seed)
    X = rng.standard_normal((n, d))
    score = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    y = (score + noise * rng.standard_normal(n) > 0).astype(int)
    return Dataset(X, y, [f"x{i}" for i in range(d)], task="classification")


def xor_dataset(n: int = 500, d: int = 4, noise: float = 0.05, seed: int = 3) -> Dataset:
    """y = XOR of the signs of x0 and x1 (pure interaction, no marginal signal)."""
    rng = _rng(seed)
    X = rng.standard_normal((n, d))
    s0 = (X[:, 0] > 0).astype(int)
    s1 = (X[:, 1] > 0).astype(int)
    y = (s0 ^ s1).astype(float) + noise * rng.standard_normal(n)
    y = (y > 0.5).astype(int)
    return Dataset(X, y, [f"x{i}" for i in range(d)], task="classification")


def redundant_dataset(
    n: int = 500, d: int = 10, noise: float = 0.05, seed: int = 11
) -> Dataset:
    """5 signal columns + 5 noisy near-duplicates: a redundancy-pruning stress test."""
    rng = _rng(seed)
    X = rng.standard_normal((n, d))
    Xc = np.column_stack([X, X[:, :5] + 0.1 * rng.standard_normal((n, 5))])
    score = X[:, 0] + X[:, 1]
    y = (score + noise * rng.standard_normal(n) > 0).astype(int)
    names = [f"x{i}" for i in range(d)] + [f"x{i}_dup" for i in range(5)]
    return Dataset(Xc, y, names, task="classification")


def friedman_dataset(n: int = 600, noise: float = 1.0, seed: int = 21) -> Dataset:
    """Friedman #1: sin(pi*x0*x1) + 20(x2-0.5)^2 + 10x3 + 5x4 — interactions + nonlinear."""
    rng = _rng(seed)
    X = rng.uniform(0, 1, (n, 5))
    score = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    y = (score + noise * rng.standard_normal(n) > np.median(score)).astype(int)
    return Dataset(X, y, [f"x{i}" for i in range(5)], task="classification")


def default_benchmark_set() -> Dict[str, Dataset]:
    """Curated difficulty gradient: 5 datasets, each stresses a different aspect."""
    return {
        "interaction": interaction_dataset(),
        "nonlinear": nonlinear_dataset(),
        "xor": xor_dataset(),
        "redundant": redundant_dataset(),
        "friedman": friedman_dataset(),
    }
