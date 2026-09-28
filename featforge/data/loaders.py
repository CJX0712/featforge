"""Loaders: CSV / NPY / JSON -> Dataset.

CSV/JSON go through pandas only when available; NPY is pure numpy. A pure-numpy
CSV fallback keeps the loader usable with zero heavy dependencies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.errors import DataError
from ..core.types import Dataset


def _split_xy(X: np.ndarray, y_col: Optional[int], has_header: bool):
    if y_col is None:
        return X, None
    if y_col < 0:
        y_col = X.shape[1] + y_col
    if not (0 <= y_col < X.shape[1]):
        raise DataError(f"y_col {y_col} out of range for {X.shape[1]} columns")
    y = X[:, y_col]
    X2 = np.delete(X, y_col, axis=1)
    return X2, y


def from_npy(path: str, y_col: Optional[int] = -1, task: str = "classification") -> Dataset:
    try:
        arr = np.load(path)
    except Exception as e:  # pragma: no cover - defensive
        raise DataError(f"failed to load npy {path}: {e}") from e
    if arr.ndim != 2:
        raise DataError("npy array must be 2-D (n_samples, n_features[+label])")
    X, y = _split_xy(arr, y_col, False)
    return Dataset(X, y, task=task)


def from_csv(path: str, y_col: Optional[int] = -1, task: str = "classification") -> Dataset:
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(path)
        arr = df.to_numpy(dtype=np.float64)
        feature_names = [str(c) for c in df.columns]
    except Exception:
        # pure-numpy fallback: minimal CSV parser (comma-separated, numeric)
        arr, feature_names = _numpy_csv(path)
    if arr.size == 0:
        raise DataError(f"empty CSV {path}")
    X, y = _split_xy(arr, y_col, False)
    if feature_names:
        if y is not None and y_col is not None:
            idx = y_col if y_col >= 0 else arr.shape[1] + y_col
            feature_names = [n for i, n in enumerate(feature_names) if i != idx]
    return Dataset(X, y, feature_names or None, task=task)


def _numpy_csv(path: str):
    rows = []
    names: list = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if not lines:
        return np.empty((0, 0)), names
    # assume first line header if it contains non-numeric tokens
    first = lines[0].split(",")
    try:
        [float(t) for t in first]
        data_lines = lines
    except ValueError:
        names = first
        data_lines = lines[1:]
    for ln in data_lines:
        tok = ln.split(",")
        try:
            rows.append([float(t) for t in tok])
        except ValueError:
            continue
    return np.asarray(rows, dtype=np.float64), names


def from_json(path: str, task: str = "classification") -> Dataset:
    import json

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "X" not in obj:
        raise DataError("json must be an object with an 'X' key")
    X = np.asarray(obj["X"], dtype=np.float64)
    y = np.asarray(obj["y"], dtype=np.float64) if obj.get("y") is not None else None
    names = obj.get("feature_names")
    return Dataset(X, y, names, task=task)
