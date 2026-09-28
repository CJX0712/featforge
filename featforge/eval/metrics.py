"""Downstream scoring used by selectors and the benchmark table.

The SAME model scaffold (scaler + estimator) is used everywhere, so a gain
reflects feature quality rather than a modelling difference. scikit-learn is the
SOTA backend; a deterministic pure-numpy cross-validated model is the offline
fallback (see `available_sklearn`).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.config import Settings


def available_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def _sanitize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _folds(n: int, cv: int, seed: int) -> List[np.ndarray]:
    """Deterministic shuffled k-fold index assignment."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, cv)


def _standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def _numpy_logreg_score(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                        yte: np.ndarray, n_iter: int = 300, lr: float = 0.1) -> float:
    Xtr, Xte = _standardize(Xtr, Xte)
    n, d = Xtr.shape
    Xb = np.column_stack([np.ones(n), Xtr])
    w = np.zeros(d + 1)
    y01 = (ytr > 0).astype(float)
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y01) / n
        w -= lr * g
    Xte1 = np.column_stack([np.ones(Xte.shape[0]), Xte])
    pr = 1.0 / (1.0 + np.exp(-np.clip(Xte1 @ w, -30, 30)))
    pred = (pr > 0.5).astype(int)
    return float(np.mean(pred == (yte > 0).astype(int)))


def _numpy_linreg_score(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                        yte: np.ndarray) -> float:
    Xtr, Xte = _standardize(Xtr, Xte)
    Xb = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
    beta, *_ = np.linalg.lstsq(Xb, ytr, rcond=None)
    pred = np.column_stack([np.ones(Xte.shape[0]), Xte]) @ beta
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _numpy_cv(X: np.ndarray, y: np.ndarray, task: str, cv: int, seed: int) -> List[float]:
    X = _sanitize(X)
    n = X.shape[0]
    cv = max(2, min(cv, n))
    parts = _folds(n, cv, seed)
    fold_of = np.empty(n, dtype=int)
    for k, p in enumerate(parts):
        fold_of[p] = k
    scores: List[float] = []
    for k in range(len(parts)):
        te = np.where(fold_of == k)[0]
        tr = np.where(fold_of != k)[0]
        if tr.size == 0 or te.size == 0:
            continue
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        if task == "classification":
            scores.append(_numpy_logreg_score(Xtr, ytr, Xte, yte))
        else:
            scores.append(_numpy_linreg_score(Xtr, ytr, Xte, yte))
    return scores


def abs_corr_matrix(X: np.ndarray) -> np.ndarray:
    """|Pearson correlation| matrix between columns (pure numpy, no sklearn)."""
    X = _sanitize(X)
    if X.shape[1] == 0:
        return np.zeros((0, 0))
    Xd = X - X.mean(axis=0)
    num = np.abs(Xd.T @ Xd)
    norms = np.sqrt((Xd ** 2).sum(axis=0))
    den = np.outer(norms, norms) + 1e-12
    return num / (den + 1e-12)


def _abs_corr_with_y(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = _sanitize(X)
    y = np.asarray(y, dtype=np.float64)
    Xd = X - X.mean(axis=0)
    yd = y - y.mean()
    num = np.abs(Xd.T @ yd)
    den = np.sqrt(((Xd ** 2).sum(axis=0)) * ((yd ** 2).sum()) + 1e-12)
    return num / (den + 1e-12)


def relevance_scores(X: np.ndarray, y, task: str = "classification",
                     seed: int = 42) -> np.ndarray:
    """Per-column relevance to the target.

    Mutual information via scikit-learn when available (true ML-relevance, not
    just linear correlation); degrades to |Pearson corr| offline.
    """
    X = _sanitize(np.asarray(X, dtype=np.float64))
    n_cols = X.shape[1]
    if n_cols == 0:
        return np.zeros(0)
    if y is None:
        return np.zeros(n_cols)
    y = np.asarray(y)
    if task == "classification":
        try:
            from sklearn.feature_selection import mutual_info_classif as _sk_mic

            rel = _sk_mic(X, y, random_state=seed)
            return np.asarray(rel, dtype=np.float64)
        except Exception:
            return _abs_corr_with_y(X, y)
    try:
        from sklearn.feature_selection import mutual_info_regression as _sk_mir

        rel = _sk_mir(X, y, random_state=seed)
        return np.asarray(rel, dtype=np.float64)
    except Exception:
        return _abs_corr_with_y(X, y)


def downstream_cv(
    X: np.ndarray, y: np.ndarray, settings: Settings
) -> Tuple[float, List[float]]:
    """Cross-validated downstream score for one feature matrix.

    Deterministic for a given `settings.random_state` on both the sklearn and
    the numpy fallback paths.
    """
    X = _sanitize(np.asarray(X, dtype=np.float64))
    y = np.asarray(y)
    n = X.shape[0]
    cv = max(2, min(settings.cv, n))

    if available_sklearn():
        from sklearn.model_selection import cross_val_score as _sk_cvs
        from sklearn.pipeline import Pipeline as _SkPipeline
        from sklearn.preprocessing import StandardScaler as _SkScaler

        if settings.task == "classification":
            from sklearn.linear_model import LogisticRegression as _SkLR
            est = _SkLR(max_iter=2000, random_state=settings.random_state)
            scoring = "accuracy"
        else:
            from sklearn.linear_model import LinearRegression as _SkLin
            est = _SkLin()
            scoring = "r2"
        pipe = _SkPipeline([("scaler", _SkScaler()), ("model", est)])
        scores = _sk_cvs(pipe, X, y, cv=cv, scoring=scoring)
        return float(np.mean(scores)), [float(s) for s in scores]

    scores = _numpy_cv(X, y, settings.task, cv, settings.random_state)
    if not scores:
        return 0.0, []
    return float(np.mean(scores)), scores
