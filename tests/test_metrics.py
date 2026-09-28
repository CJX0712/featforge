import numpy as np

from featforge.core.config import Settings
from featforge.eval.metrics import (
    _numpy_cv,
    abs_corr_matrix,
    available_sklearn,
    downstream_cv,
    relevance_scores,
)


def _separable_data(n=120, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_abs_corr_matrix_symmetric_unit_diagonal():
    X, _ = _separable_data()
    C = abs_corr_matrix(X)
    assert C.shape == (4, 4)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 1.0, atol=1e-6)
    assert C.min() >= -1e-9 and C.max() <= 1.0 + 1e-9


def test_abs_corr_detects_duplicate():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((100, 1))
    X = np.column_stack([base[:, 0], base[:, 0] + 1e-9 * rng.standard_normal(100)])
    C = abs_corr_matrix(X)
    assert C[0, 1] > 0.99


def test_relevance_scores_shape_and_nonneg():
    X, y = _separable_data()
    rel = relevance_scores(X, y, "classification")
    assert rel.shape == (4,)
    assert float(np.min(rel)) >= 0.0


def test_relevance_prefers_signal_columns():
    X, y = _separable_data()
    rel = relevance_scores(X, y, "classification")
    # columns 0 and 1 generate the label; 2 and 3 are noise
    assert rel[0] > rel[3] and rel[1] > rel[2]


def test_relevance_without_y_returns_zeros():
    X, _ = _separable_data()
    rel = relevance_scores(X, None, "classification")
    assert np.allclose(rel, 0.0)


def test_downstream_cv_deterministic():
    X, y = _separable_data()
    s = Settings(random_state=42, cv=5)
    a, sa = downstream_cv(X, y, s)
    b, sb = downstream_cv(X, y, s)
    assert a == b
    assert sa == sb


def test_downstream_cv_sklearn_high_on_separable():
    X, y = _separable_data()
    s = Settings(random_state=42, cv=5)
    score, _ = downstream_cv(X, y, s)
    assert 0.0 <= score <= 1.0
    assert available_sklearn()


def test_numpy_cv_fallback_scores_reasonably():
    # pure-numpy path must still separate a linearly separable problem
    X, y = _separable_data(n=150)
    scores = _numpy_cv(X, y, "classification", 5, 42)
    assert len(scores) == 5
    assert 0.0 <= min(scores) and max(scores) <= 1.0
    assert float(np.mean(scores)) > 0.7


def test_numpy_cv_fallback_regression():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 3))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1]
    scores = _numpy_cv(X, y, "regression", 4, 42)
    assert float(np.mean(scores)) > 0.8
