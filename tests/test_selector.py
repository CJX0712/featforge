import numpy as np
import pytest

from featforge.core.errors import SelectError
from featforge.data.synthetic import interaction_dataset
from featforge.select import FeatForgeSelect


def _pair():
    ds = interaction_dataset(n=200, seed=5)
    # manual small pool: raw features + the true interaction + a duplicate
    X = np.column_stack([ds.X, ds.X[:, 0] * ds.X[:, 1], ds.X[:, 0] * 1.0])
    names = list(ds.feature_names) + ["x0*x1", "x0_dup"]
    return X, ds.y, names


def test_returns_at_most_select_k():
    X, y, names = _pair()
    res = FeatForgeSelect(random_state=42, select_k=3, select_prefilter=8).select(X, y, names)
    assert len(res.chosen_names) <= 3
    assert len(res.chosen_indices) == len(res.chosen_names)
    assert res.metadata["pruned"] is True


def test_forward_gains_monotone_nondecreasing():
    """Greedy forward selection only accepts improvements -> trace must rise."""
    X, y, names = _pair()
    res = FeatForgeSelect(random_state=42, select_k=3, select_prefilter=8).select(X, y, names)
    gains = res.forward_gains
    assert len(gains) >= 1
    diffs = np.diff(np.asarray(gains))
    assert np.all(diffs > 0.0)


def test_small_pool_not_pruned():
    X, y, names = _pair()
    res = FeatForgeSelect(random_state=42, select_k=99).select(X, y, names)
    assert res.metadata["pruned"] is False
    assert res.chosen_names == names


def test_contains_true_interaction():
    X, y, names = _pair()
    res = FeatForgeSelect(random_state=42, select_k=3, select_prefilter=8).select(X, y, names)
    # the interaction column carries the signal the raw features miss
    assert "x0*x1" in res.chosen_names


def test_no_duplicate_selection():
    X, y, names = _pair()
    res = FeatForgeSelect(random_state=42, select_k=4, select_prefilter=8).select(X, y, names)
    assert len(set(res.chosen_indices)) == len(res.chosen_indices)


def test_requires_target_and_valid_names():
    X, y, names = _pair()
    sel = FeatForgeSelect()
    with pytest.raises(SelectError):
        sel.select(X, y, ["wrong"])
    with pytest.raises(SelectError):
        sel.select(X, None, names)
    with pytest.raises(SelectError):
        sel.select(np.empty((10, 0)), y, [])


def test_relevance_map_covers_pool():
    X, y, names = _pair()
    res = FeatForgeSelect(random_state=42, select_k=2, select_prefilter=6).select(X, y, names)
    assert set(res.relevance.keys()) == set(names)
    assert all(v >= 0.0 for v in res.relevance.values())
