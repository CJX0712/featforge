import numpy as np
import pytest

from featforge.core.errors import SynthError
from featforge.data.synthetic import interaction_dataset
from featforge.synth import FeatForgeSynth, get_synthesizer, list_synthesizers


def _data():
    return interaction_dataset(n=200, seed=5)


def test_frame_keeps_raw_columns_and_names_align():
    ds = _data()
    ff = FeatForgeSynth(random_state=42).fit_transform(ds.X, ds.y, ds.feature_names)
    assert ff.X.shape[1] == len(ff.feature_names)
    for nm in ds.feature_names:
        assert nm in ff.feature_names
        assert ff.source[nm] == "raw"


def test_all_columns_finite():
    ds = _data()
    ff = FeatForgeSynth(random_state=42).fit_transform(ds.X, ds.y, ds.feature_names)
    assert np.all(np.isfinite(ff.X))


def test_respects_synth_top_k_cap():
    ds = _data()
    top_k = 8
    synth = FeatForgeSynth(random_state=42, synth_top_k=top_k)
    ff = synth.fit_transform(ds.X, ds.y, ds.feature_names)
    n_synth = ff.X.shape[1] - ds.n_features
    assert n_synth <= top_k
    assert synth.metadata["n_synth"] == n_synth


def test_interaction_feature_is_materialised():
    """The true generative interaction x0*x1 must appear in the pool."""
    ds = _data()
    ff = FeatForgeSynth(random_state=42).fit_transform(ds.X, ds.y, ds.feature_names)
    assert "x0*x1" in ff.feature_names
    assert ff.source["x0*x1"] == "synth"


def test_transform_names_present_with_generous_budget():
    """With headroom, every transform family should materialise for each column."""
    ds = _data()
    ff = FeatForgeSynth(random_state=42, synth_top_k=100).fit_transform(
        ds.X, ds.y, ds.feature_names)
    names = set(ff.feature_names)
    for fam in ("x0^2", "|x0|", "log1p|x0|", "sqrt|x0|", "sigmoid(x0)", "inv|x0|"):
        assert fam in names


def test_single_feature_dataset_has_no_interactions():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 1))
    y = (X[:, 0] > 0).astype(int)
    ff = FeatForgeSynth(random_state=42).fit_transform(X, y, ["x0"])
    assert ff.X.shape[1] > 1  # transforms added
    assert "x0*x1" not in "".join(ff.feature_names)


def test_rejects_bad_names():
    ds = _data()
    with pytest.raises(SynthError):
        FeatForgeSynth().fit_transform(ds.X, ds.y, ["a", "b"])


def test_works_without_target():
    ds = _data()
    ff = FeatForgeSynth(random_state=42).fit_transform(ds.X, None, ds.feature_names)
    assert np.all(np.isfinite(ff.X))


def test_registry_lookup():
    assert "featforge_synth" in list_synthesizers()
    s = get_synthesizer("featforge_synth", random_state=1)
    assert isinstance(s, FeatForgeSynth)
    with pytest.raises(SynthError):
        get_synthesizer("nope")
