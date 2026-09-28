from featforge.core.config import Settings
from featforge.data.synthetic import (
    default_benchmark_set,
    interaction_dataset,
)
from featforge.pipeline import METHODS, FeatPipeline


def test_evaluate_has_all_methods():
    ds = interaction_dataset(n=150, seed=5)
    pipe = FeatPipeline(select_kw={"select_k": 3})
    out = pipe.evaluate(ds)
    for m in METHODS:
        assert m in out
        assert 0.0 <= out[m]["downstream_score"] <= 1.0
        assert out[m]["n_features"] > 0


def test_selected_is_more_compact_than_synth_all():
    ds = interaction_dataset(n=150, seed=5)
    out = FeatPipeline(select_kw={"select_k": 3}).evaluate(ds)
    assert out["selected"]["n_features"] <= out["synth_all"]["n_features"]


def test_flagship_beats_raw_on_interaction():
    """Headline claim: synthesis + selection must beat the raw baseline."""
    ds = interaction_dataset(n=300, seed=42)
    out = FeatPipeline(select_kw={"select_k": 4}).evaluate(ds)
    assert out["selected"]["downstream_score"] > out["raw"]["downstream_score"]
    assert out["synth_all"]["downstream_score"] > out["raw"]["downstream_score"]


def test_selected_feature_count_within_budget():
    ds = interaction_dataset(n=150, seed=5)
    out = FeatPipeline(select_kw={"select_k": 3}).evaluate(ds)
    assert out["selected"]["n_features"] <= 3


def test_benchmark_aggregates_all_methods():
    pipe = FeatPipeline(synth_kw={"synth_top_k": 10}, select_kw={"select_k": 3})
    report = pipe.benchmark({"interaction": interaction_dataset(n=120, seed=5)})
    agg = report["aggregate"]
    for m in METHODS:
        assert m in agg
        assert "mean_score" in agg[m]
    assert report["datasets"]["interaction"]["n_samples"] == 120


def test_settings_are_honoured():
    st = Settings(random_state=7, cv=3)
    pipe = FeatPipeline(settings=st)
    assert pipe.settings.random_state == 7
    assert pipe.settings.cv == 3


def test_default_benchmark_set_shapes():
    ds_map = default_benchmark_set()
    assert len(ds_map) >= 5
    for name, ds in ds_map.items():
        assert ds.X.shape[0] > 0 and ds.X.shape[1] > 0
        assert ds.y is not None
        assert len(ds.feature_names) == ds.X.shape[1]
