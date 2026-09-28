import os

import numpy as np
import pytest

from featforge.core import ConfigError, DataError, Dataset, FeatureFrame, Result, Settings
from featforge.core.errors import ForgeError


def test_dataset_validates_shape():
    with pytest.raises(ValueError):
        Dataset(np.zeros(10))


def test_dataset_feature_names_mismatch():
    with pytest.raises(ValueError):
        Dataset(np.zeros((5, 3)), feature_names=["a", "b"])


def test_dataset_row_mismatch():
    with pytest.raises(ValueError):
        Dataset(np.zeros((5, 3)), y=np.zeros(4))


def test_dataset_defaults_names():
    ds = Dataset(np.zeros((4, 3)))
    assert ds.feature_names == ["x0", "x1", "x2"]
    assert ds.n_samples == 4 and ds.n_features == 3


def test_featureframe_validates_names():
    with pytest.raises(ValueError):
        FeatureFrame(np.zeros((5, 2)), ["only"])


def test_result_defaults():
    r = Result(name="raw", feature_names=["a"], n_features=1, downstream_score=0.5)
    assert r.cv_scores == [] and r.metadata == {}


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("FF_RANDOM_STATE", "7")
    monkeypatch.setenv("FF_CV", "3")
    monkeypatch.setenv("FF_SELECT_K", "9")
    s = Settings.from_env()
    assert s.random_state == 7
    assert s.cv == 3
    assert s.select_k == 9


def test_settings_env_defaults(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FF_"):
            monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.random_state == 42 and s.cv == 5 and s.select_k == 15


def test_error_hierarchy():
    assert issubclass(DataError, ForgeError)
    assert issubclass(ConfigError, ForgeError)
