"""Configuration for FeatForge, overridable via FF_* environment variables.

`available_sklearn()` probes whether scikit-learn is importable so the eval /
selector layers can decide between the SOTA backend and the pure-numpy fallback.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class Settings:
    random_state: int = 42
    cv: int = 5
    n_jobs: int = 1
    # synthesis knobs
    synth_interactions: int = 20          # max pairwise interaction columns
    synth_top_k: int = 30                 # max total synthesized columns
    # selection knobs
    select_k: int = 15                    # final compact feature count
    select_prefilter: int = 40            # mRMR pre-filter width
    task: str = "classification"
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        def _int(key: str, default: int) -> int:
            v = os.environ.get(key)
            return int(v) if v not in (None, "") else default

        return cls(
            random_state=_int("FF_RANDOM_STATE", 42),
            cv=_int("FF_CV", 5),
            n_jobs=_int("FF_NJOBS", 1),
            synth_interactions=_int("FF_SYNTH_INTERACTIONS", 20),
            synth_top_k=_int("FF_SYNTH_TOP_K", 30),
            select_k=_int("FF_SELECT_K", 15),
            select_prefilter=_int("FF_SELECT_PREFILTER", 40),
            task=os.environ.get("FF_TASK", "classification") or "classification",
        )

    @staticmethod
    def available_sklearn() -> bool:
        try:
            import sklearn  # noqa: F401
            return True
        except Exception:
            return False


def get_settings() -> Settings:
    return Settings.from_env()


SETTINGS_HELP: Dict[str, str] = {
    "FF_RANDOM_STATE": "Global RNG seed (default 42).",
    "FF_CV": "Cross-validation folds for downstream scoring (default 5).",
    "FF_NJOBS": "Job parallelism hint (default 1; kept 1 for determinism).",
    "FF_SYNTH_INTERACTIONS": "Max pairwise interaction features (default 20).",
    "FF_SYNTH_TOP_K": "Max total synthesized columns (default 30).",
    "FF_SELECT_K": "Final compact feature count (default 15).",
    "FF_SELECT_PREFILTER": "mRMR pre-filter width (default 40).",
    "FF_TASK": "classification | regression (default classification).",
}
