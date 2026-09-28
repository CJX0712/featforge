"""FeatPipeline — orchestrates synthesis + selection and measures downstream gain.

Three arms are evaluated per dataset under an identical protocol so any
difference is attributable to the features, not the model:

* `raw`       — the original feature matrix (baseline).
* `synth_all` — raw + every synthesized candidate (tests synthesis value).
* `selected`  — mRMR + forward-selected compact set (tests selection value).

Direction of calls: `pipeline -> {data, synth, select, eval} -> core`.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import Settings, get_settings
from ..core.types import Dataset
from ..eval import available_sklearn, downstream_cv
from ..select import FeatForgeSelect
from ..synth import FeatForgeSynth

METHODS = ["raw", "synth_all", "selected"]


def _round(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


class FeatPipeline:
    def __init__(self, settings: Optional[Settings] = None,
                 synth_kw: Optional[Dict[str, Any]] = None,
                 select_kw: Optional[Dict[str, Any]] = None):
        self.settings = settings or get_settings()
        self.synth_kw = dict(synth_kw or {})
        self.select_kw = dict(select_kw or {})

    def _settings_for(self, ds: Dataset) -> Settings:
        return Settings(
            random_state=self.settings.random_state,
            cv=self.settings.cv,
            n_jobs=self.settings.n_jobs,
            task=ds.task,
        )

    # ---- single dataset -------------------------------------------------
    def evaluate(self, ds: Dataset) -> Dict[str, Dict[str, Any]]:
        st = self._settings_for(ds)
        out: Dict[str, Dict[str, Any]] = {}

        # 1) raw baseline
        t0 = time.perf_counter()
        score, scores = downstream_cv(ds.X, ds.y, st)
        out["raw"] = {
            "n_features": ds.n_features,
            "downstream_score": _round(score),
            "cv_scores": [_round(s) for s in scores],
            "runtime_s": _round(time.perf_counter() - t0),
            "feature_names": list(ds.feature_names or []),
        }

        # 2) synthesised (all candidates)
        t0 = time.perf_counter()
        synth = FeatForgeSynth(
            random_state=self.settings.random_state,
            synth_interactions=self.synth_kw.get(
                "synth_interactions", self.settings.synth_interactions),
            synth_top_k=self.synth_kw.get("synth_top_k", self.settings.synth_top_k),
            task=ds.task,
        )
        ff = synth.fit_transform(ds.X, ds.y, ds.feature_names)
        score, scores = downstream_cv(ff.X, ds.y, st)
        out["synth_all"] = {
            "n_features": ff.X.shape[1],
            "downstream_score": _round(score),
            "cv_scores": [_round(s) for s in scores],
            "runtime_s": _round(time.perf_counter() - t0),
            "feature_names": list(ff.feature_names),
            "metadata": synth.metadata,
        }

        # 3) selection over the synthesised pool
        t0 = time.perf_counter()
        sel = FeatForgeSelect(
            random_state=self.settings.random_state,
            cv=self.settings.cv,
            task=ds.task,
            select_k=self.select_kw.get("select_k", self.settings.select_k),
            select_prefilter=self.select_kw.get(
                "select_prefilter", self.settings.select_prefilter),
        )
        sr = sel.select(ff.X, ds.y, ff.feature_names)
        Xsel = ff.X[:, sr.chosen_indices]
        score, scores = downstream_cv(Xsel, ds.y, st)
        out["selected"] = {
            "n_features": int(Xsel.shape[1]),
            "downstream_score": _round(score),
            "cv_scores": [_round(s) for s in scores],
            "runtime_s": _round(time.perf_counter() - t0),
            "feature_names": list(sr.chosen_names),
            "forward_gains": [_round(g) for g in sr.forward_gains],
            "metadata": sr.metadata,
        }
        return out

    # ---- benchmark ------------------------------------------------------
    def benchmark(self, datasets: Dict[str, Dataset],
                  out_path: Optional[str] = None) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "meta": {
                "tool": "FeatForge",
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "random_state": self.settings.random_state,
                "cv": self.settings.cv,
                "methods": METHODS,
                "sklearn": _sklearn_version(),
                "backend": "sklearn" if available_sklearn() else "numpy-fallback",
            },
            "datasets": {},
            "aggregate": {},
        }
        for name, ds in datasets.items():
            report["datasets"][name] = {
                "n_samples": ds.n_samples,
                "n_features": ds.n_features,
                "task": ds.task,
                "results": self.evaluate(ds),
            }
        report["aggregate"] = self._aggregate(report["datasets"])
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    @staticmethod
    def _aggregate(block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        agg: Dict[str, Dict[str, List[float]]] = {}
        for ds_name, blk in block.items():
            for method, rec in blk["results"].items():
                agg.setdefault(method, {})
                agg[method].setdefault("score", []).append(float(rec["downstream_score"]))
                agg[method].setdefault("n_features", []).append(float(rec["n_features"]))
                agg[method].setdefault("runtime_s", []).append(float(rec["runtime_s"]))
        out: Dict[str, Dict[str, Any]] = {}
        for method, vals in agg.items():
            out[method] = {
                "mean_score": _round(float(np.mean(vals["score"]))),
                "mean_n_features": _round(float(np.mean(vals["n_features"])), 2),
                "mean_runtime_s": _round(float(np.mean(vals["runtime_s"]))),
            }
        return out

    # ---- pretty print ---------------------------------------------------
    def print_report(self, report: Dict[str, Any]) -> str:
        lines: List[str] = []
        for ds_name, blk in report["datasets"].items():
            lines.append(
                f"\n=== {ds_name} (n={blk['n_samples']}, raw_d={blk['n_features']}) ===")
            header = f"{'method':<12}{'features':>10}{'score':>10}{'time_s':>10}"
            lines.append(header)
            for method in METHODS:
                rec = blk["results"].get(method)
                if rec is None:
                    continue
                lines.append(
                    f"{method:<12}{rec['n_features']:>10}{rec['downstream_score']:>10.4f}"
                    f"{rec['runtime_s']:>10.2f}")
        agg = report.get("aggregate", {})
        if agg:
            lines.append("\n=== Aggregate (mean over datasets) ===")
            lines.append(f"{'method':<12}{'score':>10}{'n_feat':>10}{'time_s':>10}")
            for method in METHODS:
                if method in agg:
                    m = agg[method]
                    lines.append(
                        f"{method:<12}{m['mean_score']:>10.4f}"
                        f"{m['mean_n_features']:>10.2f}{m['mean_runtime_s']:>10.2f}")
        return "\n".join(lines)


def _sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "unknown"
