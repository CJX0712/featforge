"""FeatForgeSynth — flagship #1: relevance-guided automated feature synthesis.

Expands a raw feature matrix into a candidate pool of *domain-guarded*
nonlinear transforms (squares, abs, log1p, sqrt, sigmoid, reciprocal) plus
pairwise interaction products, then keeps only the highest-relevance columns
(mutual information with the target; |corr| offline).

Why it matters: tabular models are linear in their features, so any effect that
lives in an interaction (x0*x1) or a nonlinearity (sin x, x^2) is invisible
until someone materialises it. This flagship materialises exactly those columns,
but *guided by relevance* so the pool stays small and useful rather than
exploding combinatorially.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..core.errors import SynthError
from ..core.types import FeatureFrame
from ..eval.metrics import relevance_scores


@dataclass
class SynthSpec:
    random_state: int = 42
    synth_interactions: int = 20
    synth_top_k: int = 30
    max_pair_scan: int = 400
    task: str = "classification"


class FeatForgeSynth:
    name = "featforge_synth"

    def __init__(
        self,
        random_state: int = 42,
        synth_interactions: int = 20,
        synth_top_k: int = 30,
        max_pair_scan: int = 400,
        task: str = "classification",
        **kw,
    ) -> None:
        self.spec = SynthSpec(
            random_state=int(random_state),
            synth_interactions=int(synth_interactions),
            synth_top_k=int(synth_top_k),
            max_pair_scan=int(max_pair_scan),
            task=task,
        )
        self.metadata: Dict[str, object] = {}

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _clean(M: np.ndarray) -> np.ndarray:
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    def _transform_columns(self, X: np.ndarray, names: List[str]):
        n, d = X.shape
        cols: List[np.ndarray] = []
        cnames: List[str] = []
        for j in range(d):
            c = X[:, j]
            cols.append(c ** 2)
            cnames.append(f"{names[j]}^2")
            cols.append(np.abs(c))
            cnames.append(f"|{names[j]}|")
            cols.append(np.log1p(np.abs(c)))
            cnames.append(f"log1p|{names[j]}|")
            cols.append(np.sqrt(np.abs(c)))
            cnames.append(f"sqrt|{names[j]}|")
            cols.append(1.0 / (1.0 + np.exp(-np.clip(c, -30.0, 30.0))))
            cnames.append(f"sigmoid({names[j]})")
            cols.append(1.0 / (np.abs(c) + 1.0))
            cnames.append(f"inv|{names[j]}|")
        if not cols:
            return np.empty((n, 0)), []
        return np.column_stack(cols), cnames

    def _interaction_pairs(self, d: int):
        ii: List[int] = []
        jj: List[int] = []
        for i in range(d):
            for j in range(i + 1, d):
                ii.append(i)
                jj.append(j)
        if len(ii) > self.spec.max_pair_scan:
            step = int(np.ceil(len(ii) / self.spec.max_pair_scan))
            ii = ii[::step]
            jj = jj[::step]
        return ii, jj

    # ---- main entry -----------------------------------------------------
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> FeatureFrame:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise SynthError(f"X must be 2-D, got {X.ndim}-D")
        n, d = X.shape
        if feature_names is None:
            names = [f"x{i}" for i in range(d)]
        else:
            names = list(feature_names)
        if len(names) != d:
            raise SynthError(f"feature_names length {len(names)} != X columns {d}")

        tf, tf_names = self._transform_columns(X, names)
        n_tf = tf.shape[1]
        ii, jj = self._interaction_pairs(d)
        if ii:
            inter = np.column_stack([X[:, i] * X[:, j] for i, j in zip(ii, jj)])
            inter_names = [f"{names[i]}*{names[j]}" for i, j in zip(ii, jj)]
        else:
            inter = np.empty((n, 0))
            inter_names = []

        if tf.shape[1] == 0 and inter.shape[1] == 0:
            src = {nm: "raw" for nm in names}
            self.metadata = {"n_synth": 0, "selected": []}
            return FeatureFrame(X, names, src)

        Cand = np.column_stack([tf, inter]) if (tf.shape[1] and inter.shape[1]) else (
            tf if tf.shape[1] else inter
        )
        Cand = self._clean(Cand)
        try:
            rel = np.asarray(
                relevance_scores(Cand, y, self.spec.task, self.spec.random_state),
                dtype=np.float64,
            )
        except Exception as e:
            raise SynthError(f"relevance scoring failed: {e}") from e
        if rel.shape[0] != Cand.shape[1]:
            raise SynthError("relevance scores must align with candidate columns")

        tf_rel = rel[:n_tf]
        inter_rel = rel[n_tf:]
        cand_names_all = tf_names + inter_names

        # Interactions get a reserved budget (they are the headline transform and
        # much rarer than the per-column family), but that reservation can never
        # break the overall synth_top_k contract.
        take_inter = int(min(self.spec.synth_interactions, inter.shape[1],
                             self.spec.synth_top_k))
        inter_order = np.argsort(-inter_rel)[:take_inter]
        budget = int(max(0, self.spec.synth_top_k - take_inter))
        take_tf = int(min(budget, n_tf))
        tf_order = np.argsort(-tf_rel)[:take_tf]

        chosen_local = list(tf_order) + list(inter_order + n_tf)
        chosen_sorted = sorted(chosen_local)
        if not chosen_sorted:
            src = {nm: "raw" for nm in names}
            self.metadata = {"n_synth": 0, "selected": []}
            return FeatureFrame(X, names, src)

        Xnew = np.column_stack([X, Cand[:, chosen_sorted]])
        new_names = list(names) + [cand_names_all[k] for k in chosen_sorted]
        src: Dict[str, str] = {nm: "raw" for nm in names}
        for k in chosen_sorted:
            src[cand_names_all[k]] = "synth"

        top = np.argsort(-rel)[: min(5, len(rel))]
        self.metadata = {
            "n_synth": len(chosen_sorted),
            "n_transforms": len(tf_order),
            "n_interactions": len(inter_order),
            "selected": [cand_names_all[k] for k in chosen_sorted],
            "top_relevance": {cand_names_all[k]: float(rel[k]) for k in top},
        }
        return FeatureFrame(self._clean(Xnew), new_names, src)

    @staticmethod
    def default_params() -> dict:
        return {"synth_interactions": 20, "synth_top_k": 30}
