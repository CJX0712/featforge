"""FeatForgeSelect — flagship #2: mRMR pre-filter + CV-driven forward selection.

Two stage design, both label-guided but *test-set-free* (selection uses only
cross-validated downstream scores on the training pool):

1. **mRMR pre-filter** (max-relevance, min-redundancy): keeps candidates that
   carry information about the target while being dissimilar to what is already
   chosen, so near-duplicate synthesized columns cannot crowd out real signal.
2. **Forward selection**: greedily grows the set, adding whichever candidate
   best raises the *actual downstream metric*. This is the part that adapts to
   the estimator instead of relying on a filter heuristic alone.

The result is a compact, non-redundant feature set with a score trace, so the
user can audit exactly why each column survived.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ..core.config import Settings
from ..core.errors import SelectError
from ..eval.metrics import abs_corr_matrix, downstream_cv, relevance_scores


@dataclass
class SelectionResult:
    name: str
    chosen_names: List[str]
    chosen_indices: List[int]
    relevance: Dict[str, float] = field(default_factory=dict)
    forward_gains: List[float] = field(default_factory=list)
    runtime_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatForgeSelect:
    name = "featforge_select"

    def __init__(
        self,
        random_state: int = 42,
        cv: int = 5,
        select_k: int = 12,
        select_prefilter: int = 30,
        task: str = "classification",
        tol: float = 1e-4,
        n_jobs: int = 1,
        **kw,
    ) -> None:
        self.random_state = int(random_state)
        self.cv = int(cv)
        self.select_k = int(select_k)
        self.select_prefilter = int(select_prefilter)
        self.task = task
        self.tol = float(tol)
        self.n_jobs = int(n_jobs)

    # ---- mRMR -----------------------------------------------------------
    def _mrmr(self, rel: np.ndarray, corr: np.ndarray, limit: int) -> List[int]:
        d = int(rel.shape[0])
        limit = max(1, min(int(limit), d))
        first = int(np.argmax(rel))
        selected = [first]
        rest = [j for j in range(d) if j != first]
        while len(selected) < limit and rest:
            best = None
            best_score = -np.inf
            for j in rest:
                red = float(np.mean([corr[j, s] for s in selected]))
                score = float(rel[j]) - red
                if score > best_score:
                    best_score = score
                    best = int(j)
            if best is None:
                break
            selected.append(best)
            rest.remove(best)
        return selected

    # ---- forward selection ----------------------------------------------
    def _forward(self, X: np.ndarray, y: np.ndarray, cand: List[int],
                 settings: Settings):
        best_single = None
        best_score = -np.inf
        for c in cand:
            s, _ = downstream_cv(X[:, [c]], y, settings)
            if s > best_score:
                best_score = s
                best_single = c
        if best_single is None:
            raise SelectError("forward selection found no usable candidate")
        chosen = [int(best_single)]
        gains = [float(best_score)]
        cur = float(best_score)
        pool = [c for c in cand if c != best_single]
        while len(chosen) < self.select_k and pool:
            trial_best = None
            trial_score = -np.inf
            for c in pool:
                s, _ = downstream_cv(X[:, chosen + [c]], y, settings)
                if s > trial_score:
                    trial_score = s
                    trial_best = int(c)
            if trial_best is not None and (trial_score - cur) > self.tol:
                chosen.append(trial_best)
                pool.remove(trial_best)
                gains.append(float(trial_score))
                cur = float(trial_score)
            else:
                break
        return chosen, gains

    # ---- main entry -----------------------------------------------------
    def select(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> SelectionResult:
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        names = list(feature_names)
        if len(names) != d:
            raise SelectError(f"feature_names length {len(names)} != X columns {d}")
        if d == 0:
            raise SelectError("empty feature pool")
        if y is None:
            raise SelectError("selection requires targets")

        settings = Settings(
            random_state=self.random_state,
            cv=self.cv,
            task=self.task,
            n_jobs=self.n_jobs,
        )
        rel = np.asarray(
            relevance_scores(X, y, self.task, self.random_state), dtype=np.float64
        )
        if rel.shape[0] != d:
            raise SelectError("relevance must align with feature columns")
        rel_map = {nm: float(v) for nm, v in zip(names, rel)}

        if d <= self.select_k:
            return SelectionResult(
                name=self.name,
                chosen_names=names,
                chosen_indices=list(range(d)),
                relevance=rel_map,
                forward_gains=[],
                runtime_s=time.perf_counter() - t0,
                metadata={"pruned": False, "note": "pool already within budget"},
            )

        corr = abs_corr_matrix(X)
        pre = self._mrmr(rel, corr, self.select_prefilter)
        chosen, gains = self._forward(X, y, pre, settings)
        chosen = sorted(set(int(c) for c in chosen))
        return SelectionResult(
            name=self.name,
            chosen_names=[names[i] for i in chosen],
            chosen_indices=chosen,
            relevance=rel_map,
            forward_gains=gains,
            runtime_s=time.perf_counter() - t0,
            metadata={
                "pruned": True,
                "n_pool": d,
                "n_prefilter": len(pre),
                "prefilter": [names[i] for i in pre],
            },
        )

    @staticmethod
    def default_params() -> dict:
        return {"select_k": 12, "select_prefilter": 30}
