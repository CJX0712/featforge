"""argparse entrypoint for FeatForge.

Commands
--------
benchmark  run the full three-arm benchmark and write benchmark.json
features   inspect what the synthesizer/selector produced for one dataset
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from .data.synthetic import default_benchmark_set
from .pipeline import FeatPipeline
from .select import FeatForgeSelect
from .synth import FeatForgeSynth


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="featforge", description="FeatForge — automated feature engineering")
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("benchmark", help="run the three-arm benchmark")
    b.add_argument("--out", default="benchmark.json", help="output json path")
    b.add_argument("--random-state", type=int, default=42)
    b.add_argument("--cv", type=int, default=5)
    b.add_argument("--select-k", type=int, default=12)
    b.add_argument("--synth-top-k", type=int, default=30)

    f = sub.add_parser("features", help="show synthesized + selected features")
    f.add_argument("--dataset", default="interaction")
    f.add_argument("--random-state", type=int, default=42)
    f.add_argument("--select-k", type=int, default=12)
    f.add_argument("--synth-top-k", type=int, default=30)
    return p


def _dataset_names() -> List[str]:
    return list(default_benchmark_set().keys())


def main(argv: Optional[List[str]] = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__
        print(f"FeatForge {__version__}")
        return 0
    if args.cmd is None:
        p.print_help()
        return 0

    if args.cmd == "benchmark":
        pipe = FeatPipeline(synth_kw={"synth_top_k": args.synth_top_k},
                            select_kw={"select_k": args.select_k})
        import time
        t0 = time.perf_counter()
        report = pipe.benchmark(default_benchmark_set(), out_path=args.out)
        print(pipe.print_report(report))
        print(f"\nArtifacts written: {args.out}  ({time.perf_counter()-t0:.2f}s)")
        return 0

    if args.cmd == "features":
        ds_map = default_benchmark_set()
        if args.dataset not in ds_map:
            print(f"unknown dataset: {args.dataset} (available: {', '.join(_dataset_names())})")
            return 2
        ds = ds_map[args.dataset]
        synth = FeatForgeSynth(random_state=args.random_state,
                               synth_top_k=args.synth_top_k, task=ds.task)
        ff = synth.fit_transform(ds.X, ds.y, ds.feature_names)
        sel = FeatForgeSelect(random_state=args.random_state, select_k=args.select_k,
                              task=ds.task)
        sr = sel.select(ff.X, ds.y, ff.feature_names)
        print(f"dataset={args.dataset}  raw_features={ds.n_features}  pool={ff.X.shape[1]}")
        print("\nTop synthesized (highest relevance):")
        for nm, v in sorted(synth.metadata.get("top_relevance", {}).items(),
                            key=lambda kv: -kv[1])[:10]:
            print(f"  {nm:<28}{v:>10.4f}")
        print(f"\nSelected ({len(sr.chosen_names)} features):")
        for nm in sr.chosen_names:
            print(f"  {nm}")
        if sr.forward_gains:
            print(f"\nForward-selection trace: {[round(g,4) for g in sr.forward_gains]}")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
