"""End-to-end demo: generate benchmark set -> three-arm benchmark -> benchmark.json."""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from featforge.data.synthetic import default_benchmark_set  # noqa: E402
from featforge.pipeline import FeatPipeline  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    datasets = default_benchmark_set()
    pipe = FeatPipeline()
    out = os.path.join(_ROOT, "benchmark.json")
    report = pipe.benchmark(datasets, out_path=out)
    print(pipe.print_report(report))
    print(f"\nArtifacts written: {out}  ({time.perf_counter()-t0:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
