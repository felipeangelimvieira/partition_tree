"""Performance diagnostic for ``PartitionForestRegressor.predict_proba``.

The merged predictive distribution is built entirely in Rust
(``predict_proba_merged_segments``) and returned as flat CSR arrays. This
script times that production path and compares it against two Rust-level
reference points to confirm where wall-clock time is spent.

Stages
------
S0  ``PartitionForestRegressor.predict_proba`` -- the full production path
                                                  (Rust merge + IntervalDistribution
                                                  construction in Python).
S1  Rust ``predict_proba_merged_segments``     -- merge only, flat arrays, no
                                                  Python object construction.
B1  Rust forest-level ``predict_proba``        -- ensembled piecewise dists
                                                  (n_samples pyclass objects);
                                                  boundary/object reference.

Run
---
    uv run python benchmarks/diagnose_forest_proba.py
    uv run python benchmarks/diagnose_forest_proba.py --profile
    uv run python benchmarks/diagnose_forest_proba.py \
        --trees 10 50 100 --samples 1000 10000 --repeat 5
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from statistics import median

import numpy as np
import pandas as pd
import polars as pl

from partition_tree.skpro.partition_tree import PartitionForestRegressor
from partition_tree.utils import (
    _convert_string_columns_to_categorical,
    _ensure_numeric_float64,
    _preprocess_X,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_regression(n_rows: int, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))
    coef = rng.standard_normal(n_features)
    y = X @ coef + 0.3 * rng.standard_normal(n_rows)
    X_df = pd.DataFrame(X, columns=[f"x{j}" for j in range(n_features)])
    y_df = pd.DataFrame({"y": y})
    return X_df, y_df


def to_polars(estimator, X: pd.DataFrame) -> pl.DataFrame:
    """Replicate the preprocessing done inside ``_predict_proba``."""
    X_proc = _ensure_numeric_float64(_preprocess_X(X))
    X_pol = pl.DataFrame(X_proc)
    X_pol = _convert_string_columns_to_categorical(
        X_pol, categories_map=getattr(estimator, "_categorical_metadata", None)
    )
    X_pol = _ensure_numeric_float64(X_pol)
    return X_pol, X_proc.index


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def timeit(fn, repeat: int):
    samples = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    return median(samples), out


def run_case(n_trees: int, n_samples: int, repeat: int, max_leaves: int):
    X, y = make_regression(n_rows=max(n_samples, 2000))
    X_train, y_train = X.iloc[:2000], y.iloc[:2000]
    X_test = X.iloc[:n_samples]

    forest = PartitionForestRegressor(
        n_estimators=n_trees,
        max_leaves=max_leaves,
        random_state=0,
    )
    forest.fit(X_train, y_train)

    X_pol, _index = to_polars(forest, X_test)
    pf = forest.partition_forest_

    # warm up (build caches, stabilise allocator/OS)
    forest.predict_proba(X_test)
    pf.predict_proba_merged_segments(X_pol)
    pf.predict_proba(X_pol)

    # S0: full production path (Rust merge + Python IntervalDistribution build)
    t_s0, _ = timeit(lambda: forest.predict_proba(X_test), repeat)

    # S1: Rust merge only (flat arrays, no Python object construction)
    t_s1, _ = timeit(lambda: pf.predict_proba_merged_segments(X_pol), repeat)

    # B1: Rust forest-level predict_proba -> n_samples ensembled objects
    t_b1, _ = timeit(lambda: pf.predict_proba(X_pol), repeat)

    return {
        "trees": n_trees,
        "samples": n_samples,
        "S0_predict_proba_ms": t_s0,
        "S1_rust_merge_ms": t_s1,
        "B1_rust_ensembled_ms": t_b1,
        "py_overhead_ms": t_s0 - t_s1,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(rows):
    cols = [
        ("trees", "trees", 5),
        ("samples", "samples", 7),
        ("S0_predict_proba_ms", "S0 predict_proba", 16),
        ("S1_rust_merge_ms", "S1 rust merge", 13),
        ("B1_rust_ensembled_ms", "B1 ensembled", 12),
        ("py_overhead_ms", "S0-S1 (py)", 11),
    ]
    header = "  ".join(name.ljust(w) for _, name, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for key, _, w in cols:
            v = r[key]
            if isinstance(v, float):
                cells.append(f"{v:10.2f}".ljust(w))
            else:
                cells.append(str(v).ljust(w))
        print("  ".join(cells))


def profile_full(n_trees: int, n_samples: int, max_leaves: int):
    """cProfile the full predict_proba to surface remaining Python hotspots."""
    X, y = make_regression(n_rows=max(n_samples, 2000))
    forest = PartitionForestRegressor(
        n_estimators=n_trees, max_leaves=max_leaves, random_state=0
    )
    forest.fit(X.iloc[:2000], y.iloc[:2000])
    X_test = X.iloc[:n_samples]
    forest.predict_proba(X_test)  # warm

    pr = cProfile.Profile()
    pr.enable()
    forest.predict_proba(X_test)
    pr.disable()

    print(f"\n=== cProfile: predict_proba (trees={n_trees}, samples={n_samples}) ===")
    stats = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
    stats.print_stats(15)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trees", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--samples", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--max-leaves", type=int, default=32)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    rows = []
    for n_samples in args.samples:
        for n_trees in args.trees:
            print(f"running trees={n_trees} samples={n_samples} ...", flush=True)
            rows.append(run_case(n_trees, n_samples, args.repeat, args.max_leaves))

    print("\nAll times are median of {} runs, milliseconds.\n".format(args.repeat))
    print_table(rows)
    print(
        "\nLegend: S0 = production predict_proba (Rust merge + Python build). "
        "S1 = Rust merge only (flat arrays). B1 = Rust ensembled call.\n"
        "S0-S1 approximates the remaining Python-side IntervalDistribution "
        "construction cost."
    )

    if args.profile:
        profile_full(max(args.trees), min(args.samples), args.max_leaves)


if __name__ == "__main__":
    main()
