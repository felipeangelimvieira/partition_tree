# Benchmark Results

Machine: macOS (Apple Silicon)
Date: 2026-04-02
Command: `cargo bench --bench forest_bench`

## Optimizations applied (Phase A)

- Hoisted `col_map` out of per-row loop in `predict_distributions`
- Pre-built `ConditionedCell` cache for all leaves before per-row iteration
- Parallelised per-row loop with `into_par_iter()` via rayon
- Parallelised `merge_distributions` sample loop in `PartitionForest`

---

## Single tree build

| Dataset size | Time (ms) | vs. baseline |
| ------------ | --------- | ------------ |
| 500 rows     | 2.69      | -5.5%        |
| 2 000 rows   | 4.81      | -67.8%       |
| 10 000 rows  | 15.59     | -48.5%       |

## Forest fit

| Rows  | Trees | Time (ms) | vs. baseline |
| ----- | ----- | --------- | ------------ |
| 1 000 | 10    | 12.60     | -62.9%       |
| 1 000 | 50    | 41.98     | -62.6%       |
| 5 000 | 10    | 38.58     | -63.9%       |

## Forest predict

| Trees | Method        | Time (ms) | vs. baseline |
| ----- | ------------- | --------- | ------------ |
| 10    | predict       | 7.94      | -43.1%       |
| 10    | predict_proba | 4.41      | -40.1%       |
| 50    | predict       | 38.72     | -33.4%       |
| 50    | predict_proba | 22.33     | -31.9%       |
| 100   | predict       | 77.89     | -29.9%       |
| 100   | predict_proba | 45.41     | -33.4%       |

---

## Phase B diagnostic — `predict_proba` boundary/merge attribution (2026-05-29)

Investigates whether merging per-tree distributions in Rust (before crossing
into Python) is worthwhile. Two harnesses:

- Python stage timings: `partition_tree/benchmarks/diagnose_forest_proba.py`
- Rust ceiling: `forest_proba_ceiling` group in this bench.

### Python merged path (`PartitionForestRegressor.predict_proba`, "merged")

Median of 3 runs, milliseconds. Stages: S1 = Rust `predict_trees_proba`
(n_trees×n_samples pyclass objects), S2 = Python build loop
(`pdf_segments` + per-row `argsort` + IntervalDistribution ctor),
S3 = `IntervalDistribution.from_mixture` (pure-Python breakpoint merge).
B1 = Rust forest-level `predict_proba` (already ensembles cells → n_samples
objects only).

| Trees | Samples | S1 rust | S2 py build | S3 merge | B1 ensembled | Total merged | S3 share |
| ----- | ------- | ------- | ----------- | -------- | ------------ | ------------ | -------- |
| 10    | 1 000   | 5.4     | 188.8       | 351.8    | 13.1         | 546.0        | 64.4%    |
| 50    | 1 000   | 24.9    | 963.6       | 5 862.8  | 58.1         | 6 851.3      | 85.6%    |
| 100   | 1 000   | 47.8    | 1 888.9     | 18 889.6 | 108.3        | 20 826.2     | 90.7%    |
| 10    | 10 000  | 48.2    | 1 894.0     | 3 379.9  | 114.4        | 5 322.2      | 63.5%    |
| 50    | 10 000  | 270.4   | 10 267.9    | 54 594.9 | 825.0        | 65 133.2     | 83.8%    |
| 100   | 10 000  | 661.7   | 20 721.0    | 193 661  | 1 546.5      | 215 044      | 90.1%    |

`MixtureIntervalDistribution` construction (lazy alternative to S3) is ~0.1 ms
in all cases — but defers cost to later pdf/cdf/energy calls.

### cProfile (50 trees × 1 000 samples, merged)

```
11.9 s total
 10.08 s  from_mixture                         (84% cumulative)
  7.53 s    └ tottime in from_mixture body
  2.06 s  Interval.contains  ← 28,897,602 calls (the O(n·T²·S) inner loop)
  1.81 s  predict_proba_per_tree (S2)
  1.63 s    └ IntervalDistribution.__init__ (eager energy cache, discarded after merge)
```

### Rust ceiling (`forest_proba_ceiling`, criterion median)

| Trees | Samples | `predict_proba` (Rust) | + `pdf_segments` extract |
| ----- | ------- | ---------------------- | ------------------------ |
| 10    | 1 000   | 13.4 ms                | 15.5 ms                  |
| 100   | 1 000   | 127.7 ms               | 153.1 ms                 |
| 10    | 10 000  | 135.0 ms               | 156.5 ms                 |
| 100   | 10 000  | 1 290.7 ms             | 1 536.5 ms               |

### Conclusions

1. **The bottleneck is pure Python, not the FFI boundary nor Rust.** At
   100 trees × 10 000 samples the merged path is **215 s**, of which
   `from_mixture` (S3) is **194 s (90%)** and S2 is 21 s; total Rust work (S1)
   is **0.66 s (<1%)**.
2. **`from_mixture` scales ~quadratically in trees** (10→50→100 at 10k:
   3.4 s → 54.6 s → 193.7 s), driven by ~29 M `Interval.contains` calls. Python
   stages scale ~linearly in samples.
3. **S1 vs B1**: per-tree transfer (S1) is only marginally above the ensembled
   call (B1); object-count overhead is negligible relative to the merge. The win
   comes from replacing the *algorithm*, not just from reducing object count.
4. **Rust ceiling is ~1.5 s** for 100×10 000 (ensemble + segment extraction),
   i.e. the proposed Rust-merge solution has **>100× headroom** versus today's
   215 s, even before adding the breakpoint-union merge (which becomes compiled,
   rayon-parallel code instead of the Python triple loop).

→ Merging in Rust (and returning flat segment arrays rather than
n_trees×n_samples pyclass objects) is the right lever. Sorting segments in Rust
and avoiding eager per-tree energy caches are cheap independent wins.

---

## Phase C — Rust-merged path validation (2026-05-29)

`_predict_proba_rust_merged` now replaces the S1+S2+S3 pipeline for the
`merged` output mode. S0 = the new path end-to-end. `speedup` = old
total_merged / S0.

Median of 3 runs, milliseconds.

| Trees | Samples | S0 new (ms) | old total (ms) | speedup   |
| ----- | ------- | ----------- | -------------- | --------- |
| 10    | 1 000   | 77          | 534            | 6.9×      |
| 50    | 1 000   | 263         | 6 936          | 26.3×     |
| 100   | 1 000   | 405         | 21 182         | 52.4×     |
| 10    | 10 000  | 643         | 5 374          | 8.4×      |
| 50    | 10 000  | 2 313       | 63 628         | 27.5×     |
| 100   | 10 000  | 4 087       | 213 918        | **52.3×** |

### Observations

- Speedup scales with n_trees (S3 grows quadratically; S0 grows linearly in trees × samples).
- At the most demanding workload (100t × 10k samples): **4.1 s vs 214 s**.
- All 117 Python tests pass; 6 parity tests confirm numerical agreement with old path.
