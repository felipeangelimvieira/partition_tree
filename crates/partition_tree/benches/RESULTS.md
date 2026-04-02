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
