# Rust Merged Segments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 90%-of-total-time Python `IntervalDistribution.from_mixture` bottleneck by performing the piecewise-density merge in Rust before crossing the FFI boundary.

**Architecture:** Add `predict_proba_merged_segments` to `PartitionForest` in Rust, expose it via a new PyO3 method returning four flat arrays (densities, lows, highs, offsets), and update `PartitionForestRegressor._predict_proba` to call this method instead of the current `predict_trees_proba → per-tree Python loop → from_mixture` path.

**Tech Stack:** Rust (rayon, partition_tree crate), PyO3, NumPy (Python side only — no Rust numpy dep), pytest, maturin/uv for build.

---

## Background: why the current path is slow

| Stage                                                    | Cost (100t × 10k samples) | Where                         |
| -------------------------------------------------------- | ------------------------- | ----------------------------- |
| S1 FFI boundary (`predict_trees_proba`)                  | ~1 s                      | Rust→Python object allocation |
| S2 Python build loop (`pdf_segments` + `argsort` + ctor) | ~15 s                     | Python                        |
| S3 `from_mixture` breakpoint merge                       | ~194 s                    | Python (O(n·T²))              |

Root cause: `from_mixture` iterates O(n_samples × n_breakpoints × n_trees × n_segments).  
Since each sample hits exactly one leaf per tree, `n_segments_per_tree = 1` and `n_breakpoints = O(n_trees)`, making it O(n_samples × n_trees²).

Target: move the breakpoint-union merge to Rust (rayon-parallel across samples).  
Expected: >50× total speedup on 100t×10k samples, getting close to the 1.5s Rust ceiling.

---

## Key architecture facts to keep in mind

- `PartitionForest` lives in `crates/partition_tree/src/estimators/forest.rs`.
- `PyPartitionForest` lives in `pyo3_partition_tree/src/lib.rs`.
- `PiecewiseConstantDistribution::pdf_segments()` returns `Vec<(density, low, high)>`. For a single-tree distribution (one cell per sample), `density = 1/vol` always.
- For a forest with uniform weights, the merged density at a point covered by trees t₁…tₖ is `(1/n_trees) × Σ density_tᵢ`.
- Python `IntervalDistribution.__init__` accepts `intervals` as list of `(low, high)` tuples and `pdf_values` as list of raw densities (non-normalized; normalization happens inside the ctor).
- All Python tests run via `uv run pytest` from `partition_tree/` (never the global interpreter).
- Rust benchmarks run from `crates/partition_tree/` (not workspace root).

---

## File Map

| File                                                        | Action | Responsibility                                                              |
| ----------------------------------------------------------- | ------ | --------------------------------------------------------------------------- |
| `crates/partition_tree/src/estimators/forest.rs`            | Modify | Add `merge_segments_for_sample` fn + `predict_proba_merged_segments` method |
| `pyo3_partition_tree/src/lib.rs`                            | Modify | Add `predict_proba_merged_segments` to `PyPartitionForest`                  |
| `partition_tree/src/partition_tree/skpro/partition_tree.py` | Modify | Add `_predict_proba_rust_merged`; update `_predict_proba`                   |
| `partition_tree/tests/skpro/test_merged_predict_proba.py`   | Create | Parity tests: new path vs old path                                          |
| `partition_tree/benchmarks/diagnose_forest_proba.py`        | Modify | Add S0 stage for new path                                                   |

---

## Task 1: Rust — segment merge helper

**Files:**
- Modify: `crates/partition_tree/src/estimators/forest.rs`

### Context

Each sample hits exactly one leaf per tree, so `pdf_segments()` returns exactly one `(density, low, high)` tuple per tree per sample. Uniform weight `= 1 / n_trees`.

The merge algorithm for one sample (input: n_trees segments):
1. Collect all 2×n_trees boundary values.
2. Sort + dedup (f64 dedup with tolerance ε=1e-12).
3. For each sub-interval `[bps[j], bps[j+1]]`, test the midpoint against every input segment → sum matching densities / n_trees.
4. Emit `(merged_density, low, high)` for any sub-interval with density > 0.

- [ ] **Step 1: Write failing Rust unit test**

Add this test module at the bottom of `crates/partition_tree/src/estimators/forest.rs`:

```rust
#[cfg(test)]
mod merge_tests {
    use super::merge_segments_for_sample;

    #[test]
    fn test_single_segment_passthrough() {
        // One tree, one segment → same segment with density / 1
        let segs = vec![(0.5_f64, 0.0_f64, 2.0_f64)]; // (density, low, high)
        let result = merge_segments_for_sample(&segs, 1.0);
        assert_eq!(result.len(), 1);
        let (d, l, h) = result[0];
        assert!((d - 0.5).abs() < 1e-10, "density={d}");
        assert!((l - 0.0).abs() < 1e-10);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_non_overlapping_trees() {
        // Tree 1: [0, 1] density=1.0, Tree 2: [2, 4] density=0.5
        // After merge (weight 0.5 each):
        //   [0,1] → 1.0/2 = 0.5
        //   [2,4] → 0.5/2 = 0.25
        let segs = vec![(1.0_f64, 0.0_f64, 1.0_f64), (0.5_f64, 2.0_f64, 4.0_f64)];
        let result = merge_segments_for_sample(&segs, 2.0);
        assert_eq!(result.len(), 2, "expected 2 non-overlapping output segments");
        // Sorted by low
        let (d0, l0, h0) = result[0];
        let (d1, l1, h1) = result[1];
        assert!((l0 - 0.0).abs() < 1e-10 && (h0 - 1.0).abs() < 1e-10);
        assert!((d0 - 0.5).abs() < 1e-10, "d0={d0}");
        assert!((l1 - 2.0).abs() < 1e-10 && (h1 - 4.0).abs() < 1e-10);
        assert!((d1 - 0.25).abs() < 1e-10, "d1={d1}");
    }

    #[test]
    fn test_two_overlapping_trees() {
        // Tree 1: [0, 2] density=0.5 (vol=2, mass=1)
        // Tree 2: [1, 3] density=0.5 (vol=2, mass=1)
        // Expected sub-intervals:
        //   [0,1] → only tree1 → 0.5/2 = 0.25
        //   [1,2] → both → (0.5+0.5)/2 = 0.5
        //   [2,3] → only tree2 → 0.5/2 = 0.25
        let segs = vec![(0.5_f64, 0.0_f64, 2.0_f64), (0.5_f64, 1.0_f64, 3.0_f64)];
        let result = merge_segments_for_sample(&segs, 2.0);
        assert_eq!(result.len(), 3);
        let (d0, l0, h0) = result[0];
        let (d1, l1, h1) = result[1];
        let (d2, l2, h2) = result[2];
        assert!((l0 - 0.0).abs() < 1e-10 && (h0 - 1.0).abs() < 1e-10);
        assert!((d0 - 0.25).abs() < 1e-10, "d0={d0}");
        assert!((l1 - 1.0).abs() < 1e-10 && (h1 - 2.0).abs() < 1e-10);
        assert!((d1 - 0.50).abs() < 1e-10, "d1={d1}");
        assert!((l2 - 2.0).abs() < 1e-10 && (h2 - 3.0).abs() < 1e-10);
        assert!((d2 - 0.25).abs() < 1e-10, "d2={d2}");
    }

    #[test]
    fn test_merged_density_integrates_to_component_sum_proportion() {
        // With uniform weights: ∫ merged_pdf dx = (Σ_t ∫ density_t dx) / n_trees
        // For a single tree with density=1/vol, ∫ density dx = 1.
        // So ∫ merged_pdf dx should = n_active_samples_in_trees / n_trees.
        // Here both trees overlap completely → ∫ = (1+1)/2 = 1 (but densities add)
        let segs = vec![(1.0_f64, 0.0_f64, 1.0_f64), (1.0_f64, 0.0_f64, 1.0_f64)];
        let result = merge_segments_for_sample(&segs, 2.0);
        assert_eq!(result.len(), 1);
        let (d, l, h) = result[0];
        let integral: f64 = d * (h - l);
        assert!((integral - 1.0).abs() < 1e-10, "integral={integral}");
    }
}
```

- [ ] **Step 2: Verify test fails**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/crates/partition_tree
cargo test merge_tests 2>&1 | tail -20
```

Expected: `error[E0425]: cannot find function merge_segments_for_sample`

- [ ] **Step 3: Implement `merge_segments_for_sample` and `predict_proba_merged_segments`**

Add the following to `crates/partition_tree/src/estimators/forest.rs`, inside the `impl PartitionForest` block after the existing `merge_distributions` method:

```rust
/// Merge per-tree pdf segments for one sample into non-overlapping intervals.
///
/// `segments` — one `(density, low, high)` per tree (exactly one segment per
/// tree per sample, since each sample maps to one leaf).
/// `n_trees` — divisor for uniform weighting (weight = 1/n_trees per tree).
///
/// Returns sorted, non-overlapping `(merged_density, low, high)` tuples with
/// positive density.
pub(crate) fn merge_segments_for_sample(
    segments: &[(f64, f64, f64)],
    n_trees: f64,
) -> Vec<(f64, f64, f64)> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Collect all boundary points
    let mut bps: Vec<f64> = Vec::with_capacity(2 * segments.len());
    for &(_, low, high) in segments {
        bps.push(low);
        bps.push(high);
    }
    bps.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Dedup with floating-point tolerance
    let epsilon = 1e-12_f64;
    bps.dedup_by(|a, b| (*a - *b).abs() <= epsilon);

    if bps.len() < 2 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(bps.len() - 1);

    for j in 0..bps.len() - 1 {
        let (low, high) = (bps[j], bps[j + 1]);
        if high <= low + epsilon {
            continue;
        }
        let mid = 0.5 * (low + high);

        let density: f64 = segments
            .iter()
            .filter(|&&(_, seg_low, seg_high)| seg_low <= mid && mid < seg_high)
            .map(|&(d, _, _)| d)
            .sum::<f64>()
            / n_trees;

        if density > 0.0 {
            result.push((density, low, high));
        }
    }

    result
}

/// Predict merged piecewise-constant segment arrays for all samples.
///
/// Returns four flat arrays in CSR style:
/// - `densities[offsets[i]..offsets[i+1]]` — raw densities for sample i
/// - `lows[offsets[i]..offsets[i+1]]`     — interval lower bounds
/// - `highs[offsets[i]..offsets[i+1]]`    — interval upper bounds
/// - `offsets` — length n_samples + 1
///
/// Densities are NOT normalized (sum ≠ 1 in general); Python
/// `IntervalDistribution` normalizes internally.
pub fn predict_proba_merged_segments(
    &self,
    x: &DataFrame,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>), PredictError> {
    let trees = self.fitted_trees()?;
    let xy = self.expand_with_schema(x)?;
    let dataset = self.build_prediction_dataset(&xy)?;
    let n_samples = x.height();
    let n_trees = trees.len() as f64;

    // Parallel per-tree: Vec<tree_idx> of Vec<sample_idx> of Vec<(density,low,high)>
    let per_tree: Vec<Vec<Vec<(f64, f64, f64)>>> = trees
        .par_iter()
        .map(|tree| {
            tree.predict_distributions(&dataset)
                .into_iter()
                .map(|dist| dist.pdf_segments())
                .collect()
        })
        .collect();

    // Parallel per-sample: merge segments from all trees
    let merged: Vec<Vec<(f64, f64, f64)>> = (0..n_samples)
        .into_par_iter()
        .map(|s| {
            let sample_segs: Vec<(f64, f64, f64)> = per_tree
                .iter()
                .flat_map(|tree_segs| tree_segs[s].iter().copied())
                .collect();
            Self::merge_segments_for_sample(&sample_segs, n_trees)
        })
        .collect();

    // Flatten into CSR arrays
    let total: usize = merged.iter().map(|v| v.len()).sum();
    let mut densities = Vec::with_capacity(total);
    let mut lows = Vec::with_capacity(total);
    let mut highs = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(n_samples + 1);
    offsets.push(0usize);

    for sample_segs in &merged {
        for &(d, l, h) in sample_segs {
            densities.push(d);
            lows.push(l);
            highs.push(h);
        }
        offsets.push(densities.len());
    }

    Ok((densities, lows, highs, offsets))
}
```

- [ ] **Step 4: Run failing tests, confirm they pass**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/crates/partition_tree
cargo test merge_tests 2>&1 | tail -20
```

Expected: `test merge_tests::test_single_segment_passthrough ... ok` (all 4 pass)

- [ ] **Step 5: Commit**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency
git add crates/partition_tree/src/estimators/forest.rs
git commit -m "feat(rust): add merge_segments_for_sample + predict_proba_merged_segments

Parallel rayon merge of per-tree pdf segments into non-overlapping
CSR arrays. Replaces the O(n·T²) Python from_mixture bottleneck.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: PyO3 — expose method to Python

**Files:**
- Modify: `pyo3_partition_tree/src/lib.rs`

- [ ] **Step 1: Add `predict_proba_merged_segments` to `PyPartitionForest`**

Inside the existing `#[pymethods] impl PyPartitionForest` block (after `predict_trees_proba`, around line 697), add:

```rust
/// Predict merged piecewise-constant segments for all samples.
///
/// Returns a 4-tuple ``(densities, lows, highs, offsets)`` where all are
/// Python lists of floats / ints. For sample ``i``, the segments are at
/// indices ``offsets[i]`` to ``offsets[i+1]`` (exclusive) in the flat arrays.
///
/// This is the fast path that replaces ``predict_trees_proba`` +
/// Python ``IntervalDistribution.from_mixture``.
pub fn predict_proba_merged_segments(
    &self,
    x: PyDataFrame,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>)> {
    let x_df: PolarsDataFrame = x.into();
    self.inner
        .predict_proba_merged_segments(&x_df)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
```

- [ ] **Step 2: Build the Python extension**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/partition_tree
uv run maturin develop --release 2>&1 | tail -10
```

Expected: `Finished release [optimized] target(s)` with no errors.

- [ ] **Step 3: Smoke-test the new method from Python**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/partition_tree
uv run python - <<'EOF'
import numpy as np
import polars as pl
from sklearn.datasets import make_regression
from pyo3_partition_tree import PyPartitionForest

X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=0)
X_pl = pl.DataFrame(X, schema=[f"f{i}" for i in range(4)])
y_pl = pl.DataFrame({"y": y.tolist()})

forest = PyPartitionForest(n_estimators=5, max_leaves=10, seed=0)
forest.fit(X_pl, y_pl)

densities, lows, highs, offsets = forest.predict_proba_merged_segments(X_pl)
print(f"offsets len={len(offsets)}, total_segs={offsets[-1]}")
print(f"sample 0: {offsets[1]-offsets[0]} segments")
print(f"lows[:5]={lows[:5]}")
EOF
```

Expected: offsets len=101, reasonable segment counts and values.

- [ ] **Step 4: Commit**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency
git add pyo3_partition_tree/src/lib.rs
git commit -m "feat(pyo3): expose predict_proba_merged_segments on PyPartitionForest

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: Python — add `_predict_proba_rust_merged` + parity tests

**Files:**
- Modify: `partition_tree/src/partition_tree/skpro/partition_tree.py`
- Create: `partition_tree/tests/skpro/test_merged_predict_proba.py`

### Context

`_predict_proba_rust_merged` will:
1. Call `predict_proba_merged_segments` → 4 flat Python lists
2. Convert to NumPy and split per sample using `offsets`
3. Build `IntervalDistribution` directly (bypassing `from_mixture`)

The intervals passed to `IntervalDistribution.__init__` must be `(low, high)` tuples (2-element). The segments from Rust are already sorted by `low` (guaranteed by `merge_segments_for_sample` which sorts all breakpoints before processing).

Add the import at the top of `partition_tree.py` (it likely already has `import numpy as np`; confirm it does before adding).

- [ ] **Step 1: Write the failing parity test first**

Create `partition_tree/tests/skpro/test_merged_predict_proba.py`:

```python
"""Parity tests: Rust-merged path vs Python from_mixture path."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from partition_tree.skpro.partition_tree import PartitionForestRegressor


@pytest.fixture(scope="module")
def fitted_forest_and_data():
    X, y = make_regression(
        n_samples=300, n_features=5, noise=0.5, random_state=42
    )
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    model = PartitionForestRegressor(
        n_estimators=10,
        max_leaves=20,
        max_depth=4,
        seed=7,
        output_distribution="merged",
    )
    model.fit(
        pd.DataFrame(X_train, columns=[f"f{i}" for i in range(5)]),
        pd.DataFrame({"y": y_train}),
    )
    X_test_df = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(5)])
    return model, X_test_df


def _old_merged_dist(model, X_test_df):
    """Reproduce the original from_mixture path for comparison."""
    from partition_tree.skpro.distribution import IntervalDistribution

    interval_dists = model.predict_proba_per_tree(X_test_df)
    weights = [1.0 / len(interval_dists)] * len(interval_dists)
    return IntervalDistribution.from_mixture(
        distributions=interval_dists,
        weights=weights,
        index=X_test_df.index,
        columns=model._y_columns,
    )


def test_rust_merged_returns_interval_distribution(fitted_forest_and_data):
    model, X_test_df = fitted_forest_and_data
    dist = model._predict_proba_rust_merged(X_test_df)
    from partition_tree.skpro.distribution import IntervalDistribution
    assert isinstance(dist, IntervalDistribution)


def test_rust_merged_has_correct_number_of_instances(fitted_forest_and_data):
    model, X_test_df = fitted_forest_and_data
    dist = model._predict_proba_rust_merged(X_test_df)
    assert len(dist.index) == len(X_test_df)


def test_rust_merged_intervals_are_sorted_and_non_overlapping(fitted_forest_and_data):
    model, X_test_df = fitted_forest_and_data
    dist = model._predict_proba_rust_merged(X_test_df)
    for i, intervals in enumerate(dist._intervals):
        for j in range(len(intervals) - 1):
            assert intervals[j].high <= intervals[j + 1].low, (
                f"Instance {i}: intervals not sorted/disjoint at j={j}: "
                f"{intervals[j]} vs {intervals[j+1]}"
            )


def test_rust_merged_pdf_integrates_to_one(fitted_forest_and_data):
    """Normalized densities × widths should sum to 1 for every instance."""
    model, X_test_df = fitted_forest_and_data
    dist = model._predict_proba_rust_merged(X_test_df)
    for i in range(len(dist.index)):
        intervals = dist._intervals[i]
        densities = dist.pdf_values[i]
        norm = dist._normalization_factor[i]
        if norm <= 0:
            continue
        total = sum(
            float(d) * (iv.high - iv.low) / norm
            for d, iv in zip(densities, intervals)
        )
        assert abs(total - 1.0) < 1e-8, f"Instance {i}: integral={total}"


def test_rust_merged_mean_close_to_python_from_mixture(fitted_forest_and_data):
    """Means should agree to within 1e-6 (same algorithm, floating-point order differs)."""
    model, X_test_df = fitted_forest_and_data
    dist_new = model._predict_proba_rust_merged(X_test_df)
    dist_old = _old_merged_dist(model, X_test_df)
    mean_new = dist_new.mean().values.flatten()
    mean_old = dist_old.mean().values.flatten()
    np.testing.assert_allclose(mean_new, mean_old, rtol=1e-4, atol=1e-6,
        err_msg="Rust-merged means differ from Python from_mixture means")


def test_rust_merged_pdf_close_to_python_from_mixture(fitted_forest_and_data):
    """PDF at fixed test points should agree between old and new path."""
    model, X_test_df = fitted_forest_and_data
    dist_new = model._predict_proba_rust_merged(X_test_df)
    dist_old = _old_merged_dist(model, X_test_df)

    # Sample a few query values from interval midpoints of the old dist
    rng = np.random.default_rng(0)
    n_check = min(20, len(X_test_df))
    for i in rng.choice(len(X_test_df), size=n_check, replace=False):
        intervals_old = dist_old._intervals[i]
        if not intervals_old:
            continue
        # Pick midpoint of a random interval
        iv = intervals_old[rng.integers(len(intervals_old))]
        xq = np.full(len(dist_old.index), np.nan)
        xq[i] = 0.5 * (iv.low + iv.high)
        pdf_new = float(dist_new._pdf(xq).iloc[i, 0])
        pdf_old = float(dist_old._pdf(xq).iloc[i, 0])
        assert abs(pdf_new - pdf_old) < 1e-6, (
            f"Instance {i}: pdf_new={pdf_new} vs pdf_old={pdf_old}"
        )
```

- [ ] **Step 2: Run tests to confirm they fail (method missing)**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/partition_tree
uv run pytest tests/skpro/test_merged_predict_proba.py -v 2>&1 | tail -20
```

Expected: `AttributeError: 'PartitionForestRegressor' object has no attribute '_predict_proba_rust_merged'`

- [ ] **Step 3: Implement `_predict_proba_rust_merged` in `partition_tree.py`**

In `partition_tree/src/partition_tree/skpro/partition_tree.py`, add the following method to `PartitionForestRegressor` after `predict_proba_per_tree` (around line 335):

```python
def _predict_proba_rust_merged(self, X):
    """Compute merged predictive distribution via Rust segment merge.

    Replaces ``predict_proba_per_tree`` + ``IntervalDistribution.from_mixture``
    with a single Rust call that merges piecewise-constant segments in parallel
    and returns flat CSR arrays.
    """
    from partition_tree.skpro.distribution import IntervalDistribution

    check_is_fitted(self)
    X_proc = _ensure_numeric_float64(_preprocess_X(X))
    X_pol = pl.DataFrame(X_proc)
    X_pol = _convert_string_columns_to_categorical(
        X_pol, categories_map=getattr(self, "_categorical_metadata", None)
    )
    X_pol = _ensure_numeric_float64(X_pol)

    densities_flat, lows_flat, highs_flat, offsets = (
        self.partition_forest_.predict_proba_merged_segments(X_pol)
    )

    densities_arr = np.asarray(densities_flat, dtype=float)
    lows_arr = np.asarray(lows_flat, dtype=float)
    highs_arr = np.asarray(highs_flat, dtype=float)

    n_samples = len(offsets) - 1
    intervals_per_row = []
    pdf_values_per_row = []

    for i in range(n_samples):
        start, end = offsets[i], offsets[i + 1]
        row_lows = lows_arr[start:end]
        row_highs = highs_arr[start:end]
        row_densities = densities_arr[start:end]
        intervals_per_row.append(
            list(zip(row_lows.tolist(), row_highs.tolist()))
        )
        pdf_values_per_row.append(row_densities)

    return IntervalDistribution(
        intervals=intervals_per_row,
        pdf_values=pdf_values_per_row,
        index=X_proc.index,
        columns=self._y_columns,
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/partition_tree
uv run pytest tests/skpro/test_merged_predict_proba.py -v 2>&1 | tail -30
```

Expected: all 6 tests pass (`PASSED`).

- [ ] **Step 5: Commit**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency
git add partition_tree/src/partition_tree/skpro/partition_tree.py \
        partition_tree/tests/skpro/test_merged_predict_proba.py
git commit -m "feat(python): add _predict_proba_rust_merged + parity tests

New method calls predict_proba_merged_segments (Rust) and constructs
IntervalDistribution directly from flat CSR arrays, bypassing
from_mixture. 6 parity tests verify correctness vs old path.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: Switch `_predict_proba` to the new path

**Files:**
- Modify: `partition_tree/src/partition_tree/skpro/partition_tree.py`

- [ ] **Step 1: Write failing regression test**

Add to `partition_tree/tests/skpro/test_merged_predict_proba.py`:

```python
def test_predict_proba_uses_rust_merged_path(fitted_forest_and_data, monkeypatch):
    """After switching, _predict_proba for 'merged' must call the Rust path."""
    model, X_test_df = fitted_forest_and_data
    calls = []
    orig = model._predict_proba_rust_merged

    def spy(X):
        calls.append(True)
        return orig(X)

    monkeypatch.setattr(model, "_predict_proba_rust_merged", spy)
    model._predict_proba(X_test_df)
    assert len(calls) == 1, "_predict_proba did not delegate to _predict_proba_rust_merged"
```

- [ ] **Step 2: Confirm test fails**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/partition_tree
uv run pytest tests/skpro/test_merged_predict_proba.py::test_predict_proba_uses_rust_merged_path -v 2>&1 | tail -10
```

Expected: `FAILED` — assertion error (calls == 0).

- [ ] **Step 3: Update `_predict_proba`**

In `partition_tree/src/partition_tree/skpro/partition_tree.py`, replace the existing `_predict_proba` body:

```python
# Before (original):
def _predict_proba(self, X):
    check_is_fitted(self)
    if self.output_distribution not in ("merged", "mixture"):
        raise ValueError(
            f"output_distribution must be 'merged' or 'mixture', "
            f"got {self.output_distribution!r}"
        )
    X_proc = _ensure_numeric_float64(_preprocess_X(X))
    interval_dists = self.predict_proba_per_tree(X)
    weights = [1.0 / len(interval_dists)] * len(interval_dists)
    if self.output_distribution == "mixture":
        return MixtureIntervalDistribution(
            distributions=interval_dists,
            weights=weights,
            index=X_proc.index,
            columns=self._y_columns,
        )

    return IntervalDistribution.from_mixture(
        distributions=interval_dists,
        weights=weights,
        index=X_proc.index,
        columns=self._y_columns,
    )
```

```python
# After:
def _predict_proba(self, X):
    check_is_fitted(self)
    if self.output_distribution not in ("merged", "mixture"):
        raise ValueError(
            f"output_distribution must be 'merged' or 'mixture', "
            f"got {self.output_distribution!r}"
        )
    if self.output_distribution == "mixture":
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        interval_dists = self.predict_proba_per_tree(X)
        weights = [1.0 / len(interval_dists)] * len(interval_dists)
        return MixtureIntervalDistribution(
            distributions=interval_dists,
            weights=weights,
            index=X_proc.index,
            columns=self._y_columns,
        )

    return self._predict_proba_rust_merged(X)
```

- [ ] **Step 4: Run full test suite**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/partition_tree
uv run pytest tests/ -v 2>&1 | tail -40
```

Expected: all previously-passing tests still pass, plus 7 new tests all green.

- [ ] **Step 5: Commit**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency
git add partition_tree/src/partition_tree/skpro/partition_tree.py \
        partition_tree/tests/skpro/test_merged_predict_proba.py
git commit -m "feat(python): switch _predict_proba 'merged' path to Rust merge

Eliminates predict_proba_per_tree + IntervalDistribution.from_mixture
for the merged output mode. 'mixture' path is unchanged.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: Benchmark the new path

**Files:**
- Modify: `partition_tree/benchmarks/diagnose_forest_proba.py`

- [ ] **Step 1: Add S0 (Rust-merged) stage to the diagnostic script**

In `partition_tree/benchmarks/diagnose_forest_proba.py`, find the section that measures stages and add a new `S0` measurement. Look for the block that measures `S1` (the `predict_trees_proba` call) and add before it:

```python
# S0: new Rust-merged path (end-to-end, merged output)
t0 = time.perf_counter()
for _ in range(args.repeat):
    _ = model._predict_proba_rust_merged(X_test_df)
t_s0 = (time.perf_counter() - t0) / args.repeat
results.append(("S0_rust_merged", n_trees, n_samples, t_s0))
```

Also add `"S0_rust_merged"` to the printed table header and rows so it appears alongside S1/S2/S3/B1.

- [ ] **Step 2: Run the benchmark sweep**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency/partition_tree
uv run python benchmarks/diagnose_forest_proba.py --trees 10 50 100 --samples 1000 10000 --repeat 2 2>&1
```

Expected: S0 times are dramatically lower than S1+S2+S3 total. At 100t×10k, S0 should be in the 1–10s range (vs ~215s before).

- [ ] **Step 3: Append benchmark results to RESULTS.md**

Append the S0 timing table to `crates/partition_tree/benches/RESULTS.md` under a new section `## Phase C: Rust-Merged Path Speedup`.

- [ ] **Step 4: Commit**

```bash
cd /Users/felipeangelim/Workspace/partition_tree.worktrees/copilot-optimize-partition-forest-efficiency
git add partition_tree/benchmarks/diagnose_forest_proba.py \
        crates/partition_tree/benches/RESULTS.md
git commit -m "bench: add S0 rust-merged stage to diagnostic; record Phase C results

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Merge in Rust (`merge_segments_for_sample` + `predict_proba_merged_segments`) — Task 1
- [x] PyO3 binding returning flat arrays — Task 2
- [x] Python constructs `IntervalDistribution` directly from arrays — Task 3
- [x] `_predict_proba` delegates to new path — Task 4
- [x] Benchmark S0 vs S1+S2+S3 — Task 5
- [x] Tests for numerical parity — Task 3
- [x] Rust unit tests — Task 1

**Placeholder scan:** All code blocks are complete. No TBD, TODO, or "similar to above" references.

**Type consistency:**
- `merge_segments_for_sample` — used in both Rust test (Task 1) and `predict_proba_merged_segments` body (Task 1) ✓
- `predict_proba_merged_segments` — defined in `forest.rs` (Task 1), wrapped in `lib.rs` (Task 2), called from Python (Task 3) ✓
- `_predict_proba_rust_merged` — defined (Task 3), spy-tested (Task 4), called from `_predict_proba` (Task 4) ✓
- Return type `(Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>)` consistent across Rust and PyO3 binding ✓
