# Tree refinement after fit (`Tree::refined`, `refine_after_fit`)

Date: 2026-05-15

## Problem

After a `Tree` is fitted, the unique split coordinates observed anywhere in
the tree do not, in general, partition every leaf. Concretely: a threshold
`x = 5` may have been used in one branch of the tree but a leaf in a
*different* branch can still span `x ∈ [0, 10]`. Drawing that split as a
hyperplane across the whole space cuts through the interior of the second
leaf.

For downstream visualization, integration, and density estimation we want
the invariant:

> No unique split coordinate (X or Y, continuous, integer, quantized, or
> categorical) crosses the interior of any leaf cell.

The naive cure — subdivide every leaf by every unique coordinate — must
not corrupt the model: density attributes and predictions should remain
identical to the un-refined tree.

## Request

Implement this refinement as a method on the existing tree, exposed at the
estimator level as an optional flag, with the following user-stated
constraints:

1. **Scope**: propagate **both** X-splits and Y-splits.
2. **Output**: a method `tree.refined() -> Tree` returning a new tree;
   the original is left untouched.
3. **Categorical (`BelongsTo`) splits**: propagate all unique subsets; a
   leaf whose active set *strictly contains* a used subset is refined into
   the equivalence classes induced by that subset.
4. **Inherited volume on refined leaves**: add
   `inherited_target_volume: Option<f64>` to `FittedNode` so refined
   leaves preserve the *parent leaf's* density and volume attributes (the
   geometric cell volume shrinks, but the inherited override gives the
   source value).
5. **Estimator-level flag**: expose an optional
   `refine_after_fit: bool` on `PartitionTree`, and verify that predictions
   on the training data are identical whether or not refinement is applied.

## Strategy

### 1. `FittedNode` extension (minimal)

Added an optional metadata field plus a helper:

- `FittedNode.inherited_target_volume: Option<f64>` —
  the source leaf's `target_volume()` at the moment of refinement.
- `FittedNode::effective_target_volume()` — returns the inherited override
  when set, otherwise falls back to the geometric `cell.target_volume()`.
- `FittedNode::conditional_density()` and `Tree::leaf_info()` use
  `effective_target_volume`, so refined leaves report the source leaf's
  density and volume.

All existing `FittedNode` construction sites (in `tree.rs`,
`tree_builder.rs`, `predict/conditioned_cell.rs` test fixtures) were
updated to initialize `inherited_target_volume: None`. The field is
`#[serde(default)]` so older serialized trees deserialize cleanly.

### 2. `crates/partition_tree/src/refine.rs` (new module)

- `enum SplitCoord` — dtype-erased recovered split point (Continuous,
  Integer, Quantized, Categorical).
- `recover_split_coord(parent, left, col)` — reconstructs the actual
  threshold / subset from a `SplitRecord` by comparing parent and child
  rules.
- `is_interior(cell, col, coord)` — true iff `coord` is strictly inside
  the cell's rule for `col`. The refinement invariant is *no
  interior crossings on any leaf*.
- `collect_unique_split_coords(tree)` — gathers every distinct
  `SplitCoord` from `tree.split_history`, keyed by column.
- `Refiner` — owns a fresh node arena, an `old → new` index map, and a
  list of synthetic `SplitRecord`s (with `gain = 0.0` so feature
  importances are unaffected).
- `Refiner::walk(...)` — top-down traversal that copies original internal
  nodes verbatim and, on reaching a leaf, hands off to
  `refine_leaf_subtree`. Candidate `SplitCoord`s are pruned as we descend
  (a coord that already lies on a boundary of the current sub-cell is
  dropped from the candidate list).
- `Refiner::refine_leaf_subtree(...)` — recursively subdivides a leaf by
  every still-interior candidate, growing a binary sub-tree of synthetic
  splits. **Each refined sub-leaf keeps `w_xy`, `w_x`, `w_y` literally
  equal to the source leaf's values** and records
  `inherited_target_volume = Some(source.target_volume())`.
- `Refiner::into_tree()` — finalizes the arena, remaps the original
  split history through the `old → new` map, and appends the synthetic
  records.
- `impl Tree { pub fn refined(&self) -> Tree }` — public API.

The refinement is idempotent: a second `refined()` call adds no new
leaves because no coord is interior anymore.

### 3. Prediction-invariance: rescale at the boundary, not inside refine

The naive design (preserve the source's `w_xy` on every refined leaf,
also override `target_volume` for density) breaks `predict_distributions`
under Y-refinement: multiple refined cells match the same `X`, each
carrying the full source mass, so the total mass for a query doubles
(or worse), distorting `predict_mean` and `predict_proba`.

We considered rescaling `w_xy` / `w_x` / `w_y` inside `refine` itself
(mirroring `TreeBuilder` X/Y semantics), but that mutates the
"refined leaves inherit source weights" invariant the user asked for.

The minimal-change fix is to rescale **only at the prediction boundary**,
inside `ConditionedCell::from_fitted_node`, using the already-stored
`inherited_target_volume`:

```text
mass = (w_xy / w_x) * (cell.target_volume() / inherited_target_volume)
```

When `inherited_target_volume` is `None` (unrefined node) this is a no-op
and `mass = w_xy / w_x` unchanged. For refined sibling cells from one
source leaf, the target-volume fractions sum to 1, so their rescaled
masses sum back to the source leaf's mass — making the per-row
distribution identical to the unrefined tree's.

This isolates the only place that needs to "know" about refinement to
the conditioned-cell construction site. `FittedNode` itself stays
honest: `w_xy`, `w_x`, `w_y` on refined leaves *are* the source leaf's
weights, exactly as requested.

### 4. Estimator-level `refine_after_fit`

`PartitionTree` (in `crates/partition_tree/src/estimators/tree.rs`)
gained a `pub refine_after_fit: bool` field, defaulted to `false` and
serialized with `#[serde(default)]` for backward compatibility.
`_fit_impl` calls `tree.refined()` at the end of fitting when the flag
is set.

### 5. Tests

- **`refine` module** (11 tests, in `refine.rs`):
  - `refined_root_only_is_unchanged`
  - `refined_propagates_x_split_to_target_branch_leaf`
  - `refined_y_split_uses_inherited_volume_to_preserve_density`
  - `refined_density_matches_original_routing`
  - `refined_invariant_no_interior_crossings_after_one_pass`
  - `refined_is_idempotent`
  - `refined_categorical_partitions_into_equivalence_classes`
  - `refined_serde_roundtrip_preserves_inherited_target_volume`
  - `refined_split_history_remaps_indices`
  - `refined_feature_importances_unchanged_by_synthetic_splits`
  - `refined_uses_collect_unique_split_coords_deduplication`

- **Estimator-level** (new test in `estimators/tree.rs`):
  `refine_after_fit_preserves_predictions` — fits two `PartitionTree`s
  on the same data with identical seeds, one with `refine_after_fit =
  true`, and asserts:
  - The refined tree has ≥ leaves than the plain tree.
  - Per-row `predict(...)` matches to within `1e-12`.
  - Per-row `predict_proba(...)` has identical `total_mass` and
    `mean_vector` to within `1e-10`.

All 135 lib tests and 11 integration tests pass; no new linter
warnings introduced.

## Files touched

### Rust core

- `crates/partition_tree/src/tree.rs` — added
  `inherited_target_volume` field + `effective_target_volume` helper,
  routed `conditional_density()` and `LeafInfo` through the override.
- `crates/partition_tree/src/tree_builder.rs` — initialized
  `inherited_target_volume: None` at root/child construction sites.
- `crates/partition_tree/src/predict/conditioned_cell.rs` — rescale
  `mass` by `cell.target_volume() / inherited_target_volume` when set
  (no-op for unrefined nodes).
- `crates/partition_tree/src/refine.rs` — new module, `pub mod refine;`
  added to `lib.rs`.

### Rust estimators

- `crates/partition_tree/src/estimators/tree.rs` —
  `PartitionTree.refine_after_fit: bool` (default `false`,
  `#[serde(default)]`), branch in `_fit_impl` calling
  `tree.refined()`, and
  `refine_after_fit_preserves_predictions` test.
- `crates/partition_tree/src/estimators/forest.rs` —
  `PartitionForest.refine_after_fit: bool` (same defaults / serde
  treatment), per-tree refinement inside the parallel `_fit_impl`
  loop, and the field propagated into the returned struct.

### Python bindings (pyo3)

- `pyo3_partition_tree/src/lib.rs`:
  - `PyPartitionTree.__new__` gained a `refine_after_fit: bool = False`
    keyword argument, plus matching getter/setter
    (`@property` / `@setter` on the Rust side via `#[getter]` /
    `#[setter]`) so callers can flip the flag on an already-constructed
    instance.
  - `PyPartitionForest.__new__` gained the same keyword + accessor pair.
  - Both classes plumb the flag onto the inner Rust estimator before
    returning, mirroring how `max_candidate_split_points` is wired.

### Python tests

- `pyo3_partition_tree/tests/test_refine_after_fit.py` — 12 new tests
  covering, for both `PyPartitionTree` and `PyPartitionForest`:
  - default value of the new flag,
  - getter/setter roundtrip,
  - per-row equality of `predict(...)` between refined and unrefined
    fits with identical seed,
  - per-row equality of `predict_proba(...)` total mass and mean
    vectors,
  - leaf count is monotone non-decreasing under refinement,
  - pickle roundtrip preserves the flag and the resulting predictions.

All 135 Rust lib tests + 11 Rust integration tests + 116 Python tests
(104 pre-existing + 12 new) pass. No new linter warnings introduced.
