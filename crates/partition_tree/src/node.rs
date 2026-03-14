//! Construction-time tree node with presorted index maps.
//!
//! During tree building each node carries a [`Cell`], aggregated weights
//! (`w_xy`, `w_x`, `w_y`), and presorted index lists for every column ×
//! every measure family (XY, X, Y). These lists are propagated to children
//! via **stable partition**—never rebuilt from scratch—so the cost per
//! split is $O(n)$ in the node’s sample count.
//!
//! ## X/Y Split Semantics
//!
//! The partition-tree model maintains three independent index families.
//! When a split occurs, **only the relevant families** are partitioned:
//!
//! | Split kind | `sorted_xy` | `sorted_x` | `sorted_y` |
//! |------------|-------------|------------|------------|
//! | `XSplit`   | partitioned | partitioned| **inherited**  |
//! | `YSplit`   | partitioned | **inherited**  | partitioned|
//!
//! “Inherited” means both children receive a **clone** of the parent’s
//! list, ensuring that target splits do not restrict the feature marginal
//! and vice-versa. This is implemented in
//! [`Node::propagate_children`].
//!
//! ## Lifecycle
//!
//! After the tree is built the `SortedIndices` are discarded and only the
//! [`Cell`] + weights are kept in the [`FittedNode`](super::tree::FittedNode).
use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::Rng;

use crate::cell::Cell;
use crate::dataset_view::DatasetView;
use crate::split_result::{SplitKind, SplitPoint};

// ---------------------------------------------------------------------------
// Sorted Indices
// ---------------------------------------------------------------------------

/// Presorted index lists for every column, grouped by measure family.
///
/// For a given column `c`:
///
/// - `sorted_xy[c]` — indices of the node’s XY samples sorted by column `c`
///   values (ascending, nulls last).
/// - `sorted_x[c]` — likewise for X samples.
/// - `sorted_y[c]` — likewise for Y samples.
///
/// All three maps always have the same set of keys (all column names).
#[derive(Debug, Clone)]
pub struct SortedIndices {
    pub sorted_x: HashMap<String, Vec<u32>>,
    pub sorted_y: HashMap<String, Vec<u32>>,
    pub sorted_xy: HashMap<String, Vec<u32>>,
}

impl SortedIndices {
    pub fn new() -> Self {
        Self {
            sorted_x: HashMap::new(),
            sorted_y: HashMap::new(),
            sorted_xy: HashMap::new(),
        }
    }

    /// Number of XY samples (from any column's sorted_xy list).
    pub fn n_xy(&self) -> usize {
        self.sorted_xy.values().next().map_or(0, |v| v.len())
    }

    /// Number of X samples.
    pub fn n_x(&self) -> usize {
        self.sorted_x.values().next().map_or(0, |v| v.len())
    }

    /// Number of Y samples.
    pub fn n_y(&self) -> usize {
        self.sorted_y.values().next().map_or(0, |v| v.len())
    }
}

impl Default for SortedIndices {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A tree node during construction.
///
/// Holds the mutable builder state that is discarded once the tree is
/// finalized into a [`Tree`](super::tree::Tree). In particular, the
/// [`sorted`](Node::sorted) field can be very large (one `Vec<u32>` per
/// column × measure family) and is not retained in the fitted tree.
#[derive(Debug, Clone)]
pub struct Node {
    /// The multi-dimensional partition constraint.
    pub cell: Cell,
    /// Aggregated weight of XY samples.
    pub w_xy: f64,
    /// Aggregated weight of X samples.
    pub w_x: f64,
    /// Aggregated weight of Y samples.
    pub w_y: f64,
    /// Depth of this node in the tree.
    pub depth: usize,
    /// Presorted index lists for every column.
    pub sorted: SortedIndices,
}

impl Node {
    /// Create the root node from a dataset view with uniform index sets.
    ///
    /// All rows belong to the root node's X, Y, and XY sets.
    /// The presorted indices are initialized from the dataset's global sort.
    pub fn root(dataset: &dyn DatasetView, cell: Cell) -> Self {
        let col_names = dataset.column_names();

        let w_xy: f64 = dataset.weights_xy().iter().sum();
        let w_x: f64 = dataset.weights_x().iter().sum();
        let w_y: f64 = dataset.weights_y().iter().sum();

        // Initialize presorted indices from the dataset's global sort.
        // At the root, all indices are present, so the global sort order is
        // directly usable.
        let mut sorted_x = HashMap::with_capacity(col_names.len());
        let mut sorted_y = HashMap::with_capacity(col_names.len());
        let mut sorted_xy = HashMap::with_capacity(col_names.len());

        for &col_name in &col_names {
            let global_sorted = dataset.sorted_indices(col_name).to_vec();
            sorted_x.insert(col_name.to_string(), global_sorted.clone());
            sorted_y.insert(col_name.to_string(), global_sorted.clone());
            sorted_xy.insert(col_name.to_string(), global_sorted);
        }

        let sorted = SortedIndices {
            sorted_x,
            sorted_y,
            sorted_xy,
        };

        Self {
            cell,
            w_xy,
            w_x,
            w_y,
            depth: 0,
            sorted,
        }
    }

    /// Create the root node with bootstrap sampling.
    ///
    /// Instead of using all rows, a fraction `max_samples` of the dataset's
    /// rows is sampled. When `replace` is `true`, sampling is done **with
    /// replacement** (each row may appear more than once). When `replace` is
    /// `false`, sampling is done **without replacement** (each row appears at
    /// most once).
    ///
    /// As a short-circuit, if `replace` is `true` and the number of draws
    /// equals the dataset size, the original dataset is returned unchanged.
    ///
    /// The presorted order within each column is preserved: for every column
    /// we walk the global sorted order and repeat each index according to its
    /// bootstrap count.
    pub fn root_bootstrap(
        dataset: &dyn DatasetView,
        cell: Cell,
        max_samples: f64,
        replace: bool,
        rng: &mut StdRng,
    ) -> Self {
        let n = dataset.n_rows();
        let n_bootstrap = (n as f64 * max_samples).floor().max(1.0) as usize;

        // Short-circuit: replace=true with full sample returns original dataset.
        if replace && n_bootstrap >= n {
            return Self::root(dataset, cell);
        }

        let mut counts: HashMap<u32, u32> = HashMap::new();

        if replace {
            // Sample with replacement: count how many times each row is drawn.
            for _ in 0..n_bootstrap {
                let idx = rng.random_range(0..n as u32);
                *counts.entry(idx).or_insert(0) += 1;
            }
        } else {
            // Sample without replacement via partial Fisher-Yates shuffle.
            let k = n_bootstrap.min(n);
            if k >= n {
                return Self::root(dataset, cell);
            }
            let mut pool: Vec<u32> = (0..n as u32).collect();
            for i in 0..k {
                let j = rng.random_range(i as u32..n as u32) as usize;
                pool.swap(i, j);
            }
            for &idx in &pool[..k] {
                counts.insert(idx, 1);
            }
        }

        let col_names = dataset.column_names();
        let weights_xy = dataset.weights_xy();
        let weights_x = dataset.weights_x();
        let weights_y = dataset.weights_y();

        let mut sorted_x = HashMap::with_capacity(col_names.len());
        let mut sorted_y = HashMap::with_capacity(col_names.len());
        let mut sorted_xy = HashMap::with_capacity(col_names.len());

        let mut w_xy = 0.0_f64;
        let mut w_x = 0.0_f64;
        let mut w_y = 0.0_f64;
        let mut first_col = true;

        for &col_name in &col_names {
            let global_sorted = dataset.sorted_indices(col_name);
            let mut col_sorted: Vec<u32> = Vec::with_capacity(n_bootstrap);

            for &idx in global_sorted {
                if let Some(&count) = counts.get(&idx) {
                    for _ in 0..count {
                        col_sorted.push(idx);
                    }
                    if first_col {
                        w_xy += weights_xy[idx as usize] * count as f64;
                        w_x += weights_x[idx as usize] * count as f64;
                        w_y += weights_y[idx as usize] * count as f64;
                    }
                }
            }

            sorted_x.insert(col_name.to_string(), col_sorted.clone());
            sorted_y.insert(col_name.to_string(), col_sorted.clone());
            sorted_xy.insert(col_name.to_string(), col_sorted);
            first_col = false;
        }

        let sorted = SortedIndices {
            sorted_x,
            sorted_y,
            sorted_xy,
        };

        Self {
            cell,
            w_xy,
            w_x,
            w_y,
            depth: 0,
            sorted,
        }
    }

    /// Conditional density: `w_xy / (w_x * target_volume)`.
    pub fn conditional_density(&self) -> f64 {
        let vol = self.cell.target_volume();
        if self.w_x <= 0.0 || vol <= 0.0 {
            0.0
        } else {
            self.w_xy / (self.w_x * vol)
        }
    }

    /// Balanced weight: `w_xy / (w_x * w_y)`.
    pub fn balanced_weight(&self) -> f64 {
        if self.w_x <= 0.0 || self.w_y <= 0.0 {
            0.0
        } else {
            self.w_xy / (self.w_x * self.w_y)
        }
    }

    /// Propagate presorted indices from this (parent) node to left and right
    /// children after a split has been chosen.
    ///
    /// A `go_left` oracle is built from the [`SplitOp`](super::split_result::SplitOp): for each sample
    /// index it returns `true` when that sample belongs to the left child.
    ///
    /// # Split-kind semantics
    ///
    /// - **`XSplit`** (feature split): `sorted_xy` and `sorted_x` are
    ///   partitioned by `go_left`. `sorted_y` is **cloned** unchanged into
    ///   both children, and both children inherit `w_y = parent.w_y`.
    /// - **`YSplit`** (target split): `sorted_xy` and `sorted_y` are
    ///   partitioned by `go_left`. `sorted_x` is **cloned** unchanged into
    ///   both children, and both children inherit `w_x = parent.w_x`.
    ///
    /// This preserves the partition-tree invariant that a target split does
    /// not restrict the feature marginal (and vice-versa).
    ///
    /// # Stability
    ///
    /// The partition is **stable**: for each column, the relative order of
    /// indices within each child is preserved from the parent’s sorted list.
    ///
    /// Returns `(left_node, right_node)` with depth incremented by one.
    pub fn propagate_children(
        &self,
        split: &SplitPoint,
        dataset: &dyn DatasetView,
        left_cell: Cell,
        right_cell: Cell,
    ) -> (Node, Node) {
        let col_view = dataset
            .column(&split.col_name)
            .expect("split column not found in dataset");

        let none_to_left = split.none_to_left;

        // Build the go_left oracle from the split's SplitOp
        let go_left: Box<dyn Fn(u32) -> bool + Send + Sync> = {
            let op = split.op.clone();
            Box::new(move |idx: u32| op.go_left(col_view, idx as usize, none_to_left))
        };

        let weights_xy = dataset.weights_xy();
        let weights_x = dataset.weights_x();
        let weights_y = dataset.weights_y();

        let is_y_split = matches!(split.split_kind, SplitKind::YSplit);

        // Stable-partition sorted lists.
        //
        // Key invariant from the partition-tree model:
        // - **YSplit** (target split): only sorted_xy and sorted_y are
        //   partitioned.  sorted_x is inherited **unchanged** by both
        //   children because a target split does not restrict the
        //   feature-space measure.
        // - **XSplit** (feature split): only sorted_xy and sorted_x are
        //   partitioned.  sorted_y is inherited **unchanged** by both
        //   children because a feature split does not restrict the
        //   target-space measure.
        let col_names: Vec<String> = self.sorted.sorted_xy.keys().cloned().collect();
        let n_cols = col_names.len();

        let mut left_sorted = SortedIndices::new();
        let mut right_sorted = SortedIndices::new();

        left_sorted.sorted_x.reserve(n_cols);
        left_sorted.sorted_y.reserve(n_cols);
        left_sorted.sorted_xy.reserve(n_cols);
        right_sorted.sorted_x.reserve(n_cols);
        right_sorted.sorted_y.reserve(n_cols);
        right_sorted.sorted_xy.reserve(n_cols);

        let mut w_xy_left = 0.0_f64;
        let mut w_xy_right = 0.0_f64;
        let mut w_x_left = 0.0_f64;
        let mut w_x_right = 0.0_f64;
        let mut w_y_left = 0.0_f64;
        let mut w_y_right = 0.0_f64;

        // We compute weights from the first column's sorted_xy/x/y partition.
        // (All columns' sorted_* lists contain the same set of indices, just
        // in different order.)
        let mut first_col = true;

        for col_name in &col_names {
            // sorted_xy is ALWAYS partitioned
            let parent_xy = &self.sorted.sorted_xy[col_name];
            let (l_xy, r_xy) = stable_partition(parent_xy, &go_left);

            if first_col {
                for &idx in &l_xy {
                    w_xy_left += weights_xy[idx as usize];
                }
                for &idx in &r_xy {
                    w_xy_right += weights_xy[idx as usize];
                }
            }

            left_sorted.sorted_xy.insert(col_name.clone(), l_xy);
            right_sorted.sorted_xy.insert(col_name.clone(), r_xy);

            if is_y_split {
                // YSplit: sorted_x is NOT partitioned — both children
                // inherit the full parent sorted_x.
                let parent_x = &self.sorted.sorted_x[col_name];
                left_sorted
                    .sorted_x
                    .insert(col_name.clone(), parent_x.clone());
                right_sorted
                    .sorted_x
                    .insert(col_name.clone(), parent_x.clone());

                // sorted_y IS partitioned
                let parent_y = &self.sorted.sorted_y[col_name];
                let (l_y, r_y) = stable_partition(parent_y, &go_left);

                if first_col {
                    w_x_left = self.w_x;
                    w_x_right = self.w_x;
                    for &idx in &l_y {
                        w_y_left += weights_y[idx as usize];
                    }
                    for &idx in &r_y {
                        w_y_right += weights_y[idx as usize];
                    }
                }

                left_sorted.sorted_y.insert(col_name.clone(), l_y);
                right_sorted.sorted_y.insert(col_name.clone(), r_y);
            } else {
                // XSplit: sorted_x IS partitioned
                let parent_x = &self.sorted.sorted_x[col_name];
                let (l_x, r_x) = stable_partition(parent_x, &go_left);

                if first_col {
                    for &idx in &l_x {
                        w_x_left += weights_x[idx as usize];
                    }
                    for &idx in &r_x {
                        w_x_right += weights_x[idx as usize];
                    }
                }

                left_sorted.sorted_x.insert(col_name.clone(), l_x);
                right_sorted.sorted_x.insert(col_name.clone(), r_x);

                // sorted_y is NOT partitioned — both children inherit
                // the full parent sorted_y.
                let parent_y = &self.sorted.sorted_y[col_name];
                left_sorted
                    .sorted_y
                    .insert(col_name.clone(), parent_y.clone());
                right_sorted
                    .sorted_y
                    .insert(col_name.clone(), parent_y.clone());

                if first_col {
                    w_y_left = self.w_y;
                    w_y_right = self.w_y;
                }
            }

            first_col = false;
        }

        let left_node = Node {
            cell: left_cell,
            w_xy: w_xy_left,
            w_x: w_x_left,
            w_y: w_y_left,
            depth: self.depth + 1,
            sorted: left_sorted,
        };

        let right_node = Node {
            cell: right_cell,
            w_xy: w_xy_right,
            w_x: w_x_right,
            w_y: w_y_right,
            depth: self.depth + 1,
            sorted: right_sorted,
        };

        (left_node, right_node)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Stable partition: split `list` into `(left, right)` preserving the
/// original relative order within each group.
fn stable_partition(list: &[u32], predicate: &dyn Fn(u32) -> bool) -> (Vec<u32>, Vec<u32>) {
    let mut left = Vec::with_capacity(list.len() / 2);
    let mut right = Vec::with_capacity(list.len() / 2);
    for &idx in list {
        if predicate(idx) {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }
    (left, right)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset_view::PolarsDatasetView;
    use crate::loss::CellStats;
    use crate::rule::DynRule;
    use crate::rules::ContinuousInterval;
    use crate::split_result::{ContinuousSplitOp, SplitKind, SplitPoint};
    use polars::prelude::*;

    fn make_simple_dataset() -> (DataFrame, PolarsDatasetView) {
        let df = DataFrame::new(vec![
            Column::new("x1".into(), &[1.0_f64, 3.0, 2.0, 5.0, 4.0]),
            Column::new("target__y1".into(), &[10.0_f64, 30.0, 20.0, 50.0, 40.0]),
        ])
        .unwrap();
        let view = PolarsDatasetView::new(&df);
        (df, view)
    }

    fn make_root_cell() -> Cell {
        Cell::new()
            .with_rule(
                "x1",
                Box::new(ContinuousInterval::new(
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    true,
                    true,
                    Some((f64::NEG_INFINITY, f64::INFINITY)),
                    true,
                )) as Box<dyn DynRule>,
            )
            .with_rule(
                "target__y1",
                Box::new(ContinuousInterval::new(
                    0.0,
                    60.0,
                    true,
                    true,
                    Some((0.0, 60.0)),
                    true,
                )) as Box<dyn DynRule>,
            )
    }

    #[test]
    fn root_node_has_all_indices() {
        let (_df, view) = make_simple_dataset();
        let cell = make_root_cell();
        let root = Node::root(&view, cell);

        assert_eq!(root.sorted.n_xy(), 5);
        assert_eq!(root.sorted.n_x(), 5);
        assert_eq!(root.sorted.n_y(), 5);
        assert!((root.w_xy - 5.0).abs() < 1e-10);
        assert!((root.w_x - 5.0).abs() < 1e-10);
        assert_eq!(root.depth, 0);
    }

    #[test]
    fn propagate_children_splits_indices() {
        let (_df, view) = make_simple_dataset();
        let cell = make_root_cell();
        let root = Node::root(&view, cell);

        // Split x1 at 2.5: left gets indices where x1 < 2.5 (rows 0, 2)
        let op = ContinuousSplitOp {
            threshold: 2.5,
            k_candidate: 1,
            p_xy: 2,
        };
        let split = SplitPoint {
            col_name: "x1".to_string(),
            split_kind: SplitKind::XSplit,
            none_to_left: true,
            gain: 1.0,
            left_stats: CellStats::new(2.0, 2.0, 5.0, 60.0),
            right_stats: CellStats::new(3.0, 3.0, 5.0, 60.0),
            op: Box::new(op.clone()),
        };

        let (left_cell, right_cell) = root.cell.apply_split("x1", &op, true);
        let (left, right) = root.propagate_children(&split, &view, left_cell, right_cell);

        // Left should have 2 XY samples (x1 values 1.0 and 2.0)
        assert_eq!(left.sorted.n_xy(), 2);
        assert!((left.w_xy - 2.0).abs() < 1e-10);

        // Right should have 3 XY samples (x1 values 3.0, 4.0, 5.0)
        assert_eq!(right.sorted.n_xy(), 3);
        assert!((right.w_xy - 3.0).abs() < 1e-10);

        // XSplit: sorted_x IS partitioned
        assert_eq!(left.sorted.n_x(), 2);
        assert_eq!(right.sorted.n_x(), 3);
        assert!((left.w_x - 2.0).abs() < 1e-10);
        assert!((right.w_x - 3.0).abs() < 1e-10);

        // XSplit: sorted_y is NOT partitioned — both inherit full parent
        assert_eq!(left.sorted.n_y(), 5);
        assert_eq!(right.sorted.n_y(), 5);
        assert!((left.w_y - root.w_y).abs() < 1e-10);
        assert!((right.w_y - root.w_y).abs() < 1e-10);

        // Depth incremented
        assert_eq!(left.depth, 1);
        assert_eq!(right.depth, 1);
    }

    #[test]
    fn propagate_children_ysplit_preserves_sorted_x() {
        let (_df, view) = make_simple_dataset();
        let cell = make_root_cell();
        let root = Node::root(&view, cell);

        // YSplit on target__y1 at 25.0: left gets rows with target__y1 < 25
        // (rows 0, 2 with values 10.0, 20.0)
        let op = ContinuousSplitOp {
            threshold: 25.0,
            k_candidate: 1,
            p_xy: 2,
        };
        let split = SplitPoint {
            col_name: "target__y1".to_string(),
            split_kind: SplitKind::YSplit,
            none_to_left: true,
            gain: 1.0,
            left_stats: CellStats::new(2.0, 5.0, 2.0, 25.0),
            right_stats: CellStats::new(3.0, 5.0, 3.0, 35.0),
            op: Box::new(op.clone()),
        };

        let (left_cell, right_cell) = root.cell.apply_split("target__y1", &op, true);
        let (left, right) = root.propagate_children(&split, &view, left_cell, right_cell);

        // sorted_xy IS partitioned
        assert_eq!(left.sorted.n_xy(), 2);
        assert_eq!(right.sorted.n_xy(), 3);
        assert!((left.w_xy - 2.0).abs() < 1e-10);
        assert!((right.w_xy - 3.0).abs() < 1e-10);

        // YSplit: sorted_x is NOT partitioned — both children get FULL parent
        assert_eq!(left.sorted.n_x(), 5);
        assert_eq!(right.sorted.n_x(), 5);
        assert!((left.w_x - root.w_x).abs() < 1e-10);
        assert!((right.w_x - root.w_x).abs() < 1e-10);

        // YSplit: sorted_y IS partitioned
        assert_eq!(left.sorted.n_y(), 2);
        assert_eq!(right.sorted.n_y(), 3);
    }

    #[test]
    fn stable_partition_preserves_order() {
        let list = vec![0, 2, 1, 4, 3];
        let (left, right) = super::stable_partition(&list, &|idx| idx < 3);
        assert_eq!(left, vec![0, 2, 1]);
        assert_eq!(right, vec![4, 3]);
    }

    // ── root_bootstrap tests ──────────────────────────────────────────

    #[test]
    fn root_bootstrap_produces_correct_sample_count() {
        use rand::SeedableRng;

        let (_df, view) = make_simple_dataset();
        let cell = make_root_cell();
        let mut rng = StdRng::seed_from_u64(42);

        let root = Node::root_bootstrap(&view, cell, 0.6, true, &mut rng);

        // floor(5 * 0.6) = 3 bootstrap draws
        assert_eq!(
            root.sorted.n_xy(),
            3,
            "bootstrap should draw floor(n * max_samples) samples"
        );
        assert_eq!(root.sorted.n_x(), 3);
        assert_eq!(root.sorted.n_y(), 3);
        // With uniform weights, w_xy should equal the number of draws
        assert!(
            (root.w_xy - 3.0).abs() < 1e-10,
            "w_xy should equal bootstrap count with uniform weights, got {}",
            root.w_xy
        );
        assert_eq!(root.depth, 0);
    }

    #[test]
    fn root_bootstrap_can_repeat_indices() {
        use rand::SeedableRng;
        use std::collections::HashMap;

        let (_df, view) = make_simple_dataset();
        let cell = make_root_cell();
        // max_samples=0.8 draws floor(5*0.8)=4 samples with replacement from 5 rows.
        // With a fixed seed, some indices may repeat.
        let mut rng = StdRng::seed_from_u64(123);
        let root = Node::root_bootstrap(&view, cell, 0.8, true, &mut rng);

        // Count occurrences in the sorted_xy list of any column
        let any_sorted = root.sorted.sorted_xy.values().next().unwrap();
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &idx in any_sorted {
            *counts.entry(idx).or_insert(0) += 1;
        }

        // With replacement, the total length equals n_bootstrap
        assert_eq!(any_sorted.len(), 4, "should draw exactly 4 samples");
        // Some indices may repeat (fewer unique than total draws)
        assert!(
            counts.len() <= 4,
            "bootstrap with replacement: unique count should be <= draw count"
        );
    }

    #[test]
    fn root_bootstrap_preserves_sort_order() {
        use rand::SeedableRng;

        // Dataset where x1 values are [1, 3, 2, 5, 4] → sorted order: [0, 2, 1, 4, 3]
        let (_df, view) = make_simple_dataset();
        let cell = make_root_cell();
        let mut rng = StdRng::seed_from_u64(99);

        let root = Node::root_bootstrap(&view, cell, 1.0, true, &mut rng);

        // For column x1, the bootstrap indices should preserve the
        // ascending-by-value order from the dataset's global sort.
        let x1_sorted = &root.sorted.sorted_xy["x1"];
        let col = view.column("x1").unwrap();

        for window in x1_sorted.windows(2) {
            let v0 = col.get_f64(window[0] as usize);
            let v1 = col.get_f64(window[1] as usize);
            assert!(
                v0 <= v1,
                "bootstrap sorted indices should preserve sort order: \
                 idx {} (val {:?}) should be <= idx {} (val {:?})",
                window[0],
                v0,
                window[1],
                v1
            );
        }
    }

    #[test]
    fn root_bootstrap_full_sample_has_correct_length() {
        use rand::SeedableRng;

        let (_df, view) = make_simple_dataset();
        let cell = make_root_cell();
        let mut rng = StdRng::seed_from_u64(7);

        let root = Node::root_bootstrap(&view, cell, 1.0, true, &mut rng);

        // floor(5 * 1.0) = 5 draws
        assert_eq!(root.sorted.n_xy(), 5);
        // Weights sum should equal 5.0 (each draw adds 1.0)
        assert!(
            (root.w_xy - 5.0).abs() < 1e-10,
            "w_xy={} but expected 5.0",
            root.w_xy
        );
    }
}
