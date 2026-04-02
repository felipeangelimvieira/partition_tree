//! Fitted partition tree.
//!
//! After construction by [`TreeBuilder`](super::tree_builder::TreeBuilder), the
//! tree is stored as an arena of [`FittedNode`]s (cell + weights + tree
//! pointers) together with a list of leaf indices and a split history.
//!
//! The builder’s temporary [`SortedIndices`](super::node::SortedIndices) are
//! discarded—only the partition constraints remain, keeping the fitted tree
//! compact.
//!
//! ## Prediction
//!
//! Two prediction APIs are provided:
//!
//! - [`Tree::predict_leaf_from_map`] — single-sample prediction from
//!   `HashMap` inputs (useful for testing / debugging).
//! - [`Tree::predict_leaves`] — batch prediction over a full
//!   [`DatasetView`].
//!
//! Both walk the tree from root, evaluating left-child rules at each
//! internal node.
//!
//! ## Display
//!
//! `Tree` implements [`fmt::Display`] to print a summary of leaf statistics.
use std::collections::HashMap;
use std::fmt;

use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::rules::{BelongsTo, ContinuousInterval, IntegerInterval};

use crate::cell::Cell;
use crate::dataset_view::{ColumnView, DatasetView, LogicalDType};
use crate::predict::conditioned_cell::ConditionedCell;
use crate::predict::piecewise_distribution::{MeanVector, PiecewiseConstantDistribution};
use crate::rule::{DynRule, DynValue};
use crate::split_result::SplitKind;

// ---------------------------------------------------------------------------
// FittedNode
// ---------------------------------------------------------------------------

/// A node in the fitted tree.
///
/// Contains the partition constraint ([`Cell`]), aggregated weights, tree
/// structure (parent/child indices), and a leaf flag. No training indices
/// or sorted lists are retained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedNode {
    /// The multi-dimensional partition constraint.
    pub cell: Cell,
    /// Aggregated weight of XY samples in this node.
    pub w_xy: f64,
    /// Aggregated weight of X samples.
    pub w_x: f64,
    /// Aggregated weight of Y samples.
    pub w_y: f64,
    /// Depth of this node in the tree.
    pub depth: usize,
    /// Parent node index (None for root).
    pub parent: Option<usize>,
    /// Left child node index.
    pub left_child: Option<usize>,
    /// Right child node index.
    pub right_child: Option<usize>,
    /// Whether this is a leaf node.
    pub is_leaf: bool,
    /// Column on which this node was split (`None` for leaves).
    pub split_col: Option<String>,
    /// Whether the split refines X or Y (`None` for leaves).
    pub split_kind: Option<SplitKind>,
}

impl FittedNode {
    /// Conditional density estimate: $\hat{f}(y|x) = w_{xy} / (w_x \cdot V_{\text{target}})$.
    ///
    /// Returns `0.0` when `w_x` or the target volume is non-positive.
    pub fn conditional_density(&self) -> f64 {
        let vol = self.cell.target_volume();
        if self.w_x <= 0.0 || vol <= 0.0 {
            0.0
        } else {
            self.w_xy / (self.w_x * vol)
        }
    }

    /// Balanced weight: $w_{xy} / (w_x \cdot w_y)$.
    ///
    /// Used when the Y-measure is the counting measure instead of
    /// geometric volume.
    pub fn balanced_weight(&self) -> f64 {
        if self.w_x <= 0.0 || self.w_y <= 0.0 {
            0.0
        } else {
            self.w_xy / (self.w_x * self.w_y)
        }
    }
}

// ---------------------------------------------------------------------------
// SplitRecord
// ---------------------------------------------------------------------------

/// Record of a single split made during tree construction.
///
/// Stored in [`Tree::split_history`] in the order splits were executed
/// (i.e., best-first / highest-gain first).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRecord {
    /// Index of the parent node that was split.
    pub parent_index: usize,
    /// Column that was split.
    pub col_name: String,
    /// Whether this was a feature or target split.
    pub split_kind: SplitKind,
    /// Information gain of the split.
    pub gain: f64,
    /// Index of the left child node.
    pub left_child_index: usize,
    /// Index of the right child node.
    pub right_child_index: usize,
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

/// A fitted partition tree.
///
/// The tree is stored as an arena of [`FittedNode`]s indexed by `usize`.
/// Traversal starts at index `0` (the root). Leaf indices are cached in
/// [`leaves`](Tree::leaves) for fast iteration.
///
/// # Invariants
///
/// - `nodes[0]` is always the root.
/// - Every non-leaf node has both `left_child` and `right_child` set.
/// - `leaves` contains exactly the indices where `is_leaf == true`.
#[derive(Serialize, Deserialize)]
pub struct Tree {
    /// Arena of fitted nodes.
    pub nodes: Vec<FittedNode>,
    /// Indices of leaf nodes.
    pub leaves: Vec<usize>,
    /// History of splits made during building.
    pub split_history: Vec<SplitRecord>,
}

impl Tree {
    /// Number of leaf nodes.
    pub fn n_leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Total number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Maximum depth of the tree.
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Get the root node.
    pub fn root(&self) -> &FittedNode {
        &self.nodes[0]
    }

    /// Get a node by index.
    pub fn node(&self, index: usize) -> &FittedNode {
        &self.nodes[index]
    }

    /// Get all leaf nodes.
    pub fn leaf_nodes(&self) -> Vec<&FittedNode> {
        self.leaves.iter().map(|&i| &self.nodes[i]).collect()
    }

    /// Predict the leaf index for a single sample.
    ///
    /// `row` maps continuous column names to optional `f64` values;
    /// `cat_row` maps categorical column names to optional `usize` codes.
    ///
    /// Traverses from root, evaluating the left child’s cell rules at each
    /// internal node. If the sample satisfies all left-child rules it goes
    /// left; otherwise right.
    pub fn predict_leaf_from_map(&self, row: &HashMap<String, Option<DynValue>>) -> usize {
        let mut idx = 0;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf {
                return idx;
            }

            let left_idx = node.left_child.unwrap();
            let left_node = &self.nodes[left_idx];

            let goes_left = self.evaluate_node_membership(left_node, row);

            if goes_left {
                idx = left_idx;
            } else {
                idx = node.right_child.unwrap();
            }
        }
    }

    /// Check if a sample satisfies all rules of a node's cell.
    fn evaluate_node_membership(
        &self,
        node: &FittedNode,
        row: &HashMap<String, Option<DynValue>>,
    ) -> bool {
        for (col, rule) in &node.cell.rules {
            let value = row.get(col.as_str()).and_then(|v| v.as_ref());
            if !rule.contains(value) {
                return false;
            }
        }
        true
    }

    /// Predict leaf indices for all rows in a dataset.
    ///
    /// Returns a `Vec<usize>` of length `dataset.n_rows()`, where each
    /// element is the index of the leaf node the corresponding row falls into.
    pub fn predict_leaves(&self, dataset: &dyn DatasetView) -> Vec<usize> {
        let n_rows = dataset.n_rows();
        let columns = dataset.columns();
        let mut result = Vec::with_capacity(n_rows);

        for row_idx in 0..n_rows {
            let leaf_idx = self.predict_leaf_row(row_idx, &columns);
            result.push(leaf_idx);
        }

        result
    }

    /// Predict leaf index for a single row by column views.
    fn predict_leaf_row(&self, row_idx: usize, columns: &[&dyn ColumnView]) -> usize {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return node_idx;
            }

            let left_idx = node.left_child.unwrap();
            let left_node = &self.nodes[left_idx];

            let goes_left = self.evaluate_row_membership(left_node, row_idx, columns);

            if goes_left {
                node_idx = left_idx;
            } else {
                node_idx = node.right_child.unwrap();
            }
        }
    }

    /// Check if row_idx satisfies all rules of a node's cell.
    fn evaluate_row_membership(
        &self,
        node: &FittedNode,
        row_idx: usize,
        columns: &[&dyn ColumnView],
    ) -> bool {
        for (col_name, rule) in &node.cell.rules {
            let col = match columns.iter().find(|c| c.name() == col_name) {
                Some(c) => c,
                None => continue,
            };

            let value = col.get_dyn_value(row_idx);
            if !rule.contains(value.as_ref()) {
                return false;
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Distribution-based prediction
    // -----------------------------------------------------------------------

    /// Predict conditional distributions for all rows in a dataset.
    ///
    /// 1. Pre-builds a column lookup map and all leaf [`ConditionedCell`]s once.
    /// 2. Matches each row to leaves whose feature constraints contain it via
    ///    [`walk_leaves`](Tree::walk_leaves) (rows processed in parallel).
    /// 3. Wraps matched cells in a [`PiecewiseConstantDistribution`].
    ///
    /// Returns a `Vec` of length `dataset.n_rows()`.
    pub fn predict_distributions(
        &self,
        dataset: &dyn DatasetView,
    ) -> Vec<PiecewiseConstantDistribution> {
        let feature_columns = dataset.feature_columns();
        let all_columns = dataset.columns();

        // Build column lookup once — reused for every row.
        let col_map: HashMap<&str, &dyn ColumnView> =
            feature_columns.iter().map(|c| (c.name(), *c)).collect();

        // Pre-build ConditionedCell for every leaf once — no per-row allocation.
        let leaf_cells: HashMap<usize, ConditionedCell> = self
            .leaves
            .iter()
            .map(|&idx| (idx, ConditionedCell::from_fitted_node(&self.nodes[idx])))
            .collect();

        (0..dataset.n_rows())
            .into_par_iter()
            .map(|row_idx| {
                // Conditional prediction P(Y|X=x): match *all* leaves whose
                // feature-space constraints contain x (ignoring target rules).
                let mut matched = self.walk_leaves(row_idx, &col_map);

                // Defensive fallback: if no leaf matched by features, fall back
                // to deterministic traversal over all columns.
                if matched.is_empty() {
                    matched.push(self.predict_leaf_row(row_idx, &all_columns));
                }

                let cells = matched
                    .into_iter()
                    .map(|leaf_idx| leaf_cells[&leaf_idx].clone())
                    .collect();

                PiecewiseConstantDistribution::from_cells(cells)
            })
            .collect()
    }

    /// Match all leaf node indices whose feature constraints contain row `row_idx`.
    ///
    /// Uses an iterative root-to-leaf walk instead of scanning every leaf.
    /// At each internal node the split column is checked:
    ///
    /// - **Column present** in `columns` → evaluate the left child's rule for
    ///   that column and descend left or right accordingly.
    /// - **Column absent** (target split, or feature not in the query) →
    ///   descend into **both** children.
    ///
    /// Complexity: O(depth) when all split columns are present; degrades
    /// towards O(n_nodes) only when many splits are on absent columns.
    #[cfg_attr(not(test), allow(dead_code))]
    fn match_leaves_given_features(
        &self,
        row_idx: usize,
        feature_columns: &[&dyn ColumnView],
    ) -> Vec<usize> {
        // Build O(1) lookup from column name → ColumnView
        let col_map: HashMap<&str, &dyn ColumnView> =
            feature_columns.iter().map(|c| (c.name(), *c)).collect();

        self.walk_leaves(row_idx, &col_map)
    }

    /// Iterative DFS walk collecting all reachable leaves for a single row.
    ///
    /// `col_map` maps column names to their [`ColumnView`]s.  When the split
    /// column of an internal node is absent from `col_map`, both children are
    /// explored.  When it is present, only the matching child is visited.
    fn walk_leaves(&self, row_idx: usize, col_map: &HashMap<&str, &dyn ColumnView>) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = Vec::with_capacity(self.depth() + 1);
        stack.push(0_usize); // root

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            if node.is_leaf {
                result.push(node_idx);
                continue;
            }

            let left_idx = node.left_child.unwrap();
            let right_idx = node.right_child.unwrap();

            match node.split_col.as_deref() {
                Some(split_col) if col_map.contains_key(split_col) => {
                    // Column is available — evaluate left child's rule for this
                    // single column and route deterministically.
                    let left_node = &self.nodes[left_idx];
                    if self.evaluate_split_column(left_node, row_idx, split_col, col_map) {
                        stack.push(left_idx);
                    } else {
                        stack.push(right_idx);
                    }
                }
                _ => {
                    // Split column absent (target split or missing feature) —
                    // explore both subtrees.
                    stack.push(right_idx);
                    stack.push(left_idx);
                }
            }
        }

        result
    }

    /// Check whether `row_idx` satisfies the rule for `split_col` in `node`'s cell.
    fn evaluate_split_column(
        &self,
        node: &FittedNode,
        row_idx: usize,
        split_col: &str,
        col_map: &HashMap<&str, &dyn ColumnView>,
    ) -> bool {
        let rule = match node.cell.get_rule(split_col) {
            Some(r) => r,
            None => return true, // no rule ⇒ unconstrained
        };
        let col = match col_map.get(split_col) {
            Some(c) => *c,
            None => return true, // column absent ⇒ unconstrained
        };
        let value = col.get_dyn_value(row_idx);
        rule.contains(value.as_ref())
    }

    /// Predict mean vectors for all rows.
    ///
    /// Shorthand for [`predict_distributions`](Tree::predict_distributions)
    /// followed by [`mean_vector`](PiecewiseConstantDistribution::mean_vector)
    /// on each distribution.
    pub fn predict_mean_vectors(&self, dataset: &dyn DatasetView) -> Vec<MeanVector> {
        self.predict_distributions(dataset)
            .into_iter()
            .map(|d| d.mean_vector())
            .collect()
    }

    /// Predict means as a Polars `DataFrame`.
    ///
    /// | Target dtype   | Column type | Conversion                              |
    /// |----------------|-------------|-----------------------------------------|
    /// | Continuous     | `Float64`   | Direct midpoint                         |
    /// | Integer        | `Float64`   | Midpoint (as f64)                       |
    /// | Categorical    | `Utf8`      | Argmax category name                    |
    ///
    /// The output has `n_rows` rows and one column per target dimension.
    pub fn predict_mean(&self, dataset: &dyn DatasetView) -> DataFrame {
        let mean_vectors = self.predict_mean_vectors(dataset);

        // Discover target columns from root cell
        let root = self.root();
        let mut target_cols: Vec<(&String, &dyn DynRule)> = root.cell.target_rules().collect();
        // Sort for deterministic column order
        target_cols.sort_by_key(|(k, _)| (*k).clone());

        let mut columns: Vec<Column> = Vec::with_capacity(target_cols.len());

        for (col_name, rule) in &target_cols {
            // Determine dtype from the root rule
            if rule.as_any().downcast_ref::<ContinuousInterval>().is_some() {
                // Continuous target → Float64 column
                let values: Vec<f64> = mean_vectors
                    .iter()
                    .map(|mv| {
                        mv.get(col_name.as_str())
                            .and_then(|v| v.first().copied())
                            .unwrap_or(f64::NAN)
                    })
                    .collect();
                let series = Series::new(PlSmallStr::from_str(col_name), values);
                columns.push(series.into());
            } else if rule.as_any().downcast_ref::<IntegerInterval>().is_some() {
                // Integer target → Float64 column (midpoint as f64)
                let values: Vec<f64> = mean_vectors
                    .iter()
                    .map(|mv| {
                        mv.get(col_name.as_str())
                            .and_then(|v| v.first().copied())
                            .unwrap_or(f64::NAN)
                    })
                    .collect();
                let series = Series::new(PlSmallStr::from_str(col_name), values);
                columns.push(series.into());
            } else if let Some(bt) = rule.as_any().downcast_ref::<BelongsTo>() {
                // Categorical target → Utf8 column (argmax category name)
                let domain_names = bt.domain_names.as_ref();
                let values: Vec<String> = mean_vectors
                    .iter()
                    .map(|mv| {
                        let probs = mv.get(col_name.as_str());
                        match probs {
                            Some(p) if !p.is_empty() => {
                                let argmax = p
                                    .iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| {
                                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(i, _)| i)
                                    .unwrap_or(0);
                                domain_names
                                    .get(argmax)
                                    .cloned()
                                    .unwrap_or_else(|| format!("unknown_{argmax}"))
                            }
                            _ => "unknown".to_string(),
                        }
                    })
                    .collect();
                let series = Series::new(PlSmallStr::from_str(col_name), values);
                columns.push(series.into());
            }
        }

        if columns.is_empty() {
            // No target columns — return empty DataFrame with correct height
            DataFrame::empty()
        } else {
            DataFrame::new(columns).expect("predict_mean: DataFrame construction failed")
        }
    }

    // -----------------------------------------------------------------------
    // Feature importances
    // -----------------------------------------------------------------------

    /// Compute feature importances based on cumulative gain from all splits.
    ///
    /// Each split is counted exactly once (no double-counting across leaves).
    /// If `normalize` is true, the importances are normalized to sum to 1.0.
    pub fn feature_importances(&self, normalize: bool) -> HashMap<String, f64> {
        let mut importances: HashMap<String, f64> = HashMap::new();

        for rec in &self.split_history {
            if rec.gain.is_finite() && rec.gain > 0.0 {
                *importances.entry(rec.col_name.clone()).or_insert(0.0) += rec.gain;
            }
        }

        if normalize && !importances.is_empty() {
            let total: f64 = importances.values().sum();
            if total > 0.0 {
                for value in importances.values_mut() {
                    *value /= total;
                }
            }
        }

        importances
    }

    // -----------------------------------------------------------------------
    // Apply (leaf assignment)
    // -----------------------------------------------------------------------

    /// Apply the tree to a dataset, returning the leaf index for each row.
    ///
    /// Returns a `Vec<usize>` of length `dataset.n_rows()`, where each
    /// element is the index of the leaf node the row was routed to.
    /// These are node indices (not positions in the `leaves` array).
    pub fn apply(&self, dataset: &dyn DatasetView) -> Vec<usize> {
        self.predict_leaves(dataset)
    }
    /// Get target column metadata from the root cell.
    ///
    /// Returns a list of `(column_name, logical_dtype)` pairs for each
    /// `target__`-prefixed rule in the root cell.
    pub fn target_schema(&self) -> Vec<(String, LogicalDType)> {
        let root = self.root();
        let mut schema = Vec::new();
        for (name, rule) in root.cell.target_rules() {
            let dtype = if rule.as_any().downcast_ref::<ContinuousInterval>().is_some() {
                LogicalDType::Continuous
            } else if rule.as_any().downcast_ref::<IntegerInterval>().is_some() {
                LogicalDType::Integer
            } else if rule.as_any().downcast_ref::<BelongsTo>().is_some() {
                LogicalDType::Categorical
            } else {
                LogicalDType::Continuous // fallback
            };
            schema.push((name.clone(), dtype));
        }
        schema.sort_by_key(|(k, _)| k.clone());
        schema
    }

    /// Get leaf information for display/debugging.
    pub fn leaf_info(&self) -> Vec<LeafInfo> {
        self.leaves
            .iter()
            .map(|&idx| {
                let node = &self.nodes[idx];
                LeafInfo {
                    index: idx,
                    depth: node.depth,
                    w_xy: node.w_xy,
                    w_x: node.w_x,
                    w_y: node.w_y,
                    conditional_density: node.conditional_density(),
                    target_volume: node.cell.target_volume(),
                }
            })
            .collect()
    }
}

/// Summary statistics for a single leaf node.
///
/// Returned by [`Tree::leaf_info`] for display and debugging.
#[derive(Debug, Clone)]
pub struct LeafInfo {
    pub index: usize,
    pub depth: usize,
    pub w_xy: f64,
    pub w_x: f64,
    pub w_y: f64,
    pub conditional_density: f64,
    pub target_volume: f64,
}

impl fmt::Display for LeafInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Leaf[{}] depth={} w_xy={:.2} w_x={:.2} w_y={:.2} density={:.6} vol={:.4}",
            self.index,
            self.depth,
            self.w_xy,
            self.w_x,
            self.w_y,
            self.conditional_density,
            self.target_volume,
        )
    }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Tree(nodes={}, leaves={}, depth={})",
            self.n_nodes(),
            self.n_leaves(),
            self.depth()
        )?;
        for info in self.leaf_info() {
            writeln!(f, "  {info}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::dataset_view::PolarsDatasetView;
    use crate::rule::{DynRule, DynValue};
    use crate::rules::ContinuousInterval;

    fn make_simple_tree() -> Tree {
        // Root → Left (leaf), Right (leaf)
        let root_cell = Cell::new().with_rule(
            "x",
            Box::new(ContinuousInterval::new(
                0.0,
                10.0,
                true,
                true,
                Some((0.0, 10.0)),
                true,
            )) as Box<dyn DynRule>,
        );

        let left_cell = Cell::new().with_rule(
            "x",
            Box::new(ContinuousInterval::new(
                0.0,
                5.0,
                true,
                false,
                Some((0.0, 10.0)),
                true,
            )) as Box<dyn DynRule>,
        );

        let right_cell = Cell::new().with_rule(
            "x",
            Box::new(ContinuousInterval::new(
                5.0,
                10.0,
                true,
                true,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>,
        );

        let root = FittedNode {
            cell: root_cell,
            w_xy: 100.0,
            w_x: 100.0,
            w_y: 100.0,
            depth: 0,
            parent: None,
            left_child: Some(1),
            right_child: Some(2),
            is_leaf: false,
            split_col: Some("x".to_string()),
            split_kind: Some(SplitKind::XSplit),
        };

        let left = FittedNode {
            cell: left_cell,
            w_xy: 60.0,
            w_x: 60.0,
            w_y: 100.0,
            depth: 1,
            parent: Some(0),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };

        let right = FittedNode {
            cell: right_cell,
            w_xy: 40.0,
            w_x: 40.0,
            w_y: 100.0,
            depth: 1,
            parent: Some(0),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };

        Tree {
            nodes: vec![root, left, right],
            leaves: vec![1, 2],
            split_history: vec![],
        }
    }

    #[test]
    fn tree_basic_properties() {
        let tree = make_simple_tree();
        assert_eq!(tree.n_nodes(), 3);
        assert_eq!(tree.n_leaves(), 2);
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn predict_leaf_from_map_routes_correctly() {
        let tree = make_simple_tree();

        // x=3.0 should go to left (x < 5)
        let mut row = HashMap::new();
        row.insert("x".to_string(), Some(DynValue::Continuous(3.0)));
        assert_eq!(tree.predict_leaf_from_map(&row), 1);

        // x=7.0 should go to right (x >= 5)
        row.insert("x".to_string(), Some(DynValue::Continuous(7.0)));
        assert_eq!(tree.predict_leaf_from_map(&row), 2);
    }

    #[test]
    fn predict_leaf_null_routing() {
        let tree = make_simple_tree();

        // None should go to left (accept_none=true on left, false on right)
        let mut row = HashMap::new();
        row.insert("x".to_string(), None);
        assert_eq!(tree.predict_leaf_from_map(&row), 1);
    }

    #[test]
    fn leaf_info_computes_density() {
        let tree = make_simple_tree();
        let infos = tree.leaf_info();
        assert_eq!(infos.len(), 2);

        // Cells only have feature "x", no target columns → target_volume = 1.0
        // Left:  w_xy=60, w_x=60, vol=1.0 → density = 60/(60*1) = 1.0
        assert!((infos[0].conditional_density - 1.0).abs() < 1e-10);
        assert!((infos[0].target_volume - 1.0).abs() < 1e-10);

        // Right: w_xy=40, w_x=40, vol=1.0 → density = 40/(40*1) = 1.0
        assert!((infos[1].conditional_density - 1.0).abs() < 1e-10);
    }

    fn make_tree_with_target_split_then_feature_split() -> Tree {
        let full_x = || {
            Box::new(ContinuousInterval::new(
                0.0,
                10.0,
                true,
                true,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };
        let x_left = || {
            Box::new(ContinuousInterval::new(
                0.0,
                5.0,
                true,
                false,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };
        let x_right = || {
            Box::new(ContinuousInterval::new(
                5.0,
                10.0,
                true,
                true,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };
        let y_low = || {
            Box::new(ContinuousInterval::new(
                0.0,
                5.0,
                true,
                false,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };
        let y_high = || {
            Box::new(ContinuousInterval::new(
                5.0,
                10.0,
                true,
                true,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };

        let root = FittedNode {
            cell: Cell::new().with_rule("x", full_x()).with_rule(
                "target_y",
                Box::new(ContinuousInterval::new(
                    0.0,
                    10.0,
                    true,
                    true,
                    Some((0.0, 10.0)),
                    false,
                )),
            ),
            w_xy: 100.0,
            w_x: 100.0,
            w_y: 100.0,
            depth: 0,
            parent: None,
            left_child: Some(1),
            right_child: Some(2),
            is_leaf: false,
            split_col: Some("target_y".to_string()),
            split_kind: Some(SplitKind::YSplit),
        };

        // Internal nodes after target split
        let target_left = FittedNode {
            cell: Cell::new()
                .with_rule("x", full_x())
                .with_rule("target_y", y_low()),
            w_xy: 50.0,
            w_x: 50.0,
            w_y: 100.0,
            depth: 1,
            parent: Some(0),
            left_child: Some(3),
            right_child: Some(4),
            is_leaf: false,
            split_col: Some("x".to_string()),
            split_kind: Some(SplitKind::XSplit),
        };
        let target_right = FittedNode {
            cell: Cell::new()
                .with_rule("x", full_x())
                .with_rule("target_y", y_high()),
            w_xy: 50.0,
            w_x: 50.0,
            w_y: 100.0,
            depth: 1,
            parent: Some(0),
            left_child: Some(5),
            right_child: Some(6),
            is_leaf: false,
            split_col: Some("x".to_string()),
            split_kind: Some(SplitKind::XSplit),
        };

        // Leaves
        let ll = FittedNode {
            cell: Cell::new()
                .with_rule("x", x_left())
                .with_rule("target_y", y_low()),
            w_xy: 20.0,
            w_x: 25.0,
            w_y: 50.0,
            depth: 2,
            parent: Some(1),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };
        let lr = FittedNode {
            cell: Cell::new()
                .with_rule("x", x_right())
                .with_rule("target_y", y_low()),
            w_xy: 10.0,
            w_x: 25.0,
            w_y: 50.0,
            depth: 2,
            parent: Some(1),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };
        let rl = FittedNode {
            cell: Cell::new()
                .with_rule("x", x_left())
                .with_rule("target_y", y_high()),
            w_xy: 5.0,
            w_x: 25.0,
            w_y: 50.0,
            depth: 2,
            parent: Some(2),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };
        let rr = FittedNode {
            cell: Cell::new()
                .with_rule("x", x_right())
                .with_rule("target_y", y_high()),
            w_xy: 25.0,
            w_x: 25.0,
            w_y: 50.0,
            depth: 2,
            parent: Some(2),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };

        Tree {
            nodes: vec![root, target_left, target_right, ll, lr, rl, rr],
            leaves: vec![3, 4, 5, 6],
            split_history: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Categorical prediction helpers and tests
    // -----------------------------------------------------------------------

    /// Build a tree whose target column is categorical with three classes:
    /// `"cat_A"` (code 0), `"cat_B"` (code 1), `"cat_C"` (code 2).
    ///
    /// Structure:
    /// - Root: x ∈ [0, 10], target_label ∈ {0, 1, 2}
    /// - Left leaf (index 1):  x ∈ [0, 5),  target_label = {0} → "cat_A"
    /// - Right leaf (index 2): x ∈ [5, 10], target_label = {1} → "cat_B"
    ///
    /// Mass:  w_xy / w_x = 9/10 for left, 8/10 for right.
    fn make_categorical_tree() -> Tree {
        use crate::rules::BelongsTo;
        use std::collections::HashSet;
        use std::sync::Arc;

        let domain = Arc::new(vec![0_usize, 1_usize, 2_usize]);
        let domain_names = Arc::new(vec![
            "cat_A".to_string(),
            "cat_B".to_string(),
            "cat_C".to_string(),
        ]);

        let full_x = || {
            Box::new(ContinuousInterval::new(
                0.0,
                10.0,
                true,
                true,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };
        let x_left = || {
            Box::new(ContinuousInterval::new(
                0.0,
                5.0,
                true,
                false,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };
        let x_right = || {
            Box::new(ContinuousInterval::new(
                5.0,
                10.0,
                true,
                true,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };

        // Root spans the full categorical domain {0, 1, 2}
        let full_cat = || {
            Box::new(BelongsTo::new(
                HashSet::from([0, 1, 2]),
                Arc::clone(&domain),
                Arc::clone(&domain_names),
                false,
            )) as Box<dyn DynRule>
        };
        // Left leaf: only "cat_A" (code 0)
        let cat_a = || {
            Box::new(BelongsTo::new(
                HashSet::from([0]),
                Arc::clone(&domain),
                Arc::clone(&domain_names),
                false,
            )) as Box<dyn DynRule>
        };
        // Right leaf: only "cat_B" (code 1)
        let cat_b = || {
            Box::new(BelongsTo::new(
                HashSet::from([1]),
                Arc::clone(&domain),
                Arc::clone(&domain_names),
                false,
            )) as Box<dyn DynRule>
        };

        let root = FittedNode {
            cell: Cell::new()
                .with_rule("x", full_x())
                .with_rule("target_label", full_cat()),
            w_xy: 17.0,
            w_x: 20.0,
            w_y: 17.0,
            depth: 0,
            parent: None,
            left_child: Some(1),
            right_child: Some(2),
            is_leaf: false,
            split_col: Some("x".to_string()),
            split_kind: Some(SplitKind::XSplit),
        };
        let left = FittedNode {
            cell: Cell::new()
                .with_rule("x", x_left())
                .with_rule("target_label", cat_a()),
            w_xy: 9.0,
            w_x: 10.0,
            w_y: 9.0,
            depth: 1,
            parent: Some(0),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };
        let right = FittedNode {
            cell: Cell::new()
                .with_rule("x", x_right())
                .with_rule("target_label", cat_b()),
            w_xy: 8.0,
            w_x: 10.0,
            w_y: 8.0,
            depth: 1,
            parent: Some(0),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };

        Tree {
            nodes: vec![root, left, right],
            leaves: vec![1, 2],
            split_history: vec![],
        }
    }

    /// `predict_mean_vectors` should return a one-hot probability vector over
    /// the full categorical domain for each row, with `1.0` at the index that
    /// corresponds to the leaf's assigned category and `0.0` elsewhere.
    #[test]
    fn predict_mean_vectors_returns_correct_category_probabilities() {
        let tree = make_categorical_tree();

        // x=2.0 routes to the left leaf  → cat_A (code 0) → [1.0, 0.0, 0.0]
        // x=7.0 routes to the right leaf → cat_B (code 1) → [0.0, 1.0, 0.0]
        let x = DataFrame::new(vec![Column::new(
            PlSmallStr::from_static("x"),
            vec![2.0_f64, 7.0_f64],
        )])
        .unwrap();
        let dataset = PolarsDatasetView::new(&x);

        let mean_vecs = tree.predict_mean_vectors(&dataset);
        assert_eq!(mean_vecs.len(), 2, "one vector per row");

        let probs_row0 = mean_vecs[0]
            .get("target_label")
            .expect("target_label must be present");
        let probs_row1 = mean_vecs[1]
            .get("target_label")
            .expect("target_label must be present");

        // Row 0 (x=2): cat_A → index 0 should be 1.0, rest 0.0
        assert_eq!(probs_row0.len(), 3, "domain has 3 categories");
        assert!(
            (probs_row0[0] - 1.0).abs() < 1e-10,
            "cat_A probability must be 1.0, got {}",
            probs_row0[0]
        );
        assert!(probs_row0[1] < 1e-10, "cat_B probability must be 0.0");
        assert!(probs_row0[2] < 1e-10, "cat_C probability must be 0.0");

        // Row 1 (x=7): cat_B → index 1 should be 1.0, rest 0.0
        assert_eq!(probs_row1.len(), 3, "domain has 3 categories");
        assert!(probs_row1[0] < 1e-10, "cat_A probability must be 0.0");
        assert!(
            (probs_row1[1] - 1.0).abs() < 1e-10,
            "cat_B probability must be 1.0, got {}",
            probs_row1[1]
        );
        assert!(probs_row1[2] < 1e-10, "cat_C probability must be 0.0");
    }

    /// `predict_mean` should produce a `DataFrame` column whose string values
    /// are exactly the `domain_names` entries corresponding to argmax of each
    /// row's probability vector—not numeric codes.
    #[test]
    fn predict_mean_maps_argmax_category_to_label_string() {
        let tree = make_categorical_tree();

        // x=2.0 → left leaf → cat_A; x=7.0 → right leaf → cat_B
        let x = DataFrame::new(vec![Column::new(
            PlSmallStr::from_static("x"),
            vec![2.0_f64, 7.0_f64],
        )])
        .unwrap();
        let dataset = PolarsDatasetView::new(&x);

        let df = tree.predict_mean(&dataset);

        let col = df
            .column("target_label")
            .expect("predict_mean must produce a 'target_label' column");
        let utf8 = col.str().expect("categorical column must be Utf8/String");

        assert_eq!(
            utf8.get(0),
            Some("cat_A"),
            "row 0 (x=2): expected 'cat_A', got {:?}",
            utf8.get(0)
        );
        assert_eq!(
            utf8.get(1),
            Some("cat_B"),
            "row 1 (x=7): expected 'cat_B', got {:?}",
            utf8.get(1)
        );
    }

    /// When a leaf covers two categories with equal indicator mass, the
    /// resulting probability vector has equal entries for those categories.
    /// `predict_mean` should resolve ties consistently (last max wins per
    /// `Iterator::max_by`) and return a valid domain label, never a
    /// sentinel like `"unknown"`.
    #[test]
    fn predict_mean_vectors_two_category_leaf_has_equal_probabilities() {
        use crate::rules::BelongsTo;
        use std::collections::HashSet;
        use std::sync::Arc;

        let domain = Arc::new(vec![0_usize, 1_usize, 2_usize]);
        let domain_names = Arc::new(vec![
            "cat_A".to_string(),
            "cat_B".to_string(),
            "cat_C".to_string(),
        ]);

        // A single leaf spanning two categories: {0, 1} → "cat_A" and "cat_B"
        let two_cat = BelongsTo::new(
            HashSet::from([0, 1]),
            Arc::clone(&domain),
            Arc::clone(&domain_names),
            false,
        );

        let root = FittedNode {
            cell: Cell::new()
                .with_rule(
                    "x",
                    Box::new(ContinuousInterval::new(
                        0.0,
                        10.0,
                        true,
                        true,
                        Some((0.0, 10.0)),
                        false,
                    )) as Box<dyn DynRule>,
                )
                .with_rule("target_label", Box::new(two_cat) as Box<dyn DynRule>),
            w_xy: 10.0,
            w_x: 10.0,
            w_y: 10.0,
            depth: 0,
            parent: None,
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };

        let tree = Tree {
            nodes: vec![root],
            leaves: vec![0],
            split_history: vec![],
        };

        let x = DataFrame::new(vec![Column::new(
            PlSmallStr::from_static("x"),
            vec![5.0_f64],
        )])
        .unwrap();
        let dataset = PolarsDatasetView::new(&x);

        let mean_vecs = tree.predict_mean_vectors(&dataset);
        let probs = mean_vecs[0]
            .get("target_label")
            .expect("target_label must be present");

        // Both cat_A (0) and cat_B (1) are in the leaf; cat_C (2) is not.
        // With indicator/volume normalization, each active category gets 1/2 = 0.5
        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 0.5).abs() < 1e-10, "cat_A prob = {}", probs[0]);
        assert!((probs[1] - 0.5).abs() < 1e-10, "cat_B prob = {}", probs[1]);
        assert!(probs[2] < 1e-10, "cat_C prob = {}", probs[2]);

        // `predict_mean` must resolve the tie to a real domain label.
        let df = tree.predict_mean(&dataset);
        let col = df.column("target_label").unwrap();
        let label = col.str().unwrap().get(0).expect("must produce a label");
        assert!(
            label == "cat_A" || label == "cat_B",
            "tie should resolve to a valid domain label, got '{label}'"
        );
    }

    #[test]
    fn predict_distributions_matches_all_leaves_given_x() {
        let tree = make_tree_with_target_split_then_feature_split();

        let x = DataFrame::new(vec![Column::new(
            PlSmallStr::from_static("x"),
            vec![Some(2.0_f64), Some(8.0_f64)],
        )])
        .unwrap();
        let dataset = PolarsDatasetView::new(&x);

        let dists = tree.predict_distributions(&dataset);
        assert_eq!(dists.len(), 2);

        // For each x, both target partitions should be present.
        assert_eq!(dists[0].n_cells(), 2, "x=2 should match two target leaves");
        assert_eq!(dists[1].n_cells(), 2, "x=8 should match two target leaves");

        let m0 = dists[0].mean_vector();
        let m1 = dists[1].mean_vector();
        let y0 = m0["target_y"][0];
        let y1 = m1["target_y"][0];
        assert!(
            (y0 - y1).abs() > 1e-12,
            "means should differ between x=2 and x=8 ({y0} vs {y1})"
        );
    }

    // -----------------------------------------------------------------------
    // walk_leaves / match_leaves_given_features tests
    // -----------------------------------------------------------------------

    /// Build a tree with two feature columns (x1, x2) and one target column.
    ///
    /// Structure (depth 2):
    ///
    /// ```text
    ///         root (split x1 at 5)
    ///         /                \
    ///   node1 (split x2 at 3)   node2 (split target_y at 5)
    ///    /       \               /        \
    ///  leaf0    leaf1         leaf2      leaf3
    /// ```
    ///
    /// - `leaf0`: x1<5, x2<3
    /// - `leaf1`: x1<5, x2≥3
    /// - `leaf2`: x1≥5, y<5
    /// - `leaf3`: x1≥5, y≥5
    fn make_two_feature_tree() -> Tree {
        let ci = |lo, hi, inc_lo, inc_hi| {
            Box::new(ContinuousInterval::new(
                lo,
                hi,
                inc_lo,
                inc_hi,
                Some((0.0, 10.0)),
                false,
            )) as Box<dyn DynRule>
        };

        let root = FittedNode {
            cell: Cell::new()
                .with_rule("x1", ci(0.0, 10.0, true, true))
                .with_rule("x2", ci(0.0, 10.0, true, true))
                .with_rule("target__y", ci(0.0, 10.0, true, true)),
            w_xy: 100.0,
            w_x: 100.0,
            w_y: 100.0,
            depth: 0,
            parent: None,
            left_child: Some(1),
            right_child: Some(2),
            is_leaf: false,
            split_col: Some("x1".to_string()),
            split_kind: Some(SplitKind::XSplit),
        };
        // node1: x1 < 5, split on x2
        let node1 = FittedNode {
            cell: Cell::new()
                .with_rule("x1", ci(0.0, 5.0, true, false))
                .with_rule("x2", ci(0.0, 10.0, true, true))
                .with_rule("target__y", ci(0.0, 10.0, true, true)),
            w_xy: 50.0,
            w_x: 50.0,
            w_y: 100.0,
            depth: 1,
            parent: Some(0),
            left_child: Some(3),
            right_child: Some(4),
            is_leaf: false,
            split_col: Some("x2".to_string()),
            split_kind: Some(SplitKind::XSplit),
        };
        // node2: x1 >= 5, split on target__y
        let node2 = FittedNode {
            cell: Cell::new()
                .with_rule("x1", ci(5.0, 10.0, true, true))
                .with_rule("x2", ci(0.0, 10.0, true, true))
                .with_rule("target__y", ci(0.0, 10.0, true, true)),
            w_xy: 50.0,
            w_x: 50.0,
            w_y: 100.0,
            depth: 1,
            parent: Some(0),
            left_child: Some(5),
            right_child: Some(6),
            is_leaf: false,
            split_col: Some("target__y".to_string()),
            split_kind: Some(SplitKind::YSplit),
        };
        // leaf0: x1<5, x2<3
        let leaf0 = FittedNode {
            cell: Cell::new()
                .with_rule("x1", ci(0.0, 5.0, true, false))
                .with_rule("x2", ci(0.0, 3.0, true, false))
                .with_rule("target__y", ci(0.0, 10.0, true, true)),
            w_xy: 20.0,
            w_x: 20.0,
            w_y: 100.0,
            depth: 2,
            parent: Some(1),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };
        // leaf1: x1<5, x2>=3
        let leaf1 = FittedNode {
            cell: Cell::new()
                .with_rule("x1", ci(0.0, 5.0, true, false))
                .with_rule("x2", ci(3.0, 10.0, true, true))
                .with_rule("target__y", ci(0.0, 10.0, true, true)),
            w_xy: 30.0,
            w_x: 30.0,
            w_y: 100.0,
            depth: 2,
            parent: Some(1),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };
        // leaf2: x1>=5, y<5
        let leaf2 = FittedNode {
            cell: Cell::new()
                .with_rule("x1", ci(5.0, 10.0, true, true))
                .with_rule("x2", ci(0.0, 10.0, true, true))
                .with_rule("target__y", ci(0.0, 5.0, true, false)),
            w_xy: 25.0,
            w_x: 50.0,
            w_y: 50.0,
            depth: 2,
            parent: Some(2),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };
        // leaf3: x1>=5, y>=5
        let leaf3 = FittedNode {
            cell: Cell::new()
                .with_rule("x1", ci(5.0, 10.0, true, true))
                .with_rule("x2", ci(0.0, 10.0, true, true))
                .with_rule("target__y", ci(5.0, 10.0, true, true)),
            w_xy: 25.0,
            w_x: 50.0,
            w_y: 50.0,
            depth: 2,
            parent: Some(2),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };

        Tree {
            nodes: vec![root, node1, node2, leaf0, leaf1, leaf2, leaf3],
            leaves: vec![3, 4, 5, 6],
            split_history: vec![],
        }
    }

    #[test]
    fn walk_with_all_features_routes_to_single_leaf() {
        let tree = make_two_feature_tree();

        // x1=2, x2=1 → left on x1 (<5), left on x2 (<3) → leaf0 (index 3)
        let df = DataFrame::new(vec![
            Column::new(PlSmallStr::from_static("x1"), vec![2.0_f64]),
            Column::new(PlSmallStr::from_static("x2"), vec![1.0_f64]),
        ])
        .unwrap();
        let dataset = PolarsDatasetView::new(&df);
        let cols = dataset.feature_columns();
        let matched = tree.match_leaves_given_features(0, &cols);
        // Target split at node2 is unreachable (x1 < 5 goes left), so 1 leaf.
        assert_eq!(matched, vec![3]);
    }

    #[test]
    fn walk_with_all_features_explores_y_split_both_sides() {
        let tree = make_two_feature_tree();

        // x1=7, x2=5 → right on x1 (≥5), then target split → both leaf2, leaf3
        let df = DataFrame::new(vec![
            Column::new(PlSmallStr::from_static("x1"), vec![7.0_f64]),
            Column::new(PlSmallStr::from_static("x2"), vec![5.0_f64]),
        ])
        .unwrap();
        let dataset = PolarsDatasetView::new(&df);
        let cols = dataset.feature_columns();
        let mut matched = tree.match_leaves_given_features(0, &cols);
        matched.sort();
        assert_eq!(
            matched,
            vec![5, 6],
            "both target-split children must be reached"
        );
    }

    #[test]
    fn walk_with_partial_features_explores_missing_feature_splits() {
        let tree = make_two_feature_tree();

        // Only x1=2, no x2. x1<5 → go left to node1 (split on x2).
        // x2 absent → both children of node1 explored → leaf0 + leaf1.
        let df = DataFrame::new(vec![Column::new(
            PlSmallStr::from_static("x1"),
            vec![2.0_f64],
        )])
        .unwrap();
        let dataset = PolarsDatasetView::new(&df);
        let cols = dataset.feature_columns();
        let mut matched = tree.match_leaves_given_features(0, &cols);
        matched.sort();
        assert_eq!(matched, vec![3, 4], "missing x2 → both x2-split children");
    }

    #[test]
    fn walk_with_no_features_returns_all_leaves() {
        let tree = make_two_feature_tree();

        // No columns at all → every split is on an absent column → all leaves.
        let cols: Vec<&dyn ColumnView> = vec![];
        let mut matched = tree.match_leaves_given_features(0, &cols);
        matched.sort();
        assert_eq!(matched, vec![3, 4, 5, 6], "no columns → all leaves");
    }

    #[test]
    fn walk_with_only_x2_explores_x1_splits_both_sides() {
        let tree = make_two_feature_tree();

        // Only x2=1 provided, no x1 → root split on x1 absent → both children.
        // node1 split on x2: present, x2=1 < 3 → left → leaf0 (3)
        // node2 split on target__y: absent → both → leaf2 (5) + leaf3 (6)
        let df = DataFrame::new(vec![Column::new(
            PlSmallStr::from_static("x2"),
            vec![1.0_f64],
        )])
        .unwrap();
        let dataset = PolarsDatasetView::new(&df);
        let cols = dataset.feature_columns();
        let mut matched = tree.match_leaves_given_features(0, &cols);
        matched.sort();
        assert_eq!(matched, vec![3, 5, 6]);
    }
}
