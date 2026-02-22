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
    /// 1. Routes each row to its leaf via [`predict_leaves`](Tree::predict_leaves).
    /// 2. Builds a [`ConditionedCell`] from each unique leaf (deduplication).
    /// 3. Wraps each in a [`PiecewiseConstantDistribution`].
    ///
    /// Returns a `Vec` of length `dataset.n_rows()`.
    pub fn predict_distributions(
        &self,
        dataset: &dyn DatasetView,
    ) -> Vec<PiecewiseConstantDistribution> {
        let leaf_indices = self.predict_leaves(dataset);

        // Build ConditionedCell once per unique leaf
        let mut cache: HashMap<usize, ConditionedCell> = HashMap::new();

        leaf_indices
            .into_iter()
            .map(|leaf_idx| {
                let cell = cache
                    .entry(leaf_idx)
                    .or_insert_with(|| ConditionedCell::from_fitted_node(&self.nodes[leaf_idx]))
                    .clone();
                PiecewiseConstantDistribution::from_single(cell)
            })
            .collect()
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
    use crate::rules::ContinuousInterval;
    use crate::cell::Cell;
    use crate::rule::{DynRule, DynValue};

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
}
