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

use super::cell::Cell;
use super::dataset_view::{ColumnView, DatasetView};
use super::rule::DynValue;
use super::split_result::SplitKind;

// ---------------------------------------------------------------------------
// FittedNode
// ---------------------------------------------------------------------------

/// A node in the fitted tree.
///
/// Contains the partition constraint ([`Cell`]), aggregated weights, tree
/// structure (parent/child indices), and a leaf flag. No training indices
/// or sorted lists are retained.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
    pub fn predict_leaf_from_map(
        &self,
        row: &HashMap<String, Option<DynValue>>,
    ) -> usize {
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
    use crate::v2::cell::Cell;
    use crate::rules::ContinuousInterval;
    use crate::v2::rule::{DynRule, DynValue};

    fn make_simple_tree() -> Tree {
        // Root → Left (leaf), Right (leaf)
        let root_cell = Cell::new().with_rule(
            "x",
            Box::new(ContinuousInterval::new(
                0.0, 10.0, true, true, Some((0.0, 10.0)), true,
            )) as Box<dyn DynRule>,
        );

        let left_cell = Cell::new().with_rule(
            "x",
            Box::new(ContinuousInterval::new(
                0.0, 5.0, true, false, Some((0.0, 10.0)), true,
            )) as Box<dyn DynRule>,
        );

        let right_cell = Cell::new().with_rule(
            "x",
            Box::new(ContinuousInterval::new(
                5.0, 10.0, true, true, Some((0.0, 10.0)), false,
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