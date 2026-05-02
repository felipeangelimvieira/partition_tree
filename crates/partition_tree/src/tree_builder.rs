//! Best-first partition tree construction.
//!
//! The [`TreeBuilder`] grows a partition tree using a max-heap priority queue
//! keyed by split gain. At each iteration it:
//!
//! 1. Pops the node with the highest gain.
//! 2. Creates left / right children (cells + propagated indices).
//! 3. Searches for the best split on each child.
//! 4. Pushes viable children back into the heap.
//!
//! The loop terminates when:
//! - The heap is empty (no more valid splits), or
//! - The [`max_leaves`](TreeBuilderConfig::max_leaves) limit is reached.
//!
//! ## Configuration
//!
//! See [`TreeBuilderConfig`] for tunable parameters and
//! [`SplitRestrictions`] for per-split constraints.
use std::collections::BinaryHeap;
use std::sync::Arc;

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::cell::Cell;
use crate::dataset_view::DatasetView;
use crate::dtype_plugin::DTypeRegistry;
use crate::loss::LossFunc;
use crate::node::Node;
use crate::split_result::{CandidateSplit, SplitKind, SplitPoint, SplitRestrictions};
use crate::split_searcher::SplitSearcher;
use crate::tree::{FittedNode, SplitRecord, Tree};

/// Configuration for tree building.
///
/// # Defaults
///
/// | Field                         | Default        |
/// |-------------------------------|----------------|
/// | `max_leaves`                  | `usize::MAX`   |
/// | `boundaries_expansion_factor` | `0.1`          |
/// | `restrictions`                | see [`SplitRestrictions::default`] |
pub struct TreeBuilderConfig {
    /// Maximum number of leaf nodes.
    pub max_leaves: usize,
    /// Expansion factor for target column boundaries.
    pub boundaries_expansion_factor: f64,
    /// Split restrictions (min samples, max depth, etc.).
    pub restrictions: SplitRestrictions,
    /// Fraction of rows to bootstrap-sample for the root node.
    /// `None` means use all rows (default).
    pub max_samples: Option<f64>,
    /// Whether bootstrap sampling is done with replacement (`true`, default)
    /// or without replacement (`false`).
    pub replace: bool,
    /// Fraction of *feature* columns to consider at each split. `None` means
    /// use all features (default). Target columns are always included.
    pub max_features: Option<f64>,
    /// Maximum number of candidate split points evaluated per column during a
    /// split search. `None` means evaluate every valid candidate.
    pub max_candidate_split_points: Option<usize>,
    /// RNG seed for reproducible bootstrap / feature subsampling.
    /// `None` uses OS entropy.
    pub seed: Option<u64>,
}

impl Default for TreeBuilderConfig {
    fn default() -> Self {
        Self {
            max_leaves: usize::MAX,
            boundaries_expansion_factor: 0.1,
            restrictions: SplitRestrictions::default(),
            max_samples: None,
            replace: true,
            max_features: None,
            max_candidate_split_points: None,
            seed: None,
        }
    }
}

/// Best-first partition tree builder.
///
/// Owns the loss function and dtype registry (shared via `Arc`). Call
/// [`build`](TreeBuilder::build) with a [`DatasetView`] to produce a
/// fitted [`Tree`].
///
/// # Example
///
/// ```rust,ignore
/// let builder = TreeBuilder::new(config, loss, registry);
/// let tree = builder.build(&dataset);
/// ```
pub struct TreeBuilder {
    pub config: TreeBuilderConfig,
    pub loss: Box<dyn LossFunc>,
    pub registry: Arc<DTypeRegistry>,
}

impl TreeBuilder {
    pub fn new(
        config: TreeBuilderConfig,
        loss: Box<dyn LossFunc>,
        registry: Arc<DTypeRegistry>,
    ) -> Self {
        Self {
            config,
            loss,
            registry,
        }
    }

    /// Build a partition tree from the dataset.
    ///
    /// Returns a `Tree` containing the fitted nodes and leaf indices.
    pub fn build(&self, dataset: &dyn DatasetView) -> Tree {
        let split_searcher = SplitSearcher::new(Arc::clone(&self.registry));
        let dataset_size = dataset.n_rows() as f64;

        // Create RNG from seed (or OS entropy)
        let mut rng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        // 1. Create root cell from DTypeRegistry defaults
        let root_cell = self.create_root_cell(dataset);

        // 2. Create root node (with optional bootstrap sampling)
        let root_node = if let Some(max_samples) = self.config.max_samples {
            Node::root_bootstrap(
                dataset,
                root_cell,
                max_samples,
                self.config.replace,
                &mut rng,
            )
        } else {
            Node::root(dataset, root_cell)
        };

        // Resolve max_features fraction to an absolute column count
        let n_features = dataset.columns().iter().filter(|c| !c.is_target()).count();
        let max_features_count = self
            .config
            .max_features
            .map(|frac| (frac * n_features as f64).ceil().max(1.0) as usize);

        // Arena of nodes (index-based)
        let mut nodes: Vec<FittedNode> = Vec::new();
        let mut build_nodes: Vec<Option<Node>> = Vec::new(); // parallel arena for builder state
        let mut split_history: Vec<SplitRecord> = Vec::new();

        // Push root as the first node
        let root_fitted = FittedNode {
            cell: root_node.cell.clone(),
            w_xy: root_node.w_xy,
            w_x: root_node.w_x,
            w_y: root_node.w_y,
            depth: root_node.depth,
            parent: None,
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
        };
        nodes.push(root_fitted);
        build_nodes.push(Some(root_node));

        // 3. Priority queue
        let mut heap: BinaryHeap<CandidateSplit> = BinaryHeap::new();

        // Try to find a split for root
        if self
            .config
            .restrictions
            .can_split(nodes[0].w_xy, nodes[0].depth)
        {
            if let Some(split) = split_searcher.find_best_split(
                build_nodes[0].as_ref().unwrap(),
                dataset,
                self.loss.as_ref(),
                &self.config.restrictions,
                max_features_count,
                self.config.max_candidate_split_points,
                &mut rng,
                dataset_size,
            ) {
                heap.push(CandidateSplit {
                    node_index: 0,
                    split,
                });
            }
        }

        let mut n_leaves = 1_usize;

        // 4. Best-first loop
        while let Some(candidate) = heap.pop() {
            // Check if we've reached the max leaves limit.
            // Each split turns one leaf into a non-leaf and adds two new leaves
            // (net gain of 1 leaf).
            if n_leaves >= self.config.max_leaves {
                break;
            }

            let parent_idx = candidate.node_index;

            // Take the builder node out of the arena (it won't be needed after)
            let parent_build_node = match build_nodes[parent_idx].take() {
                Some(n) => n,
                None => continue, // already split
            };

            let split = &candidate.split;

            // Create child cells via the dtype-erased SplitOp
            let (left_cell, right_cell) = parent_build_node.cell.apply_split(
                &split.col_name,
                split.op.as_ref(),
                split.none_to_left,
            );

            // Propagate sorted indices
            let (left_build, right_build) =
                parent_build_node.propagate_children(split, dataset, left_cell, right_cell);

            // Allocate child indices
            let left_idx = nodes.len();
            let right_idx = left_idx + 1;

            // Create fitted nodes
            let left_fitted = FittedNode {
                cell: left_build.cell.clone(),
                w_xy: left_build.w_xy,
                w_x: left_build.w_x,
                w_y: left_build.w_y,
                depth: left_build.depth,
                parent: Some(parent_idx),
                left_child: None,
                right_child: None,
                is_leaf: true,
                split_col: None,
                split_kind: None,
            };
            let right_fitted = FittedNode {
                cell: right_build.cell.clone(),
                w_xy: right_build.w_xy,
                w_x: right_build.w_x,
                w_y: right_build.w_y,
                depth: right_build.depth,
                parent: Some(parent_idx),
                left_child: None,
                right_child: None,
                is_leaf: true,
                split_col: None,
                split_kind: None,
            };

            nodes.push(left_fitted);
            nodes.push(right_fitted);

            // Update parent
            nodes[parent_idx].left_child = Some(left_idx);
            nodes[parent_idx].right_child = Some(right_idx);
            nodes[parent_idx].is_leaf = false;
            nodes[parent_idx].split_col = Some(split.col_name.clone());
            nodes[parent_idx].split_kind = Some(split.split_kind);

            // Record the split
            split_history.push(SplitRecord {
                parent_index: parent_idx,
                col_name: split.col_name.clone(),
                split_kind: split.split_kind,
                gain: split.gain,
                left_child_index: left_idx,
                right_child_index: right_idx,
            });

            // n_leaves: we removed 1 leaf (parent) and added 2 (children)
            n_leaves += 1;

            // Store build nodes
            build_nodes.push(Some(left_build));
            build_nodes.push(Some(right_build));

            // Search for best splits on children and push to heap
            for child_idx in [left_idx, right_idx] {
                let child_node = build_nodes[child_idx].as_ref().unwrap();
                if self
                    .config
                    .restrictions
                    .can_split(nodes[child_idx].w_xy, nodes[child_idx].depth)
                {
                    if let Some(child_split) = split_searcher.find_best_split(
                        child_node,
                        dataset,
                        self.loss.as_ref(),
                        &self.config.restrictions,
                        max_features_count,
                        self.config.max_candidate_split_points,
                        &mut rng,
                        dataset_size,
                    ) {
                        heap.push(CandidateSplit {
                            node_index: child_idx,
                            split: child_split,
                        });
                    }
                }
            }
        }

        // 5. Collect leaf indices
        let leaves: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_leaf)
            .map(|(i, _)| i)
            .collect();

        Tree {
            nodes,
            leaves,
            split_history,
        }
    }

    /// Build the root cell with default rules for all columns.
    fn create_root_cell(&self, dataset: &dyn DatasetView) -> Cell {
        let mut cell = Cell::new();
        for col in dataset.columns() {
            let dtype = col.logical_dtype();
            let plugin = self.registry.get_or_panic(dtype);
            let rule = plugin.default_rule(col, self.config.boundaries_expansion_factor);
            cell.set_rule(col.name(), rule);
        }
        cell
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset_view::PolarsDatasetView;
    use crate::loss::ConditionalLogLoss;
    use polars::prelude::*;

    fn make_dataset() -> PolarsDatasetView {
        let df = DataFrame::new(vec![
            Column::new("x1".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            Column::new(
                "target__y1".into(),
                &[10.0_f64, 10.0, 10.0, 10.0, 50.0, 50.0, 50.0, 50.0],
            ),
        ])
        .unwrap();
        PolarsDatasetView::new(&df)
    }

    #[test]
    fn builds_a_tree_with_leaves() {
        let dataset = make_dataset();
        let config = TreeBuilderConfig {
            max_leaves: 10,
            boundaries_expansion_factor: 0.1,
            restrictions: SplitRestrictions {
                min_samples_xy: 1.0,
                min_samples_x: 1.0,
                min_samples_y: 1.0,
                min_gain: 0.0,
                min_volume_fraction: 0.0,
                max_depth: 5,
                min_samples_split: 2.0,
            },
            ..Default::default()
        };
        let loss = Box::new(ConditionalLogLoss);
        let registry = Arc::new(DTypeRegistry::default());
        let builder = TreeBuilder::new(config, loss, registry);

        let tree = builder.build(&dataset);

        assert!(tree.n_leaves() >= 2, "tree should have at least 2 leaves");
        assert!(
            tree.nodes.len() >= 3,
            "tree should have at least 3 nodes (root + 2 children)"
        );
        assert!(!tree.split_history.is_empty(), "should have split history");
    }

    #[test]
    fn respects_max_leaves() {
        let dataset = make_dataset();
        let config = TreeBuilderConfig {
            max_leaves: 2,
            boundaries_expansion_factor: 0.1,
            restrictions: SplitRestrictions::default(),
            ..Default::default()
        };
        let loss = Box::new(ConditionalLogLoss);
        let registry = Arc::new(DTypeRegistry::default());
        let builder = TreeBuilder::new(config, loss, registry);

        let tree = builder.build(&dataset);

        assert!(tree.n_leaves() <= 2, "should respect max_leaves=2");
    }

    #[test]
    fn respects_max_depth() {
        let dataset = make_dataset();
        let config = TreeBuilderConfig {
            max_leaves: 100,
            boundaries_expansion_factor: 0.1,
            restrictions: SplitRestrictions {
                max_depth: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let loss = Box::new(ConditionalLogLoss);
        let registry = Arc::new(DTypeRegistry::default());
        let builder = TreeBuilder::new(config, loss, registry);

        let tree = builder.build(&dataset);

        for node in &tree.nodes {
            assert!(node.depth <= 1, "no node should exceed max_depth=1");
        }
    }

    #[test]
    fn single_value_column_produces_no_split() {
        let df = DataFrame::new(vec![
            Column::new("x1".into(), &[1.0_f64, 1.0, 1.0, 1.0]),
            Column::new("target__y1".into(), &[10.0_f64, 10.0, 10.0, 10.0]),
        ])
        .unwrap();
        let dataset = PolarsDatasetView::new(&df);

        let config = TreeBuilderConfig::default();
        let loss = Box::new(ConditionalLogLoss);
        let registry = Arc::new(DTypeRegistry::default());
        let builder = TreeBuilder::new(config, loss, registry);

        let tree = builder.build(&dataset);

        assert_eq!(tree.n_leaves(), 1, "no valid splits possible");
        assert_eq!(tree.nodes.len(), 1, "only root node");
    }

    /// Set `min_volume_fraction` above 0.5 so that every binary target split
    /// is rejected: since left + right = parent volume = domain volume (all splits
    /// start from the root), both children cannot simultaneously hold more than
    /// 50 % of the domain's target volume. No `YSplit` should appear in the split
    /// history.
    #[test]
    fn respects_min_target_volume() {
        let dataset = make_dataset();
        // Any fraction > 0.5 makes every binary YSplit invalid because the two
        // children must partition the root (domain) volume, so at least one of
        // them is always < 50 % of it.
        let min_frac = 0.6_f64;

        let config = TreeBuilderConfig {
            max_leaves: 100,
            boundaries_expansion_factor: 0.1,
            restrictions: SplitRestrictions {
                min_volume_fraction: min_frac,
                ..Default::default()
            },
            ..Default::default()
        };
        let loss = Box::new(ConditionalLogLoss);
        let registry = Arc::new(DTypeRegistry::default());
        let builder = TreeBuilder::new(config, loss, registry);

        let tree = builder.build(&dataset);

        // No target-space split should appear in the history.
        let y_splits: Vec<_> = tree
            .split_history
            .iter()
            .filter(|r| r.split_kind == SplitKind::YSplit)
            .collect();

        assert!(
            y_splits.is_empty(),
            "expected no YSplits with min_volume_fraction={min_frac}, found {} YSplit(s)",
            y_splits.len()
        );

        // Safety net: for any YSplit that somehow appeared, each child must hold
        // at least `min_frac` of the total target domain volume.
        for rec in &tree.split_history {
            if rec.split_kind == SplitKind::YSplit {
                let domain_vol = tree.nodes[rec.parent_index].cell.target_domain_volume();
                let lv = tree.nodes[rec.left_child_index].cell.target_volume();
                let rv = tree.nodes[rec.right_child_index].cell.target_volume();
                assert!(
                    lv >= min_frac * domain_vol,
                    "left child fraction {:.4} < {min_frac}",
                    lv / domain_vol
                );
                assert!(
                    rv >= min_frac * domain_vol,
                    "right child fraction {:.4} < {min_frac}",
                    rv / domain_vol
                );
            }
        }
    }

    #[test]
    fn build_with_max_samples_produces_valid_tree() {
        let dataset = make_dataset();
        let config = TreeBuilderConfig {
            max_leaves: 10,
            boundaries_expansion_factor: 0.1,
            restrictions: SplitRestrictions::default(),
            max_samples: Some(0.5),
            replace: true,
            max_features: None,
            max_candidate_split_points: None,
            seed: Some(42),
        };
        let loss = Box::new(ConditionalLogLoss);
        let registry = Arc::new(DTypeRegistry::default());
        let builder = TreeBuilder::new(config, loss, registry);

        let tree = builder.build(&dataset);

        assert!(
            tree.n_leaves() >= 1,
            "tree with max_samples should have at least 1 leaf"
        );
        // The tree must have valid structure
        for node in &tree.nodes {
            if node.is_leaf {
                assert!(node.left_child.is_none());
                assert!(node.right_child.is_none());
            }
        }
    }

    #[test]
    fn build_with_max_features_produces_valid_tree() {
        let dataset = make_dataset();
        let config = TreeBuilderConfig {
            max_leaves: 10,
            boundaries_expansion_factor: 0.1,
            restrictions: SplitRestrictions::default(),
            max_samples: None,
            replace: true,
            max_features: Some(0.5),
            max_candidate_split_points: None,
            seed: Some(42),
        };
        let loss = Box::new(ConditionalLogLoss);
        let registry = Arc::new(DTypeRegistry::default());
        let builder = TreeBuilder::new(config, loss, registry);

        let tree = builder.build(&dataset);

        assert!(
            tree.n_leaves() >= 1,
            "tree with max_features should have at least 1 leaf"
        );
        assert!(!tree.split_history.is_empty(), "should still find splits");
    }

    #[test]
    fn same_seed_produces_same_tree() {
        let dataset = make_dataset();

        let build_tree = |seed: u64| {
            let config = TreeBuilderConfig {
                max_leaves: 10,
                boundaries_expansion_factor: 0.1,
                restrictions: SplitRestrictions::default(),
                max_samples: Some(0.7),
                replace: true,
                max_features: Some(0.5),
                max_candidate_split_points: None,
                seed: Some(seed),
            };
            let loss = Box::new(ConditionalLogLoss);
            let registry = Arc::new(DTypeRegistry::default());
            TreeBuilder::new(config, loss, registry).build(&dataset)
        };

        let tree_a = build_tree(42);
        let tree_b = build_tree(42);

        assert_eq!(
            tree_a.split_history.len(),
            tree_b.split_history.len(),
            "same seed should produce same number of splits"
        );
        for (a, b) in tree_a.split_history.iter().zip(tree_b.split_history.iter()) {
            assert_eq!(a.col_name, b.col_name, "split columns should match");
            assert!(
                (a.gain - b.gain).abs() < 1e-12,
                "split gains should match: {} vs {}",
                a.gain,
                b.gain
            );
        }
    }

    #[test]
    fn different_seeds_can_produce_different_trees() {
        let dataset = make_dataset();

        let build_tree = |seed: u64| {
            let config = TreeBuilderConfig {
                max_leaves: 10,
                boundaries_expansion_factor: 0.1,
                restrictions: SplitRestrictions::default(),
                max_samples: Some(0.5),
                replace: true,
                max_features: Some(0.5),
                max_candidate_split_points: None,
                seed: Some(seed),
            };
            let loss = Box::new(ConditionalLogLoss);
            let registry = Arc::new(DTypeRegistry::default());
            TreeBuilder::new(config, loss, registry).build(&dataset)
        };

        // Try multiple seed pairs; at least one pair should differ
        let mut found_different = false;
        for s in 0..20 {
            let tree_a = build_tree(s);
            let tree_b = build_tree(s + 1000);

            let same = tree_a.split_history.len() == tree_b.split_history.len()
                && tree_a
                    .split_history
                    .iter()
                    .zip(tree_b.split_history.iter())
                    .all(|(a, b)| a.col_name == b.col_name && (a.gain - b.gain).abs() < 1e-12);

            if !same {
                found_different = true;
                break;
            }
        }

        assert!(
            found_different,
            "different seeds with max_samples + max_features should produce \
             different trees for at least one seed pair"
        );
    }

    #[test]
    fn max_candidate_split_points_zero_disables_splits() {
        let dataset = make_dataset();
        let config = TreeBuilderConfig {
            max_candidate_split_points: Some(0),
            ..Default::default()
        };
        let loss = Box::new(ConditionalLogLoss);
        let registry = Arc::new(DTypeRegistry::default());
        let builder = TreeBuilder::new(config, loss, registry);

        let tree = builder.build(&dataset);

        assert_eq!(tree.n_leaves(), 1, "no candidates means no split");
        assert!(tree.split_history.is_empty(), "split history should stay empty");
    }
}
