use crate::conf::TARGET_PREFIX;

use std::any::Any;
use std::sync::Arc;
// use crate::dataframe::PTreeDataFrameExt;
use crate::dtype_adapter::*;
use crate::node::Node;
use crate::predict::probability::*;
use crate::rules::BelongsToString;
use crate::split::{SplitRestrictions, SplitResult, find_best_split_column};
use core::panic;
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// Candidate for priority queue (best-first by highest gain)
#[derive(Clone)]
struct SplitCandidate {
    gain: f64,
    column_name: String,
    split_result: SplitResult,
    node: Node,
    node_index: usize,
}

impl PartialEq for SplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}
impl Eq for SplitCandidate {}
impl PartialOrd for SplitCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // max-heap on gain
        self.gain.partial_cmp(&other.gain)
    }
}
impl Ord for SplitCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl SplitCandidate {
    pub fn is_valid(&self) -> bool {
        match &self.split_result {
            SplitResult::InvalidSplit(_) => false,
            _ => {
                if self.gain.is_finite() {
                    true
                } else {
                    panic!("SplitCandidate with non-finite gain should be InvalidSplit")
                }
            }
        }
    }
}

// For future visualization (node relationships)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub parent: Option<usize>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub is_leaf: bool,
}

// Split record to inspect build history
#[derive(Debug, Clone)]
pub struct SplitRecord {
    pub parent_index: usize,
    pub column_name: String,
    pub split_result: SplitResult,
    pub left_child_index: usize,
    pub right_child_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LeafReason {
    MinSamplesSplit(usize),
    MinSamplesLeaf(usize),
    MaxDepth(usize),
    NoValidSplit(String),
    IterationLimit,
}

/// Information about a partition dimension in a leaf
#[derive(Debug, Clone)]
pub enum PartitionInfo {
    /// Continuous interval: (low, high, lower_closed, upper_closed)
    Continuous {
        low: f64,
        high: f64,
        lower_closed: bool,
        upper_closed: bool,
    },
    /// Categorical partition: list of category names
    Categorical { categories: Vec<String> },
}

/// Complete information about a leaf node
#[derive(Debug, Clone)]
pub struct LeafInfo {
    /// Index of this leaf in the tree's node list
    pub leaf_index: usize,
    /// Depth of the leaf in the tree
    pub depth: usize,
    /// Number of training samples in this leaf
    pub n_samples: usize,
    /// Partition info for each dimension (column name -> PartitionInfo)
    pub partitions: HashMap<String, PartitionInfo>,
    /// Indices of samples matching both X and Y constraints
    pub indices_xy: Vec<u32>,
    /// Indices of samples matching X constraints only
    pub indices_x: Vec<u32>,
    /// Indices of samples matching Y constraints only
    pub indices_y: Vec<u32>,
    /// Feature contributions: cumulative gain from root to this leaf per column
    pub feature_contributions: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeBuilderStatus {
    FAILED(String),
    SUCCESS,
}

pub struct Tree {
    nodes: Vec<Node>,
    leaves: Vec<usize>,
    max_iter: usize,
    restrictions: SplitRestrictions,
    boundaries_expansion_factor: f64, // expand continuous target boundaries by this factor
    max_samples: Option<f64>,
    max_features: Option<f64>,
    exploration_split_budget: usize,
    feature_split_fraction: Option<f64>, // None means no restriction; Some(x) forces first x% of splits on non-target columns
    seed: Option<usize>,
    split_history: Vec<SplitRecord>,
    leaf_reasons: HashMap<usize, LeafReason>,
    schema: Option<DataFrame>,
    build_status: TreeBuilderStatus,
}

impl Tree {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_iter: usize,
        min_samples_split: usize,
        min_samples_leaf_y: usize,
        min_samples_leaf_x: usize,
        min_samples_leaf: usize,
        max_depth: usize,
        min_target_volume: f64,
        min_split_gain: f64,
        min_density_value: f64,
        max_density_value: f64,
        max_measure_value: f64,
        boundaries_expansion_factor: f64,
        max_samples: Option<f64>,
        max_features: Option<f64>,
        exploration_split_budget: usize,
        feature_split_fraction: Option<f64>,
        seed: Option<usize>,
    ) -> Self {
        Tree {
            nodes: Vec::with_capacity(max_iter),
            leaves: Vec::new(),
            max_iter,
            restrictions: SplitRestrictions {
                min_samples_split,
                min_samples_leaf_y,
                min_samples_leaf_x,
                min_samples_leaf,
                max_depth,
                min_target_volume,
                min_split_gain,
                total_target_volume: 1.0, // total_target_volume (will be set at node evaluation)
                min_density_value,
                max_density_value,
                max_measure_value,
                dataset_size: 0.0,
            },
            boundaries_expansion_factor,
            max_samples,
            max_features,
            seed: seed,
            exploration_split_budget: exploration_split_budget,
            feature_split_fraction,
            split_history: Vec::new(),
            leaf_reasons: HashMap::new(),
            schema: None,
            build_status: TreeBuilderStatus::SUCCESS,
        }
    }

    pub fn fit(&mut self, df: &DataFrame, sample_weights: Option<Float64Chunked>) {
        // Set dataset_size from the DataFrame
        self.restrictions.dataset_size = df.height() as f64;

        let mut builder = BestFirstTreeBuilder::new(
            self.max_iter,
            self.restrictions.clone(),
            self.boundaries_expansion_factor,
            self.max_samples,
            self.max_features,
            self.exploration_split_budget,
            self.feature_split_fraction,
            StdRng::seed_from_u64(self.seed.unwrap_or(42) as u64),
        );

        let sample_weights = match sample_weights {
            Some(w) => w,
            None => {
                Float64Chunked::full(PlSmallStr::from_static("sample_weights"), 1.0, df.height())
            }
        };
        builder.build_tree(df, &sample_weights);

        self.build_status = builder.status.clone();
        self.nodes = builder.nodes;
        self.leaves = builder.leaves;
        self.split_history = builder.split_history;

        //print_split_summary(&self.split_history, &self.leaf_reasons);
        self.leaf_reasons = builder.leaf_reasons;
        self.schema = Some(df.clone()); // keep schema and categories
    }

    pub fn get_build_status(&self) -> &TreeBuilderStatus {
        &self.build_status
    }

    pub fn get_leaves(&self) -> &Vec<usize> {
        &self.leaves
    }
    pub fn get_node(&self, index: usize) -> Option<&Node> {
        self.nodes.get(index)
    }
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn get_split_history(&self) -> &Vec<SplitRecord> {
        &self.split_history
    }
    pub fn get_leaf_reasons(&self) -> &HashMap<usize, LeafReason> {
        &self.leaf_reasons
    }

    /// Compute feature importances based on cumulative gain from all splits.
    /// Each split is counted exactly once (no double-counting across leaves).
    /// Returns a HashMap where keys are column names and values are the sum of gains.
    /// If `normalize` is true, the importances are normalized to sum to 1.0.
    pub fn get_feature_importances(&self, normalize: bool) -> HashMap<String, f64> {
        let mut importances: HashMap<String, f64> = HashMap::new();

        // Iterate over split_history: each split is counted exactly once
        for split in &self.split_history {
            let gain = split.split_result.gain();
            if gain.is_finite() && gain > 0.0 {
                *importances.entry(split.column_name.clone()).or_insert(0.0) += gain;
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

    /// For each row, find which leaves contain it using X-only columns.
    pub fn match_leaves_given_x(&self, df: &DataFrame) -> Vec<Vec<usize>> {
        use crate::conf::TargetBehaviour;
        let mut matches_per_row = vec![Vec::<usize>::new(); df.height()];
        for (leaf_idx_pos, &leaf_node_idx) in self.leaves.iter().enumerate() {
            let leaf_node = &self.nodes[leaf_node_idx];
            // Use cell to match X-only
            let height = df.height();

            let idx_range: std::ops::Range<u32> = 0u32..(height as u32);
            let subset = UInt32Chunked::new(PlSmallStr::from_static("idx"), idx_range);
            let mask_idx = leaf_node
                .cell
                .match_dataframe(df, &subset, TargetBehaviour::Exclude)
                .expect("match_dataframe_x");

            for opt_idx in mask_idx.into_iter() {
                if let Some(row_idx) = opt_idx {
                    matches_per_row[row_idx as usize].push(leaf_idx_pos);
                }
            }
        }
        matches_per_row
    }

    /// Apply the tree to the input data, returning the leaf index for each sample.
    /// Returns a Vec where each element is a Vec of leaf indices (positions in the leaves vector)
    /// that the sample belongs to. In most cases, each sample belongs to exactly one leaf.
    /// Note: These are indices into the leaves array, not node indices.
    pub fn apply(&self, df: &DataFrame) -> Vec<Vec<usize>> {
        use crate::conf::TargetBehaviour;
        let mut matches_per_row = vec![Vec::<usize>::new(); df.height()];
        for (leaf_pos, &leaf_node_idx) in self.leaves.iter().enumerate() {
            let leaf_node = &self.nodes[leaf_node_idx];
            let height = df.height();

            let idx_range: std::ops::Range<u32> = 0u32..(height as u32);
            let subset = UInt32Chunked::new(PlSmallStr::from_static("idx"), idx_range);
            let mask_idx = leaf_node
                .cell
                .match_dataframe(df, &subset, TargetBehaviour::Exclude)
                .expect("match_dataframe_x");

            for opt_idx in mask_idx.into_iter() {
                if let Some(row_idx) = opt_idx {
                    matches_per_row[row_idx as usize].push(leaf_pos);
                }
            }
        }
        matches_per_row
    }

    /// Predict mean for target columns via volume-weighted average across matching leaves
    pub fn predict_mean(&self, df: &DataFrame) -> DataFrame {
        let probas = self.predict_proba(df);
        let mean_vector_or_rows: Vec<HashMap<String, Vec<f64>>> =
            probas.iter().map(|p| p.mean()).collect();

        self.decode_mean_from_mean_vector(&mean_vector_or_rows)
    }

    pub fn decode_mean_from_mean_vector(
        &self,
        mean_vector: &Vec<HashMap<String, Vec<f64>>>,
    ) -> DataFrame {
        let root_cell = &self.nodes[0].cell;
        assert!(self.schema.is_some());

        let out = self
            .schema
            .as_ref()
            .unwrap()
            .get_columns()
            .into_iter()
            .filter(|col| col.name().starts_with(TARGET_PREFIX))
            .map(|col| {
                let col_name = col.name();
                let col_str = col_name.as_str().to_string();
                let means: Vec<Vec<f64>> = mean_vector
                    .iter()
                    .map(|m| {
                        m.get(col_name.as_str()).cloned().unwrap_or_else(|| {
                            let available: Vec<String> = m.keys().cloned().collect();
                            println!("{}", self.tree_info());
                            panic!(
                                "Prediction failed: missing mean for target column '{col}'\nAvailable target keys in this distribution: {available:?}\nLikely cause: the sample did not match any leaf (e.g., unseen or mismatched target category).",
                                col = col_str,
                            )
                        })
                    })
                    .collect();
                let dtype_adapter = DtypeAdapter::new_from_dtype(col.dtype());
                dtype_adapter.predict_mean_for_dtype(root_cell, col_name, means)
            })
            .collect::<Vec<Column>>();

        DataFrame::new(out).expect("DataFrame::new")
    }

    pub fn predict_proba(&self, df: &DataFrame) -> Vec<PiecewiseConstantDistribution> {
        let _matches_per_row = self.match_leaves_given_x(df);
        _matches_per_row
            .iter()
            .map(|leaf_idxs| {
                let matched_nodes: Vec<&Node> = leaf_idxs
                    .iter()
                    .map(|&leaf_pos| {
                        let node_idx = self.leaves[leaf_pos];
                        &self.nodes[node_idx]
                    })
                    .collect();

                let cells = matched_nodes
                    .iter()
                    .map(|n| n.cell.project_to_target_cells())
                    .collect::<Vec<_>>();
                let mass: Vec<f64> = matched_nodes
                    .iter()
                    .map(|n| n.conditional_measure())
                    .collect::<Vec<_>>();
                // Create PiecewiseConstantDistribution
                PiecewiseConstantDistribution::new(cells, mass)
            })
            .collect()
    }

    fn predict_scalar_with<F>(&self, df: &DataFrame, eval: F) -> Vec<f64>
    where
        F: Fn(&PiecewiseConstantDistribution, &DataFrame) -> f64 + Send + Sync,
    {
        let probas = self.predict_proba(df);
        let df = Arc::new(df.clone());

        probas
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let idx = UInt32Chunked::from_slice(PlSmallStr::from_str("idx"), &[i as u32]);
                let row = df.take(&idx).expect("Failed to take row");
                eval(&p, &row)
            })
            .collect()
    }

    pub fn predict_pdf(&self, df: &DataFrame) -> Vec<f64> {
        self.predict_scalar_with(df, |p, row| p.pdf(row)[0])
    }

    pub fn predict_mass(&self, df: &DataFrame) -> Vec<f64> {
        self.predict_scalar_with(df, |p, row| p.mass(row)[0])
    }

    pub fn predict_pdf_scaled(&self, df: &DataFrame) -> Vec<f64> {
        self.predict_scalar_with(df, |p, row| p.pdf_scaled(row)[0])
    }

    fn _categorical_target_columns(&self) -> Vec<String> {
        self.nodes[0]
            .cell
            .categorical_features()
            .into_iter()
            .filter(|name| name.starts_with(TARGET_PREFIX))
            .collect()
    }

    /// Build a mapping from node index to (parent_index, column_name, gain)
    /// This is used to trace the path from any node back to the root.
    fn build_parent_map(&self) -> HashMap<usize, (usize, String, f64)> {
        let mut parent_map = HashMap::new();
        for split in &self.split_history {
            let gain = split.split_result.gain();
            parent_map.insert(
                split.left_child_index,
                (split.parent_index, split.column_name.clone(), gain),
            );
            parent_map.insert(
                split.right_child_index,
                (split.parent_index, split.column_name.clone(), gain),
            );
        }
        parent_map
    }

    /// Compute cumulative feature contributions from root to a given node.
    /// Returns a HashMap where keys are column names and values are the sum of gains.
    fn compute_feature_contributions(
        &self,
        node_idx: usize,
        parent_map: &HashMap<usize, (usize, String, f64)>,
    ) -> HashMap<String, f64> {
        let mut contributions: HashMap<String, f64> = HashMap::new();
        let mut current_idx = node_idx;

        // Walk up the tree from leaf to root, accumulating gains per feature
        while let Some((parent_idx, col_name, gain)) = parent_map.get(&current_idx) {
            *contributions.entry(col_name.clone()).or_insert(0.0) += gain;
            current_idx = *parent_idx;
        }

        contributions
    }

    /// Get detailed information about all leaves in the tree
    pub fn get_leaves_info(&self) -> Vec<LeafInfo> {
        use crate::rules::{BelongsToGeneric, ContinuousInterval};

        // Build parent map once for all leaves
        let parent_map = self.build_parent_map();

        self.leaves
            .iter()
            .map(|&leaf_idx| {
                let node = &self.nodes[leaf_idx];
                let mut partitions = HashMap::new();

                for (name, part) in &node.cell.partitions {
                    let rule_any = part.rule_any();

                    if let Some(interval) = rule_any.downcast_ref::<ContinuousInterval>() {
                        partitions.insert(
                            name.clone(),
                            PartitionInfo::Continuous {
                                low: interval.low,
                                high: interval.high,
                                lower_closed: interval.lower_closed,
                                upper_closed: interval.upper_closed,
                            },
                        );
                    } else if let Some(belongs_to) =
                        rule_any.downcast_ref::<BelongsToGeneric<usize>>()
                    {
                        // usize-coded categories: find position in domain, then get name
                        let categories: Vec<String> = belongs_to
                            .values
                            .iter()
                            .filter_map(|&code| {
                                // Find position of code in domain
                                belongs_to
                                    .domain
                                    .iter()
                                    .position(|&d| d == code)
                                    .and_then(|pos| belongs_to.domain_names.get(pos).cloned())
                            })
                            .collect();
                        partitions.insert(name.clone(), PartitionInfo::Categorical { categories });
                    } else if let Some(belongs_to) =
                        rule_any.downcast_ref::<BelongsToGeneric<u32>>()
                    {
                        // u32-coded categories: find position in domain, then get name
                        let categories: Vec<String> = belongs_to
                            .values
                            .iter()
                            .filter_map(|&code| {
                                // Find position of code in domain
                                belongs_to
                                    .domain
                                    .iter()
                                    .position(|&d| d == code)
                                    .and_then(|pos| belongs_to.domain_names.get(pos).cloned())
                            })
                            .collect();
                        partitions.insert(name.clone(), PartitionInfo::Categorical { categories });
                    } else if let Some(belongs_to) =
                        rule_any.downcast_ref::<BelongsToGeneric<String>>()
                    {
                        // String-coded categories: values are the category names directly
                        let categories: Vec<String> = belongs_to.values.iter().cloned().collect();
                        partitions.insert(name.clone(), PartitionInfo::Categorical { categories });
                    } else if let Some(belongs_to) =
                        rule_any.downcast_ref::<BelongsToGeneric<bool>>()
                    {
                        // Boolean categories
                        let categories: Vec<String> =
                            belongs_to.values.iter().map(|&v| v.to_string()).collect();
                        partitions.insert(name.clone(), PartitionInfo::Categorical { categories });
                    }
                    // Add more BelongsToGeneric variants as needed
                }

                // Compute feature contributions for this leaf
                let feature_contributions =
                    self.compute_feature_contributions(leaf_idx, &parent_map);

                LeafInfo {
                    leaf_index: leaf_idx,
                    depth: node.depth,
                    n_samples: node.indices_xy.len(),
                    partitions,
                    indices_xy: node
                        .indices_xy
                        .cont_slice()
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|_| node.indices_xy.into_iter().flatten().collect()),
                    indices_x: node
                        .indices_x
                        .cont_slice()
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|_| node.indices_x.into_iter().flatten().collect()),
                    indices_y: node
                        .indices_y
                        .cont_slice()
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|_| node.indices_y.into_iter().flatten().collect()),
                    feature_contributions,
                }
            })
            .collect()
    }

    pub fn tree_info(&self) -> String {
        let mut info = String::new();

        info.push_str("=== Tree Information ===\n");
        info.push_str(&format!("Total nodes: {}\n", self.nodes.len()));
        info.push_str(&format!("Total leaves: {}\n", self.leaves.len()));
        info.push_str(&format!(
            "Internal nodes: {}\n",
            self.nodes.len() - self.leaves.len()
        ));
        info.push_str(&format!(
            "Total splits performed: {}\n",
            self.split_history.len()
        ));

        // Calculate tree depth
        let max_depth = self.nodes.iter().map(|node| node.depth).max().unwrap_or(0);
        info.push_str(&format!("Maximum depth: {}\n", max_depth));

        // Leaf depth distribution
        let _leaf_depths: Vec<usize> = self
            .leaves
            .iter()
            .filter_map(|&idx| self.nodes.get(idx).map(|n| n.depth))
            .collect();

        // Split feature usage
        let mut feature_counts: HashMap<String, usize> = HashMap::new();
        for split in &self.split_history {
            *feature_counts.entry(split.column_name.clone()).or_insert(0) += 1;
        }

        if !feature_counts.is_empty() {
            info.push_str("\nFeature usage in splits:\n");
            let mut sorted_features: Vec<_> = feature_counts.iter().collect();
            sorted_features.sort_by(|a, b| b.1.cmp(a.1));
            for (feature, count) in sorted_features.iter() {
                let percentage = (**count as f64 / self.split_history.len() as f64) * 100.0;
                info.push_str(&format!("  {}: {} ({:.1}%)\n", feature, count, percentage));
            }
        }

        // Split type distribution
        let mut continuous_splits = 0;
        let mut categorical_splits = 0;
        let mut invalid_splits = 0;

        for split in &self.split_history {
            match &split.split_result {
                SplitResult::ContinuousSplit(_, _, _) => continuous_splits += 1,
                SplitResult::CategoricalSplit(_, _, _) => categorical_splits += 1,
                SplitResult::InvalidSplit(_) => invalid_splits += 1,
            }
        }

        info.push_str("\nSplit type distribution:\n");
        info.push_str(&format!("  Continuous: {}\n", continuous_splits));
        info.push_str(&format!("  Categorical: {}\n", categorical_splits));
        info.push_str(&format!("  Invalid: {}\n", invalid_splits));

        // Leaf reasons
        let mut reason_counts: HashMap<std::mem::Discriminant<LeafReason>, (usize, String)> =
            HashMap::new();

        for reason in self.leaf_reasons.values() {
            let discriminant = std::mem::discriminant(reason);
            let reason_str = match reason {
                LeafReason::MinSamplesSplit(_) => "Min samples split".to_string(),
                LeafReason::MinSamplesLeaf(_) => "Min samples leaf".to_string(),
                LeafReason::MaxDepth(_) => "Max depth reached".to_string(),
                LeafReason::NoValidSplit(_) => "No valid split".to_string(),
                LeafReason::IterationLimit => "Iteration limit".to_string(),
            };

            reason_counts
                .entry(discriminant)
                .and_modify(|(count, _)| *count += 1)
                .or_insert((1, reason_str));
        }

        info.push_str("\nLeaf stopping criteria:\n");
        for (_, (count, reason)) in reason_counts.iter() {
            let percentage = (*count as f64 / self.leaves.len() as f64) * 100.0;
            info.push_str(&format!("  {}: {} ({:.1}%)\n", reason, count, percentage));
        }

        // Node samples distribution
        let _leaf_samples: Vec<usize> = self
            .leaves
            .iter()
            .filter_map(|&idx| self.nodes.get(idx).map(|n| n.indices_xy.len()))
            .collect();

        info.push_str("========================\n");

        info
    }
}
struct BestFirstTreeBuilder {
    nodes: Vec<Node>,
    leaves: Vec<usize>,
    max_iter: usize,
    restrictions: SplitRestrictions,
    boundaries_expansion_factor: f64, // expand continuous target boundaries by this factor
    max_samples: Option<f64>,
    max_features: Option<f64>,
    exploration_split_budget: usize,
    feature_split_fraction: Option<f64>, // None means no restriction; Some(x) forces first x% of splits on non-target columns
    split_history: Vec<SplitRecord>,
    leaf_reasons: HashMap<usize, LeafReason>,
    rng: StdRng,
    status: TreeBuilderStatus,
}

impl BestFirstTreeBuilder {
    pub fn new(
        max_iter: usize,
        restrictions: SplitRestrictions,
        boundaries_expansion_factor: f64,
        max_samples: Option<f64>,
        max_features: Option<f64>,
        exploration_split_budget: usize,
        feature_split_fraction: Option<f64>,
        rng: StdRng,
    ) -> Self {
        Self {
            nodes: Vec::with_capacity(max_iter),
            leaves: Vec::with_capacity(max_iter),
            max_iter,
            restrictions,
            boundaries_expansion_factor,
            max_samples,
            max_features,
            exploration_split_budget,
            feature_split_fraction,
            split_history: Vec::new(),
            leaf_reasons: HashMap::new(),
            rng: rng,
            status: TreeBuilderStatus::SUCCESS,
        }
    }

    pub fn build_tree(&mut self, df: &DataFrame, sample_weights: &Float64Chunked) {
        let mut heap: BinaryHeap<SplitCandidate> = BinaryHeap::new();

        // Build order overview:
        // 1) Start with the root as a leaf.
        // 2) Run the optional geometric exploration phase first. It greedily splits the
        //    largest-volume leaves without looking at gain, to diversify early structure.
        // 3) Seed the gain-based heap from the post-exploration leaves and clear the
        //    leaf set so the best-first search can repopulate it with its own decisions.
        // 4) Continue best-first splitting until max_iter or the heap is empty.
        // 5) Any remaining heap entries after iteration budget become leaves (IterationLimit).

        // Initialize with root as a leaf so exploration (if any) can act first.
        let root = Node::default_from_dataframe(
            df,
            self.boundaries_expansion_factor,
            self.max_samples,
            &mut self.rng,
        );
        self.nodes.push(root.clone());
        self.leaves.clear();
        self.leaf_reasons.clear();
        self.leaves.push(0);

        // Optional geometric exploration phase runs before gain-based search.
        if self.exploration_split_budget > 0 {
            self.run_exploration(df);
        }

        // Build initial heap from all current leaves (post-exploration) and reset
        // the leaf set; gain-based search will repopulate final leaves.
        let frontier_leaves = self.leaves.clone();
        self.leaves.clear();
        self.leaf_reasons.clear();

        for leaf_idx in frontier_leaves {
            let node = self.nodes[leaf_idx].clone();
            let cand = self.evaluate_node(&node, leaf_idx, df, sample_weights);
            if cand.is_valid() {
                heap.push(cand);
            } else {
                // Keep as a leaf with reason for diagnostics.
                self.leaves.push(leaf_idx);
                self.leaf_reasons.insert(
                    leaf_idx,
                    LeafReason::NoValidSplit("No finite gain under this node.".to_string()),
                );
            }
        }

        if self.status != TreeBuilderStatus::SUCCESS {
            return;
        }

        let mut iters = 0usize;
        while let Some(candidate) = heap.pop() {
            if iters >= self.max_iter {
                // We've hit the iteration limit - make this popped candidate a leaf
                // instead of discarding it
                let idx = candidate.node_index;
                if !self.leaf_reasons.contains_key(&idx) {
                    self.leaves.push(idx);
                    self.leaf_reasons.insert(idx, LeafReason::IterationLimit);
                }
                break;
            }
            iters += 1;

            if !candidate.is_valid() {
                panic!("Should not be popping invalid candidates from heap");
            }

            let node = candidate.node;

            let node_idx = candidate.node_index;

            // Leaf checks
            let mut maybe_reason: Option<LeafReason> = None;
            let nxy = node.indices_xy.len();

            if nxy < self.restrictions.min_samples_split {
                maybe_reason = Some(LeafReason::MinSamplesSplit(nxy));
            } else if nxy < self.restrictions.min_samples_leaf {
                maybe_reason = Some(LeafReason::MinSamplesLeaf(nxy));
            } else if node.depth >= self.restrictions.max_depth {
                maybe_reason = Some(LeafReason::MaxDepth(node.depth));
            } else if !candidate.gain.is_finite() {
                maybe_reason = Some(LeafReason::NoValidSplit("Infinite gain".to_string()));
            }

            if let Some(reason) = maybe_reason {
                self.leaves.push(node_idx);
                self.leaf_reasons.insert(node_idx, reason);
                continue;
            }

            // Perform split
            let (left, right) = match &candidate.split_result {
                SplitResult::ContinuousSplit(v, _gain, none) => node.split(
                    df,
                    &candidate.column_name,
                    v as &dyn std::any::Any,
                    *none,
                    None,
                    None,
                ),
                SplitResult::CategoricalSplit(subset, _gain, none) => {
                    // Convert Vec<u32> to HashSet<u32> for the split
                    let subset_set: std::collections::HashSet<u32> =
                        subset.iter().cloned().collect();
                    node.split_categorical_subset(df, &candidate.column_name, &subset_set, *none)
                }
                SplitResult::InvalidSplit(msg) => {
                    self.leaves.push(node_idx);
                    self.leaf_reasons.insert(
                        node_idx,
                        LeafReason::NoValidSplit(format!(
                            "No valid split found for this node: {}",
                            msg
                        )),
                    );
                    continue;
                }
            };

            let parent_idx = node_idx;

            // Left child
            let left_idx = self.nodes.len();
            self.nodes.push(left.clone());
            let left_cand = self.evaluate_node(&left, left_idx, df, sample_weights);
            if left_cand.is_valid() {
                heap.push(left_cand);
            } else {
                self.leaves.push(left_idx);
                self.leaf_reasons.insert(
                    left_idx,
                    LeafReason::NoValidSplit("No finite gain under this node.".to_string()),
                );
            }

            // Right child
            let right_idx = self.nodes.len();
            self.nodes.push(right.clone());
            let right_cand = self.evaluate_node(&right, right_idx, df, sample_weights);
            if right_cand.is_valid() {
                heap.push(right_cand);
            } else {
                self.leaves.push(right_idx);
                self.leaf_reasons.insert(
                    right_idx,
                    LeafReason::NoValidSplit("No finite gain under this node.".to_string()),
                );
            }

            // record
            self.split_history.push(SplitRecord {
                parent_index: parent_idx,
                column_name: candidate.column_name.clone(),
                split_result: candidate.split_result.clone(),
                left_child_index: left_idx,
                right_child_index: right_idx,
            });
        }

        // Remaining candidates become leaves due to iteration limit
        while let Some(rem) = heap.pop() {
            let idx = rem.node_index;
            if !self.leaf_reasons.contains_key(&idx) {
                self.leaves.push(idx);
                self.leaf_reasons.insert(idx, LeafReason::IterationLimit);
            }
        }
    }

    fn n_features_to_consider(&self, df: &DataFrame) -> Option<usize> {
        if self.max_features.is_none() {
            return None;
        }

        let total_features = df
            .get_column_names()
            .into_iter()
            .filter(|&name| !name.starts_with(TARGET_PREFIX))
            .collect::<Vec<_>>()
            .len();

        match self.max_features {
            Some(fraction) if fraction > 0.0 && fraction < 1.0 => {
                let n = (fraction * (total_features as f64)).ceil() as usize;
                Some(n.max(1).min(total_features))
            }
            Some(n) if n > 1.0 + 1e-7 => {
                let n = n as usize;
                Some(n.min(total_features))
            }
            _ => None,
        }
    }

    fn evaluate_node(
        &mut self,
        node: &Node,
        idx: usize,
        df: &DataFrame,
        sample_weights: &Float64Chunked,
    ) -> SplitCandidate {
        // Calculate whether to exclude targets based on feature_split_fraction
        // None means no restriction (exclude_targets = false always)
        let exclude_targets = match self.feature_split_fraction {
            Some(fraction) => {
                let threshold = (fraction * self.max_iter as f64).ceil() as usize;
                let current_splits = self.split_history.len();
                current_splits < threshold
            }
            None => false,
        };

        let (col, split_result) = find_best_split_column(
            node,
            df,
            &self.restrictions,
            self.n_features_to_consider(&df),
            &mut self.rng,
            sample_weights,
            exclude_targets,
        );
        let gain = split_result.gain();

        SplitCandidate {
            gain,
            column_name: col,
            split_result,
            node: node.clone(),
            node_index: idx,
        }
    }

    /// Geometric exploration: repeatedly split the leaf with the largest volume
    /// along its widest axis. For continuous axes:
    /// - Finite interval: geometric mean of bounds (or midpoint if not both positive).
    /// - [a, +∞]: split at a + 2*(max-min of samples).
    /// - [-∞, b]: split at b - 2*(max-min of samples).
    fn run_exploration(&mut self, df: &DataFrame) {
        #[derive(Clone)]
        struct ExplorationCandidate {
            node_index: usize,
            volume: f64,
        }

        impl PartialEq for ExplorationCandidate {
            fn eq(&self, other: &Self) -> bool {
                self.volume == other.volume
            }
        }
        impl Eq for ExplorationCandidate {}
        impl PartialOrd for ExplorationCandidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.volume.partial_cmp(&other.volume)
            }
        }
        impl Ord for ExplorationCandidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut budget = self.exploration_split_budget;
        if budget == 0 {
            return;
        }

        let mut heap: BinaryHeap<ExplorationCandidate> = BinaryHeap::new();
        // Exploration strategy (gain-agnostic):
        // * Prioritize leaves by estimated cell volume (product of axis lengths).
        // * Within the chosen leaf, pick the widest axis; random tie-break avoids
        //   always splitting the same column when widths match.
        // * Continuous axes split at the median of the leaf samples; categorical axes
        //   split on the first observed category.
        // * Repeat until the exploration budget is exhausted, pushing children back
        //   into the volume-priority queue so large subcells keep getting refined.
        for &leaf_idx in &self.leaves {
            let has_samples = self
                .nodes
                .get(leaf_idx)
                .map(|n| n.indices_xy.len() > 0)
                .unwrap_or(false);
            if has_samples {
                if let Some((vol, _)) = self.cell_volume_and_lengths(leaf_idx) {
                    if vol.is_finite() && vol > 0.0 {
                        heap.push(ExplorationCandidate {
                            node_index: leaf_idx,
                            volume: vol,
                        });
                    }
                }
            }
        }

        while budget > 0 {
            let Some(candidate) = heap.pop() else {
                break;
            };
            if !self.leaves.contains(&candidate.node_index) {
                continue;
            }

            let node = self.nodes.get(candidate.node_index).cloned();
            let Some(node) = node else {
                continue;
            };
            let (_, axis_lengths) = match self.cell_volume_and_lengths(candidate.node_index) {
                Some(v) => v,
                None => continue,
            };

            // Pick the widest axis (continuous or categorical) with random tie-breaks
            let mut max_len = f64::NEG_INFINITY;
            for (_, len, _) in axis_lengths.iter() {
                if len.is_finite() && *len > max_len {
                    max_len = *len;
                }
            }

            if !max_len.is_finite() {
                continue;
            }

            let candidates: Vec<_> = axis_lengths
                .iter()
                .filter(|(_, len, _)| (*len - max_len).abs() < 1e-12)
                .collect();

            let maybe_axis = if candidates.is_empty() {
                None
            } else {
                let idx = if candidates.len() == 1 {
                    0
                } else {
                    (self.rng.next_u64() as usize) % candidates.len()
                };
                Some(candidates[idx])
            };

            let (axis_name, _len, is_continuous) = match maybe_axis {
                Some(v) => v,
                None => continue,
            };

            let (split_result, left, right) = if *is_continuous {
                let Some(split_point) = self.exploration_split_point(df, &node, axis_name) else {
                    continue;
                };
                let (left, right) = node.split(df, axis_name, &split_point, None, None, None);
                (
                    SplitResult::ContinuousSplit(split_point, 0.0, None),
                    left,
                    right,
                )
            } else {
                let Some((point_box, split_code)) =
                    self.categorical_value_for_column(df, &node, axis_name)
                else {
                    continue;
                };
                // For exploration, use single value as subset for categorical split
                let subset_set: std::collections::HashSet<u32> =
                    std::iter::once(split_code as u32).collect();
                let (left, right) = node.split_categorical_subset(df, axis_name, &subset_set, None);
                (
                    SplitResult::CategoricalSplit(vec![split_code as u32], 0.0, None),
                    left,
                    right,
                )
            };

            let parent_idx = candidate.node_index;
            self.leaves.retain(|&i| i != parent_idx);
            self.leaf_reasons.remove(&parent_idx);

            let left_idx = self.nodes.len();
            self.nodes.push(left.clone());
            let right_idx = self.nodes.len();
            self.nodes.push(right.clone());

            self.leaves.push(left_idx);
            self.leaves.push(right_idx);

            self.split_history.push(SplitRecord {
                parent_index: parent_idx,
                column_name: axis_name.to_string(),
                split_result,
                left_child_index: left_idx,
                right_child_index: right_idx,
            });

            budget -= 1;

            if left.indices_xy.len() > 0 {
                if let Some((vol_left, _)) = self.cell_volume_and_lengths(left_idx) {
                    if vol_left.is_finite() && vol_left > 0.0 {
                        heap.push(ExplorationCandidate {
                            node_index: left_idx,
                            volume: vol_left,
                        });
                    }
                }
            }
            if right.indices_xy.len() > 0 {
                if let Some((vol_right, _)) = self.cell_volume_and_lengths(right_idx) {
                    if vol_right.is_finite() && vol_right > 0.0 {
                        heap.push(ExplorationCandidate {
                            node_index: right_idx,
                            volume: vol_right,
                        });
                    }
                }
            }
        }
    }

    /// Return (volume, axis_lengths) where axis_lengths = (name, length, is_continuous)
    /// only over X (non-target) partitions.
    fn cell_volume_and_lengths(&self, node_idx: usize) -> Option<(f64, Vec<(String, f64, bool)>)> {
        let node = self.nodes.get(node_idx)?;
        let axis_lengths = node.cell.axis_lengths_non_target();
        if axis_lengths.is_empty() {
            return None;
        }
        let volume = axis_lengths.iter().map(|(_, l, _)| *l).product::<f64>();
        Some((volume, axis_lengths))
    }

    fn median_for_column(&self, df: &DataFrame, col: &str, indices: &UInt32Chunked) -> Option<f64> {
        let series = df.column(col).ok()?.as_series()?.clone();
        let taken = series.take(indices).ok()?;
        let taken_f = taken.cast(&DataType::Float64).ok()?;
        let ca = taken_f.f64().ok()?;
        ca.median()
    }

    /// Compute the exploration split point for a continuous axis.
    /// - If axis length is finite: use the geometric mean of (low, high).
    /// - If [a, +∞]: split at a + 2 * (max(samples) - min(samples)).
    /// - If [-∞, b]: split at b - 2 * (max(samples) - min(samples)).
    fn exploration_split_point(&self, df: &DataFrame, node: &Node, col: &str) -> Option<f64> {
        use crate::rules::ContinuousInterval;

        // Get the interval bounds from the cell partition
        let partition = node.cell.partitions.get(col)?;
        let interval = partition.rule_any().downcast_ref::<ContinuousInterval>()?;
        let low = interval.low;
        let high = interval.high;

        let axis_length = high - low;

        if axis_length.is_finite() && low > 0.0 && high > 0.0 {
            // Finite positive interval: geometric mean
            Some((low * high).sqrt())
        } else if axis_length.is_finite() {
            // Finite interval but not both positive: arithmetic midpoint
            Some((low + high) / 2.0)
        } else {
            // Infinite interval: use sample range
            let sample_range = self.sample_range_for_column(df, col, &node.indices_xy)?;
            if sample_range <= 0.0 {
                // Fallback to median if samples have no range
                return self.median_for_column(df, col, &node.indices_xy);
            }
            let expansion = 2.0 * sample_range;

            if low.is_finite() && high.is_infinite() {
                // [a, +∞] case
                Some(low + expansion)
            } else if low.is_infinite() && high.is_finite() {
                // [-∞, b] case
                Some(high - expansion)
            } else {
                // Both infinite: fallback to median
                self.median_for_column(df, col, &node.indices_xy)
            }
        }
    }

    /// Compute max(samples) - min(samples) for a column within the given indices.
    fn sample_range_for_column(
        &self,
        df: &DataFrame,
        col: &str,
        indices: &UInt32Chunked,
    ) -> Option<f64> {
        let series = df.column(col).ok()?.as_series()?.clone();
        let taken = series.take(indices).ok()?;
        let taken_f = taken.cast(&DataType::Float64).ok()?;
        let ca = taken_f.f64().ok()?;
        let min_val = ca.min()?;
        let max_val = ca.max()?;
        Some(max_val - min_val)
    }

    /// Pick any categorical value present in the node for exploration splitting.
    /// Returns (boxed_point, split_code_for_record).
    fn categorical_value_for_column(
        &self,
        df: &DataFrame,
        node: &Node,
        col: &str,
    ) -> Option<(Box<dyn Any>, usize)> {
        let series = df.column(col).ok()?.as_series()?.clone();
        let taken = series.take(&node.indices_xy).ok()?;
        if taken.len() == 0 {
            return None;
        }

        // Enum / categorical: use physical u32 codes
        if matches!(
            taken.dtype(),
            DataType::Enum(_, _) | DataType::Categorical(_, _)
        ) {
            let phys = taken.to_physical_repr();
            let phys_u32 = phys.cast(&DataType::UInt32).ok()?;
            let ca = phys_u32.u32().ok()?;
            if let Some(val) = ca.into_iter().flatten().next() {
                let idx = val as usize;
                return Some((Box::new(val), idx));
            }
        }

        // Boolean categories
        if taken.dtype() == &DataType::Boolean {
            if let Some(val) = taken.bool().ok()?.into_iter().flatten().next() {
                let idx = if val { 1 } else { 0 };
                return Some((Box::new(val), idx));
            }
        }

        // String categories
        if taken.dtype() == &DataType::String {
            if let Some(val) = taken.str().ok()?.into_iter().flatten().next() {
                let val_owned = val.to_string();
                let idx = node
                    .cell
                    .partitions
                    .get(col)
                    .and_then(|p| p.rule_any().downcast_ref::<BelongsToString>())
                    .and_then(|r| r.domain.iter().position(|v| v == &val_owned))
                    .unwrap_or(0);
                return Some((Box::new(val_owned), idx));
            }
        }

        // Raw unsigned codes
        if let Ok(ca) = taken.u32() {
            if let Some(val) = ca.into_iter().flatten().next() {
                let idx = val as usize;
                return Some((Box::new(val), idx));
            }
        }

        // Fallback: signed codes
        if let Ok(ca) = taken.i64() {
            if let Some(val) = ca.into_iter().flatten().next() {
                let idx = val as usize;
                return Some((Box::new(val), idx));
            }
        }

        None
    }
}

#[allow(dead_code)]
fn print_split_summary(split_history: &[SplitRecord], leaf_reasons: &HashMap<usize, LeafReason>) {
    println!("=== Tree Building Summary ===");
    println!("Total splits performed: {}", split_history.len());

    if !split_history.is_empty() {
        println!("\nSplit details:");
        for (i, split) in split_history.iter().enumerate() {
            let split_type = match &split.split_result {
                SplitResult::ContinuousSplit(value, gain, _) => {
                    format!("Continuous (value: {:.4}, gain: {:.4})", value, gain)
                }
                SplitResult::CategoricalSplit(subset, gain, _) => {
                    format!("Categorical (subset: {:?}, gain: {:.4})", subset, gain)
                }
                SplitResult::InvalidSplit(msg) => {
                    format!("Invalid ({})", msg)
                }
            };
            println!(
                "  {}. Node {} -> {} on '{}' | Left: {}, Right: {}",
                i + 1,
                split.parent_index,
                split_type,
                split.column_name,
                split.left_child_index,
                split.right_child_index
            );
        }
    }

    println!("\nLeaf summary:");
    let mut reason_counts: HashMap<std::mem::Discriminant<LeafReason>, (usize, String)> =
        HashMap::new();

    for reason in leaf_reasons.values() {
        let discriminant = std::mem::discriminant(reason);
        let reason_str = match reason {
            LeafReason::MinSamplesSplit(n) => format!("Min samples split ({})", n),
            LeafReason::MinSamplesLeaf(n) => format!("Min samples leaf ({})", n),
            LeafReason::MaxDepth(d) => format!("Max depth ({})", d),
            LeafReason::NoValidSplit(msg) => format!("No valid split ({})", msg),
            LeafReason::IterationLimit => "Iteration limit".to_string(),
        };

        reason_counts
            .entry(discriminant)
            .and_modify(|(count, _)| *count += 1)
            .or_insert((1, reason_str));
    }

    println!("  Total leaves: {}", leaf_reasons.len());
    for (_, (count, reason)) in reason_counts.iter() {
        println!("    {}: {} leaves", reason, count);
    }
    println!("========================\n");
}
