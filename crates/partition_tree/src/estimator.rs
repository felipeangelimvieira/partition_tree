use crate::{conf::TARGET_PREFIX, predict::probability::PiecewiseConstantDistribution, tree::*};
use estimator_api::api::{Estimator, FitError, PredictError};
use polars::prelude::*;
use std::collections::HashMap;

// Re-export for use in Python bindings
pub use crate::tree::{LeafInfo, PartitionInfo};

pub struct PartitionTree {
    pub max_iter: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf_y: usize,
    pub min_samples_leaf_x: usize,
    pub min_samples_leaf: usize,
    pub max_depth: usize,
    pub min_target_volume: f64,
    pub min_split_gain: f64,
    pub min_density_value: f64,
    pub max_density_value: f64,
    pub max_measure_value: f64,
    pub boundaries_expansion_factor: f64,
    pub exploration_split_budget: usize,
    pub feature_split_fraction: Option<f64>,
    pub seed: Option<usize>,
    pub tree: Option<Tree>,
    pub schema: Option<Schema>,
}

impl PartitionTree {
    pub fn new(
        max_iter: usize,
        min_samples_split: usize,
        min_samples_leaf_y: usize,
        min_samples_leaf_x: usize,
        min_samples_leaf: usize,
        min_target_volume: f64,
        max_depth: usize,
        min_split_gain: f64,
        min_density_value: f64,
        max_density_value: f64,
        max_measure_value: f64,
        boundaries_expansion_factor: f64,
        exploration_split_budget: usize,
        feature_split_fraction: Option<f64>,
        seed: Option<usize>,
    ) -> Self {
        PartitionTree {
            max_iter: max_iter,
            min_samples_split: min_samples_split,
            min_samples_leaf_y: min_samples_leaf_y,
            min_samples_leaf_x: min_samples_leaf_x,
            min_samples_leaf: min_samples_leaf,
            min_target_volume: min_target_volume,
            max_depth: max_depth,
            min_split_gain: min_split_gain,
            min_density_value: min_density_value,
            max_density_value: max_density_value,
            max_measure_value: max_measure_value,
            boundaries_expansion_factor: boundaries_expansion_factor,
            exploration_split_budget,
            feature_split_fraction,
            seed: seed,
            tree: None,
            schema: None,
        }
    }
    pub fn default() -> Self {
        PartitionTree {
            max_iter: 100,
            min_samples_split: 2,
            min_samples_leaf_y: 1,
            min_samples_leaf_x: 1,
            min_samples_leaf: 1,
            min_target_volume: 0.0,
            max_depth: usize::MAX,
            min_split_gain: 0.0,
            min_density_value: 0.0,
            max_density_value: f64::INFINITY,
            max_measure_value: f64::INFINITY,
            boundaries_expansion_factor: 0.1,
            exploration_split_budget: 0,
            feature_split_fraction: None,
            seed: None,
            tree: None,
            schema: None,
        }
    }

    pub fn predict_proba(
        &self,
        X: &DataFrame,
    ) -> Result<Vec<PiecewiseConstantDistribution>, PredictError> {
        self._predict_proba_impl(&X)
    }

    fn _predict_proba_impl(
        &self,
        X: &DataFrame,
    ) -> Result<Vec<PiecewiseConstantDistribution>, PredictError> {
        let tree = match self.tree {
            Some(ref t) => t,
            None => return Err(PredictError::NotFitted),
        };

        // Make tree_schema an Option<PartitionDataFrame> (owned, cloned)
        let tree_schema = self.schema.as_ref().cloned();
        if tree_schema.is_none() {
            panic!("Tree schema is not set");
        }

        let proba = tree.predict_proba(&X);

        Ok(proba)
    }

    pub fn tree_info(&self) -> String {
        self.tree.as_ref().unwrap().tree_info()
    }

    pub fn tree_build_status(&self) -> String {
        match &self.tree {
            Some(t) => match &t.get_build_status() {
                TreeBuilderStatus::SUCCESS => "SUCCESS".to_string(),
                TreeBuilderStatus::FAILED(msg) => format!("Tree build failed: {}", msg),
            },
            None => "Tree not fitted".to_string(),
        }
    }

    /// Get detailed information about all leaves in the tree
    pub fn get_leaves_info(&self) -> Vec<LeafInfo> {
        match &self.tree {
            Some(t) => t.get_leaves_info(),
            None => Vec::new(),
        }
    }

    /// Compute feature importances based on cumulative gain from all splits.
    /// Each split is counted exactly once (no double-counting across leaves).
    /// If `normalize` is true, the importances are normalized to sum to 1.0.
    pub fn get_feature_importances(&self, normalize: bool) -> HashMap<String, f64> {
        match &self.tree {
            Some(t) => t.get_feature_importances(normalize),
            None => HashMap::new(),
        }
    }

    /// Apply the tree to the input data, returning the leaf index for each sample.
    /// Returns a Vec where each element is a Vec of leaf indices (positions in the leaves array)
    /// that the sample belongs to. Use get_leaves_info() to get details about each leaf.
    pub fn apply(&self, x: &DataFrame) -> Vec<Vec<usize>> {
        match &self.tree {
            Some(t) => t.apply(x),
            None => Vec::new(),
        }
    }

    pub fn predict_categorical_masses(
        &self,
        x: &DataFrame,
    ) -> Result<Vec<Vec<(f64, HashMap<String, Vec<String>>)>>, PredictError> {
        let dists = self.predict_proba(x)?;
        Ok(dists
            .into_iter()
            .map(|dist| dist.masses_with_categories())
            .collect())
    }
}
impl Estimator for PartitionTree {
    fn _fit_impl(
        &mut self,
        X: &DataFrame,
        y: &DataFrame,
        sample_weights: Option<&Float64Chunked>,
    ) -> Result<Self, FitError> {
        let mut y = y.clone();

        // Collect column names first to avoid borrowing conflicts
        let column_names: Vec<String> = y
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();

        for column_name in column_names {
            if !column_name.starts_with(TARGET_PREFIX) {
                y.rename(
                    &column_name,
                    PlSmallStr::from_str(&format!("{}_{}", TARGET_PREFIX, column_name)),
                )
                .map_err(|e| FitError::InvalidInput(e.to_string()))?;
            }
        }

        let XY = polars::functions::concat_df_horizontal(&[X.clone(), y], true)
            .map_err(|e| FitError::InvalidInput(e.to_string()))?;

        let schema = XY.schema().as_ref().clone();
        let mut tree = Tree::new(
            self.max_iter,
            self.min_samples_split,
            self.min_samples_leaf_y,
            self.min_samples_leaf_x,
            self.min_samples_leaf,
            self.max_depth,
            self.min_target_volume,
            self.min_split_gain,
            self.min_density_value,
            self.max_density_value,
            self.max_measure_value,
            self.boundaries_expansion_factor,
            None,
            None,
            self.exploration_split_budget,
            self.feature_split_fraction,
            self.seed,
        );

        tree.fit(&XY, sample_weights.cloned());
        self.tree = Some(tree);

        Ok(PartitionTree {
            max_iter: self.max_iter,
            min_samples_split: self.min_samples_split,
            min_samples_leaf_y: self.min_samples_leaf_y,
            min_samples_leaf_x: self.min_samples_leaf_x,
            min_samples_leaf: self.min_samples_leaf,
            min_target_volume: self.min_target_volume,
            max_depth: self.max_depth,
            min_split_gain: self.min_split_gain,
            min_density_value: self.min_density_value,
            max_density_value: self.max_density_value,
            max_measure_value: self.max_measure_value,
            boundaries_expansion_factor: self.boundaries_expansion_factor,
            exploration_split_budget: self.exploration_split_budget,
            feature_split_fraction: self.feature_split_fraction,
            seed: self.seed,
            tree: self.tree.take(), // Move the tree out of self
            schema: Some(schema),
        })
    }

    fn _predict_impl(&self, X: &DataFrame) -> Result<DataFrame, PredictError> {
        let tree = match self.tree {
            Some(ref t) => t,
            None => return Err(PredictError::NotFitted),
        };

        // Make tree_schema an Option<PartitionDataFrame> (owned, cloned)
        let tree_schema = self.schema.as_ref().cloned();
        if tree_schema.is_none() {
            panic!("Tree schema is not set");
        }

        let mut out = tree.predict_mean(&X);

        // Remove target_ prefix from column names
        let column_names_without_prefix = &out
            .get_column_names()
            .iter()
            .map(|&name| {
                if name.starts_with("target_") {
                    name["target_".len()..].to_string()
                } else {
                    name.to_string()
                }
            })
            .collect::<Vec<String>>();
        out.set_column_names(column_names_without_prefix)
            .expect("Failed to set column names without 'target_' prefix after prediction");

        Ok(out)
    }
}
