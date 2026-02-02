use crate::serde::schema as schema_serde;
use crate::{conf::TARGET_PREFIX, predict::probability::PiecewiseConstantDistribution, tree::*};
use estimators::api::{Estimator, FitError, PredictError};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub struct PartitionForest {
    pub n_estimators: usize,
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
    pub max_features: Option<f64>,
    pub max_samples: Option<f64>,
    pub exploration_split_budget: usize,
    pub feature_split_fraction: Option<f64>,
    pub seed: Option<usize>,
    pub tree: Option<Vec<Tree>>,
    #[serde(with = "schema_serde")]
    pub schema: Option<Schema>,
}

impl PartitionForest {
    pub fn new(
        n_estimators: usize,
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
        max_features: Option<f64>,
        max_samples: Option<f64>,
        exploration_split_budget: usize,
        feature_split_fraction: Option<f64>,
        seed: Option<usize>,
    ) -> Self {
        PartitionForest {
            n_estimators,
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
            max_features: max_features,
            max_samples: max_samples,
            exploration_split_budget,
            feature_split_fraction,
            seed: seed,
            tree: None,
            schema: None,
        }
    }
    pub fn default() -> Self {
        PartitionForest {
            n_estimators: 100,
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
            max_features: Some(0.8),
            max_samples: Some(1.0),
            exploration_split_budget: 0,
            feature_split_fraction: None,
            seed: None,
            tree: None,
            schema: None,
        }
    }

    pub fn predict_proba(
        &self,
        x: &DataFrame,
    ) -> Result<Vec<PiecewiseConstantDistribution>, PredictError> {
        self._predict_proba_impl(x)
    }

    pub fn predict_trees_proba(
        &self,
        x: &DataFrame,
    ) -> Result<Vec<Vec<PiecewiseConstantDistribution>>, PredictError> {
        let tree = match self.tree {
            Some(ref t) => t,
            None => return Err(PredictError::NotFitted),
        };

        let per_tree_proba: Vec<Vec<PiecewiseConstantDistribution>> =
            tree.par_iter().map(|tree| tree.predict_proba(x)).collect();

        Ok(per_tree_proba)
    }

    fn _predict_proba_impl(
        &self,
        x: &DataFrame,
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

        let per_tree_proba: Vec<Vec<PiecewiseConstantDistribution>> =
            tree.par_iter().map(|tree| tree.predict_proba(x)).collect();

        if per_tree_proba.is_empty() {
            return Ok(Vec::new());
        }

        let n_samples = per_tree_proba[0].len();
        debug_assert!(per_tree_proba
            .iter()
            .all(|predictions| predictions.len() == n_samples));

        let proba: Vec<PiecewiseConstantDistribution> = (0..n_samples)
            .map(|sample_idx| {
                let sample_iter = per_tree_proba
                    .iter()
                    .map(|predictions| &predictions[sample_idx]);
                PiecewiseConstantDistribution::unified_from(sample_iter)
            })
            .collect();

        Ok(proba)
    }
}
impl Estimator for PartitionForest {
    fn _fit_impl(
        &mut self,
        x: &DataFrame,
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

        let xy = polars::functions::concat_df_horizontal(&[x.clone(), y], true)
            .map_err(|e| FitError::InvalidInput(e.to_string()))?;

        let schema = xy.schema().as_ref().clone();

        let base_sample_weights = sample_weights.cloned();

        // Ensure each tree gets a distinct seed even when `self.seed` is None
        let base_seed = self.seed.unwrap_or(42);

        let fitted_trees: Vec<Tree> = (0..self.n_estimators)
            .into_par_iter()
            .map(|idx| {
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
                    self.max_samples,
                    self.max_features,
                    self.exploration_split_budget,
                    self.feature_split_fraction,
                    Some(base_seed + idx),
                );
                tree.fit(&xy, base_sample_weights.clone());
                tree
            })
            .collect();

        self.tree = Some(fitted_trees);

        Ok(PartitionForest {
            n_estimators: self.n_estimators,
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
            max_features: self.max_features,
            max_samples: self.max_samples,
            exploration_split_budget: self.exploration_split_budget,
            feature_split_fraction: self.feature_split_fraction,
            seed: self.seed,
            tree: self.tree.take(), // Move the tree out of self
            schema: Some(schema),
        })
    }

    fn _predict_impl(&self, x: &DataFrame) -> Result<DataFrame, PredictError> {
        let tree = match self.tree {
            Some(ref t) => t,
            None => return Err(PredictError::NotFitted),
        };

        // Make tree_schema an Option<PartitionDataFrame> (owned, cloned)
        let tree_schema = self.schema.as_ref().cloned();
        if tree_schema.is_none() {
            panic!("Tree schema is not set");
        }

        let proba = self.predict_proba(x)?;
        let mean_vector_or_rows: Vec<HashMap<String, Vec<f64>>> =
            proba.iter().map(|p| p.mean()).collect();

        let mut out = tree[0].decode_mean_from_mean_vector(&mean_vector_or_rows);

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
