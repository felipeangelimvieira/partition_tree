//! Ensemble of v2 partition trees (random forest).
//!
//! [`PartitionForestV2`] fits multiple independent trees in parallel using
//! [`rayon`], sharing a single [`DatasetView`] across all builders for
//! efficiency. At prediction time, per-tree distributions are merged via
//! [`PiecewiseConstantDistribution::ensemble`].
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use partition_tree::v2::forest::PartitionForestV2;
//! use estimators::api::Estimator;
//!
//! let mut model = PartitionForestV2::default();
//! let fitted = model.fit(&x_train, &y_train, None).unwrap();
//! let preds  = fitted.predict(&x_test).unwrap();
//! let dists  = fitted.predict_proba(&x_test).unwrap();
//! ```
use std::collections::HashMap;
use std::sync::Arc;

use estimators::api::{Estimator, FitError, PredictError};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::conf::TARGET_PREFIX;
use crate::serde::schema as schema_serde;

use super::dataset_view::{DatasetView, PolarsDatasetView};
use super::dtype_plugin::DTypeRegistry;
use super::loss::{ConditionalLogLoss, LossFunc};
use super::predict::piecewise_distribution::{MeanVector, PiecewiseConstantDistribution};
use super::split_result::SplitRestrictions;
use super::tree::Tree;
use super::tree_builder::{TreeBuilder, TreeBuilderConfig};

// ---------------------------------------------------------------------------
// PartitionForestV2
// ---------------------------------------------------------------------------

/// Ensemble estimator wrapping multiple v2 partition trees.
///
/// Trees are grown in parallel over a shared [`DatasetView`]. Each tree
/// receives its own deterministic seed derived from `base_seed + tree_index`.
///
/// # Prediction
///
/// | Method                | Returns                                       |
/// |-----------------------|-----------------------------------------------|
/// | `predict`             | `DataFrame` (ensemble-averaged point preds)   |
/// | `predict_proba`       | `Vec<PiecewiseConstantDistribution>`           |
/// | `predict_trees_proba` | `Vec<Vec<PiecewiseConstantDistribution>>`      |
/// | `predict_mean_vectors`| `Vec<MeanVector>`                              |
/// | `feature_importances` | `HashMap<String, f64>`                         |
#[derive(Serialize, Deserialize)]
pub struct PartitionForestV2 {
    // ── Forest-level parameters ───────────────────────────────────────
    /// Number of trees in the ensemble.
    pub n_estimators: usize,
    /// Base random seed. Tree `i` receives `base_seed + i`.
    pub seed: Option<usize>,

    // ── Per-tree builder configuration ────────────────────────────────
    /// Maximum number of leaves per tree.
    pub max_leaves: usize,
    /// Expansion factor applied to target column boundaries.
    pub boundaries_expansion_factor: f64,
    /// Minimum w_xy in each child.
    pub min_samples_xy: f64,
    /// Minimum w_x in each child.
    pub min_samples_x: f64,
    /// Minimum w_y in each child.
    pub min_samples_y: f64,
    /// Minimum information gain for a split.
    pub min_gain: f64,
    /// Minimum target volume in each child.
    pub min_volume: f64,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum total samples in a parent to attempt a split.
    pub min_samples_split: f64,

    // ── Fitted state ──────────────────────────────────────────────────
    /// Fitted trees (populated after `fit`).
    pub trees: Option<Vec<Tree>>,
    /// Schema of the XY DataFrame used for fitting.
    #[serde(with = "schema_serde")]
    pub schema: Option<Schema>,
}

impl PartitionForestV2 {
    /// Create a new forest estimator with full control over parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_estimators: usize,
        max_leaves: usize,
        boundaries_expansion_factor: f64,
        min_samples_xy: f64,
        min_samples_x: f64,
        min_samples_y: f64,
        min_gain: f64,
        min_volume: f64,
        max_depth: usize,
        min_samples_split: f64,
        seed: Option<usize>,
    ) -> Self {
        Self {
            n_estimators,
            seed,
            max_leaves,
            boundaries_expansion_factor,
            min_samples_xy,
            min_samples_x,
            min_samples_y,
            min_gain,
            min_volume,
            max_depth,
            min_samples_split,
            trees: None,
            schema: None,
        }
    }

    /// Sensible defaults for the forest.
    pub fn with_defaults() -> Self {
        Self {
            n_estimators: 100,
            seed: Some(42),
            max_leaves: 101,
            boundaries_expansion_factor: 0.1,
            min_samples_xy: 1.0,
            min_samples_x: 1.0,
            min_samples_y: 1.0,
            min_gain: 0.0,
            min_volume: 0.0,
            max_depth: usize::MAX,
            min_samples_split: 2.0,
            trees: None,
            schema: None,
        }
    }

    // ── Prediction helpers ─────────────────────────────────────────────

    /// Predict ensemble-averaged conditional distributions for every row.
    ///
    /// Returns one [`PiecewiseConstantDistribution`] per row, formed by
    /// merging the per-tree leaf distributions via
    /// [`PiecewiseConstantDistribution::ensemble`].
    pub fn predict_proba(
        &self,
        x: &DataFrame,
    ) -> Result<Vec<PiecewiseConstantDistribution>, PredictError> {
        let trees = self.fitted_trees()?;
        let xy = self.expand_with_schema(x)?;
        let dataset = PolarsDatasetView::new(&xy);

        // Collect per-tree distributions in parallel
        let per_tree: Vec<Vec<PiecewiseConstantDistribution>> = trees
            .par_iter()
            .map(|tree| tree.predict_distributions(&dataset))
            .collect();

        Ok(Self::merge_distributions(&per_tree))
    }

    /// Predict per-tree conditional distributions (no ensembling).
    ///
    /// Returns `Vec<Vec<PiecewiseConstantDistribution>>` where the outer
    /// index is the tree and the inner index is the row.
    pub fn predict_trees_proba(
        &self,
        x: &DataFrame,
    ) -> Result<Vec<Vec<PiecewiseConstantDistribution>>, PredictError> {
        let trees = self.fitted_trees()?;
        let xy = self.expand_with_schema(x)?;
        let dataset = PolarsDatasetView::new(&xy);

        let per_tree: Vec<Vec<PiecewiseConstantDistribution>> = trees
            .par_iter()
            .map(|tree| tree.predict_distributions(&dataset))
            .collect();

        Ok(per_tree)
    }

    /// Predict ensemble-averaged mean vectors for every row.
    pub fn predict_mean_vectors(&self, x: &DataFrame) -> Result<Vec<MeanVector>, PredictError> {
        let dists = self.predict_proba(x)?;
        Ok(dists.into_iter().map(|d| d.mean_vector()).collect())
    }

    /// Feature importances aggregated across all trees.
    ///
    /// Each tree's importances are summed. If `normalize` is true, the
    /// result sums to 1.0.
    pub fn feature_importances(
        &self,
        normalize: bool,
    ) -> Result<HashMap<String, f64>, PredictError> {
        let trees = self.fitted_trees()?;

        let mut importances: HashMap<String, f64> = HashMap::new();
        for tree in trees {
            for (col, gain) in tree.feature_importances(false) {
                *importances.entry(col).or_insert(0.0) += gain;
            }
        }

        if normalize && !importances.is_empty() {
            let total: f64 = importances.values().sum();
            if total > 0.0 {
                for v in importances.values_mut() {
                    *v /= total;
                }
            }
        }

        Ok(importances)
    }

    /// Number of fitted trees (0 if not fitted).
    pub fn n_trees(&self) -> usize {
        self.trees.as_ref().map_or(0, |t| t.len())
    }

    // ── Internal helpers ───────────────────────────────────────────────

    /// Get the fitted trees or return `PredictError::NotFitted`.
    fn fitted_trees(&self) -> Result<&[Tree], PredictError> {
        self.trees.as_deref().ok_or(PredictError::NotFitted)
    }

    /// Expand X-only features into the full XY schema expected by trees.
    fn expand_with_schema(&self, x: &DataFrame) -> Result<DataFrame, PredictError> {
        let schema = self.schema.as_ref().ok_or(PredictError::NotFitted)?;

        let mut cols: Vec<Column> = x.get_columns().to_vec();
        let n_rows = x.height();

        for (col_name, dtype) in schema.iter() {
            if col_name.starts_with(TARGET_PREFIX) && x.column(col_name.as_str()).is_err() {
                let placeholder = Series::full_null(col_name.clone(), n_rows, dtype);
                cols.push(placeholder.into());
            }
        }

        DataFrame::new(cols).map_err(|e| {
            PredictError::InvalidInput(format!("Failed to build prediction DataFrame: {e}"))
        })
    }

    /// Build a `TreeBuilderConfig` from our fields.
    fn build_config(&self) -> TreeBuilderConfig {
        TreeBuilderConfig {
            max_leaves: self.max_leaves,
            boundaries_expansion_factor: self.boundaries_expansion_factor,
            restrictions: SplitRestrictions {
                min_samples_xy: self.min_samples_xy,
                min_samples_x: self.min_samples_x,
                min_samples_y: self.min_samples_y,
                min_gain: self.min_gain,
                min_volume: self.min_volume,
                max_depth: self.max_depth,
                min_samples_split: self.min_samples_split,
            },
        }
    }

    /// Merge per-tree distributions into per-sample ensembled distributions.
    fn merge_distributions(
        per_tree: &[Vec<PiecewiseConstantDistribution>],
    ) -> Vec<PiecewiseConstantDistribution> {
        if per_tree.is_empty() {
            return Vec::new();
        }

        let n_samples = per_tree[0].len();
        debug_assert!(per_tree.iter().all(|v| v.len() == n_samples));

        (0..n_samples)
            .map(|i| {
                let sample_dists: Vec<&PiecewiseConstantDistribution> =
                    per_tree.iter().map(|tree_dists| &tree_dists[i]).collect();
                PiecewiseConstantDistribution::ensemble(&sample_dists)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Default trait
// ---------------------------------------------------------------------------

impl Default for PartitionForestV2 {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Estimator trait implementation
// ---------------------------------------------------------------------------

impl Estimator for PartitionForestV2 {
    fn _fit_impl(
        &mut self,
        x: &DataFrame,
        y: &DataFrame,
        _sample_weights: Option<&Float64Chunked>,
    ) -> Result<Self, FitError> {
        // ── 1. Prefix target columns ──────────────────────────────────
        let mut y = y.clone();
        let column_names: Vec<String> = y
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();

        for col_name in column_names {
            if !col_name.starts_with(TARGET_PREFIX) {
                y.rename(
                    &col_name,
                    PlSmallStr::from_str(&format!("{}_{}", TARGET_PREFIX, col_name)),
                )
                .map_err(|e| FitError::InvalidInput(e.to_string()))?;
            }
        }

        // ── 2. Join X and Y horizontally ──────────────────────────────
        let xy = polars::functions::concat_df_horizontal(&[x.clone(), y], true)
            .map_err(|e| FitError::InvalidInput(e.to_string()))?;

        let schema = xy.schema().as_ref().clone();

        // ── 3. Build shared DatasetView once ──────────────────────────
        let dataset = PolarsDatasetView::new(&xy);
        let n = dataset.n_rows() as f64;

        // ── 4. Build trees in parallel ────────────────────────────────
        let base_seed = self.seed.unwrap_or(42);
        let config_template = self.build_config();
        let registry = Arc::new(DTypeRegistry::default());

        let fitted_trees: Vec<Tree> = (0..self.n_estimators)
            .into_par_iter()
            .map(|idx| {
                let config = TreeBuilderConfig {
                    max_leaves: config_template.max_leaves,
                    boundaries_expansion_factor: config_template.boundaries_expansion_factor,
                    restrictions: config_template.restrictions.clone(),
                };
                let loss: Box<dyn LossFunc> = Box::new(ConditionalLogLoss::new(n));
                let builder = TreeBuilder::new(config, loss, Arc::clone(&registry));
                let _ = base_seed + idx; // seed reserved for future stochastic splits
                builder.build(&dataset)
            })
            .collect();

        // ── 5. Store fitted state and return ──────────────────────────
        self.trees = Some(fitted_trees);
        self.schema = Some(schema.clone());

        Ok(PartitionForestV2 {
            n_estimators: self.n_estimators,
            seed: self.seed,
            max_leaves: self.max_leaves,
            boundaries_expansion_factor: self.boundaries_expansion_factor,
            min_samples_xy: self.min_samples_xy,
            min_samples_x: self.min_samples_x,
            min_samples_y: self.min_samples_y,
            min_gain: self.min_gain,
            min_volume: self.min_volume,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            trees: self.trees.take(),
            schema: Some(schema),
        })
    }

    fn _predict_impl(&self, x: &DataFrame) -> Result<DataFrame, PredictError> {
        let trees = self.fitted_trees()?;
        let xy = self.expand_with_schema(x)?;
        let dataset = PolarsDatasetView::new(&xy);

        // Collect per-tree distributions in parallel
        let per_tree: Vec<Vec<PiecewiseConstantDistribution>> = trees
            .par_iter()
            .map(|tree| tree.predict_distributions(&dataset))
            .collect();

        let ensembled = Self::merge_distributions(&per_tree);
        let mean_vectors: Vec<MeanVector> =
            ensembled.into_iter().map(|d| d.mean_vector()).collect();

        // Build output DataFrame from mean vectors
        let mut out = Self::mean_vectors_to_dataframe(&mean_vectors, &trees[0]);

        // Remove target_ prefix from column names
        let new_names: Vec<String> = out
            .get_column_names()
            .iter()
            .map(|&name| {
                let s = name.as_str();
                if s.starts_with("target_") {
                    s["target_".len()..].to_string()
                } else {
                    s.to_string()
                }
            })
            .collect();
        out.set_column_names(&new_names)
            .expect("Failed to strip target_ prefix from prediction columns");

        Ok(out)
    }
}

impl PartitionForestV2 {
    /// Convert mean vectors to a DataFrame using target schema from a tree.
    fn mean_vectors_to_dataframe(mean_vectors: &[MeanVector], tree: &Tree) -> DataFrame {
        use super::rule::DynRule;
        use crate::rules::{BelongsTo, ContinuousInterval};

        let root = tree.root();
        let mut target_cols: Vec<(&String, &dyn DynRule)> = root.cell.target_rules().collect();
        target_cols.sort_by_key(|(k, _)| (*k).clone());

        let mut columns: Vec<Column> = Vec::with_capacity(target_cols.len());

        for (col_name, rule) in &target_cols {
            if rule.as_any().downcast_ref::<ContinuousInterval>().is_some()
                || rule
                    .as_any()
                    .downcast_ref::<crate::rules::IntegerInterval>()
                    .is_some()
            {
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
            DataFrame::empty()
        } else {
            DataFrame::new(columns)
                .expect("mean_vectors_to_dataframe: DataFrame construction failed")
        }
    }
}
