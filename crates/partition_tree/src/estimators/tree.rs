//! Scikit-learn style estimator wrapper around the v2 partition tree.
//!
//! [`PartitionTreeV2`] implements the [`Estimator`](estimators::api::Estimator)
//! trait, providing a familiar `fit` / `predict` interface on top of the v2
//! [`TreeBuilder`] and [`Tree`].
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use partition_tree::v2::estimator::PartitionTreeV2;
//! use estimators::api::Estimator;
//!
//! let mut model = PartitionTreeV2::default();
//! let fitted = model.fit(&x_train, &y_train, None).unwrap();
//! let preds  = fitted.predict(&x_test).unwrap();
//! ```
//!
//! ## Prediction API
//!
//! | Method                    | Returns                                      |
//! |---------------------------|----------------------------------------------|
//! | `predict`                 | `DataFrame` (point predictions per target)   |
//! | `predict_proba`           | `Vec<PiecewiseConstantDistribution>`          |
//! | `predict_mean_vectors`    | `Vec<MeanVector>`                             |
//! | `feature_importances`     | `HashMap<String, f64>`                        |
//! | `apply`                   | `Vec<usize>` (leaf node indices)              |
use std::collections::HashMap;
use std::sync::Arc;

use estimators::api::{Estimator, FitError, PredictError};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::conf::TARGET_PREFIX;
use crate::serde::schema as schema_serde;

use crate::dataset_view::{DatasetView, PolarsDatasetView};
use crate::dtype_plugin::DTypeRegistry;
use crate::loss::{ConditionalLogLoss, LossFunc};
use crate::predict::piecewise_distribution::{MeanVector, PiecewiseConstantDistribution};
use crate::split_result::SplitRestrictions;
use crate::tree::Tree;
use crate::tree_builder::{TreeBuilder, TreeBuilderConfig};

// ---------------------------------------------------------------------------
// PartitionTreeV2
// ---------------------------------------------------------------------------

/// Estimator wrapper around the v2 partition tree.
///
/// Stores the builder configuration as public fields (mirroring v1's
/// `PartitionTree` API) plus a fitted `Tree` after calling `fit`.
#[derive(Serialize, Deserialize)]
pub struct PartitionTree {
    // ── Builder configuration ──────────────────────────────────────────
    /// Maximum number of leaves (`max_iter + 1` equivalent).
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
    /// Minimum target volume fraction `[0.0, 1.0]` each child must hold
    /// relative to its parent's target volume.
    pub min_volume_fraction: f64,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum total samples in a parent to attempt a split.
    pub min_samples_split: f64,
    /// Fraction of rows to bootstrap-sample.
    /// `None` means use all rows.
    pub max_samples: Option<f64>,
    /// Whether bootstrap sampling is done with replacement (`true`, default)
    /// or without replacement (`false`).
    pub replace: bool,
    /// Fraction of feature columns to consider at each split.
    /// `None` means use all features.
    pub max_features: Option<f64>,
    // Loss function to be used
    pub loss: Option<Box<dyn LossFunc>>,
    /// RNG seed for reproducible bootstrap / feature subsampling.
    pub seed: Option<u64>,

    // ── Fitted state ───────────────────────────────────────────────────
    /// The fitted tree (populated after `fit`).
    pub tree: Option<Tree>,
    /// Schema of the XY DataFrame used for fitting.
    #[serde(with = "schema_serde")]
    pub schema: Option<Schema>,
    /// Category label mappings from training (col_name → sorted labels).
    /// Used to ensure consistent code assignment at prediction time.
    pub cat_labels: Option<HashMap<String, Vec<String>>>,
}

impl PartitionTree {
    /// Create a new estimator with full control over all parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_leaves: usize,
        boundaries_expansion_factor: f64,
        min_samples_xy: f64,
        min_samples_x: f64,
        min_samples_y: f64,
        min_gain: f64,
        min_volume_fraction: f64,
        max_depth: usize,
        min_samples_split: f64,
        max_samples: Option<f64>,
        replace: bool,
        max_features: Option<f64>,
        loss: Option<Box<dyn LossFunc>>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            max_leaves,
            boundaries_expansion_factor,
            min_samples_xy,
            min_samples_x,
            min_samples_y,
            min_gain,
            min_volume_fraction,
            max_depth,
            min_samples_split,
            max_samples,
            replace,
            max_features,
            loss,
            seed,
            tree: None,
            schema: None,
            cat_labels: None,
        }
    }

    /// Sensible defaults matching the v1 `PartitionTree::default()`.
    pub fn with_defaults() -> Self {
        Self {
            max_leaves: 101,
            boundaries_expansion_factor: 0.1,
            min_samples_xy: 1.0,
            min_samples_x: 1.0,
            min_samples_y: 1.0,
            min_gain: 0.0,
            min_volume_fraction: 0.0,
            max_depth: usize::MAX,
            min_samples_split: 2.0,
            max_samples: None,
            replace: true,
            max_features: None,
            loss: None,
            seed: None,
            tree: None,
            schema: None,
            cat_labels: None,
        }
    }

    // ── Prediction helpers ─────────────────────────────────────────────

    /// Predict conditional distributions for every row of `x`.
    ///
    /// Returns a `Vec<PiecewiseConstantDistribution>` — one per row.
    /// The distributions can be queried for `mean_vector`, `pdf`,
    /// `pdf_segments`, `category_probabilities`, etc.
    pub fn predict_proba(
        &self,
        x: &DataFrame,
    ) -> Result<Vec<PiecewiseConstantDistribution>, PredictError> {
        let tree = self.fitted_tree()?;
        let xy = self.expand_with_schema(x)?;
        let dataset = self.build_prediction_dataset(&xy);
        Ok(tree.predict_distributions(&dataset))
    }

    /// Predict mean vectors for every row of `x`.
    pub fn predict_mean_vectors(&self, x: &DataFrame) -> Result<Vec<MeanVector>, PredictError> {
        let tree = self.fitted_tree()?;
        let xy = self.expand_with_schema(x)?;
        let dataset = self.build_prediction_dataset(&xy);
        Ok(tree.predict_mean_vectors(&dataset))
    }

    /// Feature importances (cumulative gain per column).
    ///
    /// If `normalize` is true, importances sum to 1.0.
    pub fn feature_importances(
        &self,
        normalize: bool,
    ) -> Result<HashMap<String, f64>, PredictError> {
        let tree = self.fitted_tree()?;
        Ok(tree.feature_importances(normalize))
    }

    /// Apply the tree, returning the leaf node index for every row.
    pub fn apply(&self, x: &DataFrame) -> Result<Vec<usize>, PredictError> {
        let tree = self.fitted_tree()?;
        let xy = self.expand_with_schema(x)?;
        let dataset = self.build_prediction_dataset(&xy);
        Ok(tree.apply(&dataset))
    }

    /// Get detailed leaf information.
    pub fn leaves_info(&self) -> Result<Vec<crate::tree::LeafInfo>, PredictError> {
        let tree = self.fitted_tree()?;
        Ok(tree.leaf_info())
    }

    /// Target schema: `(column_name, logical_dtype)` pairs.
    pub fn target_schema(
        &self,
    ) -> Result<Vec<(String, crate::dataset_view::LogicalDType)>, PredictError> {
        let tree = self.fitted_tree()?;
        Ok(tree.target_schema())
    }

    // ── Internal helpers ───────────────────────────────────────────────

    /// Get the fitted tree or return `PredictError::NotFitted`.
    fn fitted_tree(&self) -> Result<&Tree, PredictError> {
        self.tree.as_ref().ok_or(PredictError::NotFitted)
    }

    /// Expand X-only features into the full XY schema expected by the tree.
    ///
    /// During prediction, target columns are not available. We create
    /// placeholder null columns so that `PolarsDatasetView` can resolve all
    /// column names that appear in the tree's cell rules.
    fn expand_with_schema(&self, x: &DataFrame) -> Result<DataFrame, PredictError> {
        let schema = self.schema.as_ref().ok_or(PredictError::NotFitted)?;

        let mut cols: Vec<Column> = x.get_columns().to_vec();
        let n_rows = x.height();

        // Add target columns as typed placeholders (None-filled)
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

    /// Build a `PolarsDatasetView` for prediction, using stored category
    /// labels to ensure consistent code assignment.
    fn build_prediction_dataset(&self, xy: &DataFrame) -> PolarsDatasetView {
        match &self.cat_labels {
            Some(labels) => PolarsDatasetView::with_category_labels(xy, labels),
            None => PolarsDatasetView::new(xy),
        }
    }

    /// Build the `TreeBuilderConfig` from our fields.
    fn build_config(&self) -> TreeBuilderConfig {
        TreeBuilderConfig {
            max_leaves: self.max_leaves,
            boundaries_expansion_factor: self.boundaries_expansion_factor,
            restrictions: SplitRestrictions {
                min_samples_xy: self.min_samples_xy,
                min_samples_x: self.min_samples_x,
                min_samples_y: self.min_samples_y,
                min_gain: self.min_gain,
                min_volume_fraction: self.min_volume_fraction,
                max_depth: self.max_depth,
                min_samples_split: self.min_samples_split,
            },
            max_samples: self.max_samples,
            replace: self.replace,
            max_features: self.max_features,
            seed: self.seed,
        }
    }
}

// ---------------------------------------------------------------------------
// Estimator trait implementation
// ---------------------------------------------------------------------------

impl Estimator for PartitionTree {
    fn _fit_impl(
        &mut self,
        x: &DataFrame,
        y: &DataFrame,
        _sample_weights: Option<&Float64Chunked>,
    ) -> Result<Self, FitError> {
        // ── 1. Prefix target columns with TARGET_PREFIX if needed ──────
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

        // ── 3. Build tree ─────────────────────────────────────────────
        let dataset = PolarsDatasetView::new(&xy);

        // Extract category label mappings before the tree consumes the dataset.
        let cat_labels = dataset.category_labels();

        let config = self.build_config();
        // Clone the loss non-destructively so `self.loss` is preserved for
        // future `fit` calls and so we can return it in the fitted struct.
        let loss: Box<dyn LossFunc> = self
            .loss
            .as_deref()
            .map(|l| l.clone_box())
            .unwrap_or_else(|| Box::new(ConditionalLogLoss));
        let registry = Arc::new(DTypeRegistry::default());

        let builder = TreeBuilder::new(config, loss.clone_box(), registry);
        let tree = builder.build(&dataset);

        // ── 4. Store fitted state and return ──────────────────────────
        self.tree = Some(tree);
        self.schema = Some(schema.clone());
        self.cat_labels = Some(cat_labels.clone());

        Ok(PartitionTree {
            max_leaves: self.max_leaves,
            boundaries_expansion_factor: self.boundaries_expansion_factor,
            min_samples_xy: self.min_samples_xy,
            min_samples_x: self.min_samples_x,
            min_samples_y: self.min_samples_y,
            min_gain: self.min_gain,
            min_volume_fraction: self.min_volume_fraction,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_samples: self.max_samples,
            replace: self.replace,
            max_features: self.max_features,
            loss: Some(loss),
            seed: self.seed,
            tree: self.tree.take(),
            schema: Some(schema),
            cat_labels: Some(cat_labels),
        })
    }

    fn _predict_impl(&self, x: &DataFrame) -> Result<DataFrame, PredictError> {
        let tree = self.fitted_tree()?;
        let xy = self.expand_with_schema(x)?;
        let dataset = self.build_prediction_dataset(&xy);

        let mut out = tree.predict_mean(&dataset);

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use estimators::api::Estimator;

    fn make_xy() -> (DataFrame, DataFrame) {
        let x1: Vec<Option<f64>> = (0..100)
            .map(|i| Some(if i % 4 < 2 { 1.0 } else { 2.0 }))
            .collect();
        let x2: Vec<Option<f64>> = (0..100)
            .map(|i| Some(if i % 3 == 0 { 1.0 } else { 2.0 }))
            .collect();
        let y: Vec<Option<f64>> = (0..100)
            .map(|i| Some(if i % 4 < 2 { 2.0 } else { 4.0 }))
            .collect();

        let x = DataFrame::new(vec![
            Column::new(PlSmallStr::from_static("x1"), x1),
            Column::new(PlSmallStr::from_static("x2"), x2),
        ])
        .unwrap();

        let y_df = DataFrame::new(vec![Column::new(PlSmallStr::from_static("y"), y)]).unwrap();

        (x, y_df)
    }

    #[test]
    fn fit_and_predict_roundtrip() {
        let (x, y) = make_xy();
        let mut model = PartitionTree::new(
            13, 0.0, 0.0, 0.0, 0.0, 1e-8, 0.0, 6, 2.0, None, true, None, None, None,
        );
        let fitted = model.fit(&x, &y, None).expect("fit should succeed");
        let preds = fitted.predict(&x).expect("predict should succeed");

        assert_eq!(preds.height(), x.height());
        assert!(
            preds.column("y").is_ok(),
            "prediction should have column 'y'"
        );
    }

    #[test]
    fn predict_proba_returns_distributions() {
        let (x, y) = make_xy();
        let mut model = PartitionTree::new(
            13, 0.0, 0.0, 0.0, 0.0, 1e-8, 0.0, 6, 2.0, None, true, None, None, None,
        );
        let fitted = model.fit(&x, &y, None).unwrap();
        let dists = fitted
            .predict_proba(&x)
            .expect("predict_proba should succeed");

        assert_eq!(dists.len(), x.height());
        // At least some distributions should have positive mass
        let has_positive = dists.iter().any(|d| d.total_mass() > 0.0);
        assert!(
            has_positive,
            "at least one distribution should have positive mass"
        );
    }

    #[test]
    fn feature_importances_are_nonempty() {
        let (x, y) = make_xy();
        let mut model = PartitionTree::new(
            13, 0.0, 0.0, 0.0, 0.0, 1e-8, 0.0, 6, 2.0, None, true, None, None, None,
        );
        let fitted = model.fit(&x, &y, None).unwrap();
        let imp = fitted.feature_importances(true).unwrap();

        // There should be at least one feature with positive importance
        assert!(!imp.is_empty());
        let total: f64 = imp.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "normalized importances should sum to 1.0"
        );
    }

    #[test]
    fn apply_returns_leaf_indices() {
        let (x, y) = make_xy();
        let mut model = PartitionTree::new(
            13, 0.0, 0.0, 0.0, 0.0, 1e-8, 0.0, 6, 2.0, None, true, None, None, None,
        );
        let fitted = model.fit(&x, &y, None).unwrap();
        let leaf_indices = fitted.apply(&x).unwrap();

        assert_eq!(leaf_indices.len(), x.height());
        let tree = fitted.tree.as_ref().unwrap();
        for &idx in &leaf_indices {
            assert!(
                tree.nodes[idx].is_leaf,
                "apply should return leaf node indices"
            );
        }
    }

    #[test]
    fn leaves_info_matches_tree() {
        let (x, y) = make_xy();
        let mut model = PartitionTree::new(
            13, 0.0, 0.0, 0.0, 0.0, 1e-8, 0.0, 6, 2.0, None, true, None, None, None,
        );
        let fitted = model.fit(&x, &y, None).unwrap();
        let infos = fitted.leaves_info().unwrap();

        let tree = fitted.tree.as_ref().unwrap();
        assert_eq!(infos.len(), tree.n_leaves());
    }

    #[test]
    fn not_fitted_returns_error() {
        let model = PartitionTree::with_defaults();
        let x = DataFrame::new(vec![Column::new(
            PlSmallStr::from_static("x1"),
            vec![1.0f64],
        )])
        .unwrap();

        assert!(matches!(model.predict(&x), Err(PredictError::NotFitted)));
        assert!(matches!(
            model.predict_proba(&x),
            Err(PredictError::NotFitted)
        ));
        assert!(matches!(
            model.feature_importances(true),
            Err(PredictError::NotFitted)
        ));
        assert!(matches!(model.apply(&x), Err(PredictError::NotFitted)));
    }

    #[test]
    fn predictions_match_actual_values() {
        // y = 2*x1, so for x1=1 → y=2, x1=2 → y=4
        let (x, y) = make_xy();
        let mut model = PartitionTree::new(
            13, 0.0, 0.0, 0.0, 0.0, 1e-8, 0.0, 6, 2.0, None, true, None, None, None,
        );
        let fitted = model.fit(&x, &y, None).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let pred_col = preds.column("y").unwrap().f64().unwrap();
        let y_col = y.column("y").unwrap().f64().unwrap();

        // Predictions should be in a plausible range (midpoint of leaf interval)
        for i in 0..x.height() {
            let p = pred_col.get(i).unwrap();
            let a = y_col.get(i).unwrap();
            assert!(
                (p - a).abs() <= 2.0,
                "row {i}: predicted {p} vs actual {a} differs by more than 2.0"
            );
        }
    }

    #[test]
    fn serde_roundtrip_bincode() {
        let (x, y) = make_xy();
        let mut model = PartitionTree::new(
            13, 0.0, 0.0, 0.0, 0.0, 1e-8, 0.0, 6, 2.0, None, true, None, None, None,
        );
        let fitted = model.fit(&x, &y, None).unwrap();

        // ── Serialize ──
        let bytes = bincode::serialize(&fitted).expect("serialize should succeed");
        assert!(!bytes.is_empty());

        // ── Deserialize ──
        let restored: PartitionTree =
            bincode::deserialize(&bytes).expect("deserialize should succeed");

        // ── Config fields should match ──
        assert_eq!(restored.max_leaves, fitted.max_leaves);
        assert_eq!(restored.max_depth, fitted.max_depth);
        assert!((restored.min_gain - fitted.min_gain).abs() < 1e-15);

        // ── Tree structure should match ──
        let t1 = fitted.tree.as_ref().unwrap();
        let t2 = restored.tree.as_ref().unwrap();
        assert_eq!(t1.n_leaves(), t2.n_leaves());
        assert_eq!(t1.nodes.len(), t2.nodes.len());
        assert_eq!(t1.split_history.len(), t2.split_history.len());

        // ── Predictions should be identical ──
        let preds_orig = fitted.predict(&x).unwrap();
        let preds_rest = restored.predict(&x).unwrap();
        let col_orig = preds_orig.column("y").unwrap().f64().unwrap();
        let col_rest = preds_rest.column("y").unwrap().f64().unwrap();

        for i in 0..x.height() {
            let a = col_orig.get(i).unwrap();
            let b = col_rest.get(i).unwrap();
            assert!(
                (a - b).abs() < 1e-12,
                "row {i}: original {a} vs restored {b} differ after serde roundtrip"
            );
        }
    }
}
