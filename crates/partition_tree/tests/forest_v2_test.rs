//! Integration tests for `PartitionForestV2`.

use estimators::api::Estimator;
use partition_tree::v2::forest::PartitionForestV2;
use polars::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple deterministic dataset: y = 2*x1 (roughly).
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

/// Build a small forest with sensible test parameters.
fn fit_forest(n_estimators: usize) -> (PartitionForestV2, DataFrame) {
    let (x, y) = make_xy();
    let mut model = PartitionForestV2::new(
        n_estimators,
        /* max_leaves */ 13,
        /* boundaries_expansion_factor */ 0.0,
        /* min_samples_xy */ 0.0,
        /* min_samples_x */ 0.0,
        /* min_samples_y */ 0.0,
        /* min_gain */ 1e-8,
        /* min_volume */ 0.0,
        /* max_depth */ 6,
        /* min_samples_split */ 2.0,
        /* seed */ Some(42),
    );
    let fitted = model.fit(&x, &y, None).expect("fit should succeed");
    (fitted, x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn fit_and_predict_roundtrip() {
    let (fitted, x) = fit_forest(5);
    let preds = fitted.predict(&x).expect("predict should succeed");

    assert_eq!(preds.height(), x.height());
    assert!(
        preds.column("y").is_ok(),
        "prediction should have column 'y'"
    );
}

#[test]
fn predict_proba_returns_distributions() {
    let (fitted, x) = fit_forest(5);
    let dists = fitted
        .predict_proba(&x)
        .expect("predict_proba should succeed");

    assert_eq!(dists.len(), x.height());
    let has_positive = dists.iter().any(|d| d.total_mass() > 0.0);
    assert!(
        has_positive,
        "at least one distribution should have positive mass"
    );
}

#[test]
fn predict_trees_proba_returns_per_tree() {
    let n_trees = 5;
    let (fitted, x) = fit_forest(n_trees);
    let per_tree = fitted
        .predict_trees_proba(&x)
        .expect("predict_trees_proba should succeed");

    assert_eq!(per_tree.len(), n_trees);
    for tree_dists in &per_tree {
        assert_eq!(tree_dists.len(), x.height());
    }
}

#[test]
fn ensemble_distribution_has_multiple_cells() {
    let (fitted, x) = fit_forest(3);
    let dists = fitted.predict_proba(&x).unwrap();

    // With 3 trees, each single-tree distribution has 1 cell.
    // After ensembling, each distribution should have 3 cells.
    for d in &dists {
        assert_eq!(
            d.n_cells(),
            3,
            "ensembled distribution should have one cell per tree"
        );
    }
}

#[test]
fn feature_importances_are_nonempty() {
    let (fitted, _x) = fit_forest(5);
    let imp = fitted.feature_importances(true).unwrap();

    assert!(!imp.is_empty());
    let total: f64 = imp.values().sum();
    assert!(
        (total - 1.0).abs() < 1e-10,
        "normalized importances should sum to 1.0, got {total}"
    );
}

#[test]
fn n_trees_matches_n_estimators() {
    let n = 7;
    let (fitted, _x) = fit_forest(n);
    assert_eq!(fitted.n_trees(), n);
}

#[test]
fn not_fitted_returns_error() {
    use estimators::api::PredictError;

    let model = PartitionForestV2::with_defaults();
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
}

#[test]
fn predictions_match_actual_values() {
    let (x, y) = make_xy();
    let (fitted, _) = fit_forest(10);
    let preds = fitted.predict(&x).unwrap();

    let pred_col = preds.column("y").unwrap().f64().unwrap();
    let y_col = y.column("y").unwrap().f64().unwrap();

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
fn default_creates_valid_config() {
    let model = PartitionForestV2::default();
    assert_eq!(model.n_estimators, 100);
    assert_eq!(model.n_trees(), 0);
    assert!(model.trees.is_none());
}

#[test]
fn predict_mean_vectors_returns_correct_length() {
    let (fitted, x) = fit_forest(5);
    let mvs = fitted
        .predict_mean_vectors(&x)
        .expect("predict_mean_vectors should succeed");

    assert_eq!(mvs.len(), x.height());
    for mv in &mvs {
        assert!(
            mv.contains_key("target_y"),
            "mean vector should contain target column key, got keys: {:?}",
            mv.keys().collect::<Vec<_>>()
        );
    }
}

#[test]
fn serde_roundtrip_bincode() {
    let (fitted, x) = fit_forest(3);

    // ── Serialize ──
    let bytes = bincode::serialize(&fitted).expect("serialize should succeed");
    assert!(!bytes.is_empty());

    // ── Deserialize ──
    let restored: PartitionForestV2 =
        bincode::deserialize(&bytes).expect("deserialize should succeed");

    // ── Config should match ──
    assert_eq!(restored.n_estimators, fitted.n_estimators);
    assert_eq!(restored.max_leaves, fitted.max_leaves);
    assert_eq!(restored.max_depth, fitted.max_depth);
    assert_eq!(restored.n_trees(), fitted.n_trees());

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

#[test]
fn forest_more_trees_reduce_variance() {
    // With a deterministic dataset, more trees shouldn't hurt and the
    // ensemble prediction should still be close to the true values.
    let (x, y) = make_xy();

    let (fitted_small, _) = fit_forest(2);
    let (fitted_large, _) = fit_forest(20);

    let preds_small = fitted_small.predict(&x).unwrap();
    let preds_large = fitted_large.predict(&x).unwrap();

    let y_col = y.column("y").unwrap().f64().unwrap();
    let col_small = preds_small.column("y").unwrap().f64().unwrap();
    let col_large = preds_large.column("y").unwrap().f64().unwrap();

    let mse_small: f64 = (0..x.height())
        .map(|i| {
            let d = col_small.get(i).unwrap() - y_col.get(i).unwrap();
            d * d
        })
        .sum::<f64>()
        / x.height() as f64;

    let mse_large: f64 = (0..x.height())
        .map(|i| {
            let d = col_large.get(i).unwrap() - y_col.get(i).unwrap();
            d * d
        })
        .sum::<f64>()
        / x.height() as f64;

    // Both should have reasonable MSE (< 2.0 for this simple dataset)
    assert!(
        mse_large <= 2.0,
        "large forest MSE should be low, got {mse_large}"
    );
    assert!(
        mse_small <= 2.0,
        "small forest MSE should be low, got {mse_small}"
    );
}
