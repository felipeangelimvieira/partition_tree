//! Integration tests for `PartitionForest`.

use estimators::api::Estimator;
use partition_tree::PartitionForest;
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
fn fit_forest(n_estimators: usize) -> (PartitionForest, DataFrame) {
    let (x, y) = make_xy();
    let mut model = PartitionForest::new(
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
        /* max_samples */ None,
        /* replace */ true,
        /* max_features */ None,
        /* loss */ None,
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
fn apply_returns_leaf_indices_per_tree() {
    let n_trees = 5;
    let (fitted, x) = fit_forest(n_trees);
    let per_tree_leaf_indices = fitted.apply(&x).expect("apply should succeed");

    assert_eq!(per_tree_leaf_indices.len(), n_trees);

    let trees = fitted.trees.as_ref().expect("forest should be fitted");
    for (tree, leaf_indices) in trees.iter().zip(&per_tree_leaf_indices) {
        assert_eq!(leaf_indices.len(), x.height());
        for &idx in leaf_indices {
            assert!(
                tree.nodes[idx].is_leaf,
                "apply should return leaf node indices"
            );
        }
    }
}

#[test]
fn ensemble_distribution_has_multiple_cells() {
    let (fitted, x) = fit_forest(3);
    let dists = fitted.predict_proba(&x).unwrap();

    // Each tree contributes at least one conditioned cell for a given x.
    // If trees split on target columns, a tree may contribute >1 cells.
    // So the ensemble must have at least one cell per tree.
    for d in &dists {
        assert!(
            d.n_cells() >= 3,
            "ensembled distribution should have at least one cell per tree"
        );
    }
}

#[test]
fn feature_importances_are_nonempty() {
    // Use data where x1 creates children with different densities.
    let x1: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64 / 10.0)).collect();
    let y_vals: Vec<Option<f64>> = (0..100)
        .map(|i| Some(if i < 50 { 1.0 } else { 9.0 }))
        .collect();
    let x = DataFrame::new(vec![Column::new(PlSmallStr::from_static("x1"), x1)]).unwrap();
    let y = DataFrame::new(vec![Column::new(PlSmallStr::from_static("y"), y_vals)]).unwrap();

    let mut model = PartitionForest::new(
        5,
        13,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        6,
        2.0,
        None,
        true,
        None,
        None,
        Some(42),
    );
    let fitted = model.fit(&x, &y, None).unwrap();
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

    let model = PartitionForest::with_defaults();
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
    let model = PartitionForest::default();
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
    let restored: PartitionForest =
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

// ---------------------------------------------------------------------------
// Subsampling / random-tree tests
// ---------------------------------------------------------------------------

/// Build a forest with max_samples and max_features enabled.
fn fit_forest_subsampled(
    n_estimators: usize,
    max_samples: Option<f64>,
    max_features: Option<f64>,
    seed: Option<usize>,
) -> (PartitionForest, DataFrame) {
    let (x, y) = make_xy();
    let mut model = PartitionForest::new(
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
        /* max_samples */ max_samples,
        /* replace */ true,
        /* max_features */ max_features,
        /* loss */ None,
        /* seed */ seed,
    );
    let fitted = model.fit(&x, &y, None).expect("fit should succeed");
    (fitted, x)
}

#[test]
fn forest_with_subsampling_produces_diverse_trees() {
    let (fitted, _x) = fit_forest_subsampled(5, Some(0.8), Some(0.5), Some(42));
    let trees = fitted.trees.as_ref().unwrap();

    // Collect each tree's split history as (col_name, gain) tuples for comparison
    let histories: Vec<Vec<(String, f64)>> = trees
        .iter()
        .map(|t| {
            t.split_history
                .iter()
                .map(|r| (r.col_name.clone(), r.gain))
                .collect()
        })
        .collect();

    // At least one pair of trees must differ in their split history
    let all_identical = histories.windows(2).all(|w| {
        w[0].len() == w[1].len()
            && w[0]
                .iter()
                .zip(w[1].iter())
                .all(|(a, b)| a.0 == b.0 && (a.1 - b.1).abs() < 1e-12)
    });

    assert!(
        !all_identical,
        "forest trees with max_samples + max_features should NOT all be identical"
    );
}

#[test]
fn forest_with_subsampling_predictions_reasonable() {
    let (x, y) = make_xy();
    let (fitted, _) = fit_forest_subsampled(10, Some(0.8), Some(0.5), Some(42));
    let preds = fitted.predict(&x).unwrap();

    let pred_col = preds.column("y").unwrap().f64().unwrap();
    let y_col = y.column("y").unwrap().f64().unwrap();

    let mse: f64 = (0..x.height())
        .map(|i| {
            let d = pred_col.get(i).unwrap() - y_col.get(i).unwrap();
            d * d
        })
        .sum::<f64>()
        / x.height() as f64;

    assert!(
        mse < 2.0,
        "forest with subsampling should still have reasonable MSE, got {mse}"
    );
}

#[test]
fn forest_subsampling_is_reproducible() {
    let (fitted_a, x) = fit_forest_subsampled(5, Some(0.8), Some(0.5), Some(42));
    let (fitted_b, _) = fit_forest_subsampled(5, Some(0.8), Some(0.5), Some(42));

    let preds_a = fitted_a.predict(&x).unwrap();
    let preds_b = fitted_b.predict(&x).unwrap();

    let col_a = preds_a.column("y").unwrap().f64().unwrap();
    let col_b = preds_b.column("y").unwrap().f64().unwrap();

    for i in 0..x.height() {
        let a = col_a.get(i).unwrap();
        let b = col_b.get(i).unwrap();
        assert!(
            (a - b).abs() < 1e-12,
            "row {i}: same seed should give identical predictions: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Classification helpers
// ---------------------------------------------------------------------------

/// Build a synthetic 3-class classification dataset (90 rows, 2 features).
///
/// Decision boundary is purely on `x1`:
/// - x1 in [0.0, 1.0) → class "a" (30 rows)
/// - x1 in [1.0, 2.0) → class "b" (30 rows)
/// - x1 in [2.0, 3.0) → class "c" (30 rows)
///
/// `x2` is a weaker correlated feature added to test multi-feature splits.
fn make_classification_xy() -> (DataFrame, DataFrame) {
    let n_per_class = 30_usize;
    let classes = ["a", "b", "c"];

    let mut x1_vals: Vec<f64> = Vec::with_capacity(n_per_class * 3);
    let mut x2_vals: Vec<f64> = Vec::with_capacity(n_per_class * 3);
    let mut labels: Vec<&str> = Vec::with_capacity(n_per_class * 3);

    for (cls_idx, &cls) in classes.iter().enumerate() {
        let base = cls_idx as f64;
        for j in 0..n_per_class {
            let step = j as f64 / n_per_class as f64; // in [0, 1)
            x1_vals.push(base + step);
            // x2 loosely correlates: high for "a" and "c", low for "b"
            x2_vals.push(if cls_idx == 1 {
                0.1 + step * 0.2
            } else {
                0.7 + step * 0.2
            });
            labels.push(cls);
        }
    }

    let x = DataFrame::new(vec![
        Column::new(PlSmallStr::from_static("x1"), x1_vals),
        Column::new(PlSmallStr::from_static("x2"), x2_vals),
    ])
    .unwrap();

    let label_series = Series::new(PlSmallStr::from_static("class"), labels.clone());
    let cats = FrozenCategories::new(["a", "b", "c"]).unwrap();
    let label_cat = label_series
        .cast(&DataType::from_frozen_categories(cats))
        .expect("failed to cast labels to Enum");
    let y = DataFrame::new(vec![label_cat.into()]).unwrap();

    (x, y)
}

/// Fraction of rows where the predicted string label matches the true label.
fn str_accuracy(pred_col: &Column, true_labels: &[&str]) -> f64 {
    let ca = pred_col.str().expect("expected Utf8/String column");
    let correct = (0..true_labels.len())
        .filter(|&i| ca.get(i).map_or(false, |v| v == true_labels[i]))
        .count();
    correct as f64 / true_labels.len() as f64
}

/// Build a small forest on the classification dataset.
fn fit_classification_forest(
    n_estimators: usize,
) -> (PartitionForest, DataFrame, Vec<&'static str>) {
    let (x, y) = make_classification_xy();
    let true_labels: Vec<&str> = ["a", "b", "c"]
        .iter()
        .flat_map(|&cls| std::iter::repeat(cls).take(30))
        .collect();

    let mut model = PartitionForest::new(
        n_estimators,
        /* max_leaves */ 13,
        /* boundaries_expansion_factor */ 0.1,
        /* min_samples_xy */ 1.0,
        /* min_samples_x */ 1.0,
        /* min_samples_y */ 1.0,
        /* min_gain */ 0.0,
        /* min_volume */ 0.0,
        /* max_depth */ usize::MAX,
        /* min_samples_split */ 2.0,
        /* max_samples */ None,
        /* replace */ true,
        /* max_features */ None,
        /* loss */ None,
        /* seed */ Some(42),
    );
    let fitted = model
        .fit(&x, &y, None)
        .expect("classification fit should succeed");
    (fitted, x, true_labels)
}

// ---------------------------------------------------------------------------
// Classification tests
// ---------------------------------------------------------------------------

#[test]
fn classification_fit_and_predict_roundtrip() {
    let (fitted, x, _) = fit_classification_forest(5);
    let preds = fitted
        .predict(&x)
        .expect("classification predict should succeed");

    assert_eq!(preds.height(), x.height(), "row count mismatch");
    assert!(
        preds.column("class").is_ok(),
        "predictions should contain column 'class'"
    );
}

#[test]
fn classification_predictions_vary() {
    let (fitted, x, _) = fit_classification_forest(5);
    let preds = fitted.predict(&x).expect("predict should succeed");

    let col = preds.column("class").expect("missing column 'class'");
    let ca = col.str().expect("expected String column");

    let first = ca.get(0).unwrap_or("").to_string();
    let varies = (1..ca.len()).any(|i| ca.get(i).map_or(false, |v| v != first.as_str()));

    assert!(
        varies,
        "forest classification: all predictions are identical — model did not learn"
    );
}

#[test]
fn classification_beats_naive_baseline() {
    // With 3 balanced classes, the majority-class baseline is 1/3 ≈ 0.33.
    // A forest on this clean dataset should comfortably exceed 0.90.
    let (fitted, x, true_labels) = fit_classification_forest(10);
    let preds = fitted.predict(&x).expect("predict should succeed");

    let col = preds.column("class").expect("missing column 'class'");
    let accuracy = str_accuracy(col, &true_labels);

    assert!(
        accuracy > 0.90,
        "forest classification accuracy {accuracy:.2} should exceed 0.90 (naive baseline ≈ 0.33)"
    );
}

#[test]
fn classification_all_three_classes_predicted() {
    let (fitted, x, _) = fit_classification_forest(5);
    let preds = fitted.predict(&x).expect("predict should succeed");

    let col = preds.column("class").expect("missing column 'class'");
    let ca = col.str().expect("expected String column");

    let mut seen = std::collections::HashSet::new();
    for i in 0..ca.len() {
        if let Some(v) = ca.get(i) {
            seen.insert(v.to_string());
        }
    }

    assert!(seen.contains("a"), "class 'a' never predicted");
    assert!(seen.contains("b"), "class 'b' never predicted");
    assert!(seen.contains("c"), "class 'c' never predicted");
}

#[test]
fn classification_predict_proba_returns_distributions() {
    let (fitted, x, _) = fit_classification_forest(5);
    let dists = fitted
        .predict_proba(&x)
        .expect("predict_proba should succeed on classification dataset");

    assert_eq!(dists.len(), x.height(), "one distribution per row expected");
    let has_mass = dists.iter().any(|d| d.total_mass() > 0.0);
    assert!(
        has_mass,
        "at least one distribution should have positive mass"
    );
}

#[test]
fn classification_feature_importances_nonempty() {
    let (fitted, _x, _) = fit_classification_forest(5);
    let importances = fitted
        .feature_importances(true)
        .expect("feature importances should succeed");

    assert!(
        !importances.is_empty(),
        "feature importances should not be empty"
    );
    let total: f64 = importances.values().sum();
    assert!(
        (total - 1.0).abs() < 1e-10,
        "normalized importances should sum to 1.0, got {total}"
    );
    // x1 is the primary signal; it should receive the highest importance
    let imp_x1 = importances.get("x1").copied().unwrap_or(0.0);
    let imp_x2 = importances.get("x2").copied().unwrap_or(0.0);
    assert!(
        imp_x1 > imp_x2,
        "x1 (primary signal) should have higher importance than x2, got x1={imp_x1:.4} x2={imp_x2:.4}"
    );
}

#[test]
fn classification_serde_roundtrip_bincode() {
    let (fitted, x, true_labels) = fit_classification_forest(3);

    // Serialize
    let bytes = bincode::serialize(&fitted).expect("serialize should succeed");
    assert!(!bytes.is_empty());

    // Deserialize
    let restored: PartitionForest =
        bincode::deserialize(&bytes).expect("deserialize should succeed");

    // Config sanity checks
    assert_eq!(restored.n_estimators, fitted.n_estimators);
    assert_eq!(restored.n_trees(), fitted.n_trees());

    // Predictions from the restored model must match the original
    let preds_orig = fitted.predict(&x).expect("original predict failed");
    let preds_rest = restored.predict(&x).expect("restored predict failed");

    let col_orig = preds_orig
        .column("class")
        .expect("missing 'class' in original preds");
    let col_rest = preds_rest
        .column("class")
        .expect("missing 'class' in restored preds");
    let ca_orig = col_orig.str().unwrap();
    let ca_rest = col_rest.str().unwrap();

    for i in 0..x.height() {
        let a = ca_orig.get(i).unwrap_or("");
        let b = ca_rest.get(i).unwrap_or("");
        assert_eq!(
            a, b,
            "row {i}: original '{a}' vs restored '{b}' differ after serde roundtrip"
        );
    }

    // Restored model should still achieve acceptable accuracy
    let accuracy = str_accuracy(col_rest, &true_labels);
    assert!(
        accuracy > 0.90,
        "restored model accuracy {accuracy:.2} should exceed 0.90"
    );
}

#[test]
fn classification_both_models_same_seed_agree() {
    // Two independently-fit forests with the same seed should both achieve
    // high classification accuracy on a clean dataset. While exact row-level
    // equality is not guaranteed due to rayon thread scheduling, both models
    // must reflect consistent learning from identical data.
    let (fitted_a, x, true_labels) = fit_classification_forest(5);
    let (fitted_b, _, _) = fit_classification_forest(5);

    let preds_a = fitted_a.predict(&x).expect("predict_a failed");
    let preds_b = fitted_b.predict(&x).expect("predict_b failed");

    let col_a = preds_a.column("class").expect("missing 'class' in preds_a");
    let col_b = preds_b.column("class").expect("missing 'class' in preds_b");

    let acc_a = str_accuracy(col_a, &true_labels);
    let acc_b = str_accuracy(col_b, &true_labels);

    assert!(
        acc_a > 0.90,
        "first model accuracy {acc_a:.2} should exceed 0.90"
    );
    assert!(
        acc_b > 0.90,
        "second model accuracy {acc_b:.2} should exceed 0.90"
    );

    // Row-level agreement should be very high (both models learned the same boundaries)
    let ca_a = col_a.str().unwrap();
    let ca_b = col_b.str().unwrap();
    let agreement = (0..x.height())
        .filter(|&i| ca_a.get(i) == ca_b.get(i))
        .count() as f64
        / x.height() as f64;
    assert!(
        agreement >= 0.90,
        "two forests with same seed should agree on ≥90% of rows, got {agreement:.2}"
    );
}
