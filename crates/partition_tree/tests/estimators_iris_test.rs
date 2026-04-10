//! Integration tests for [`PartitionTree`] and [`PartitionForest`] on the
//! Iris dataset.
//!
//! Two problem setups are exercised:
//!
//! - **Regression** – predict `petal_length` from `sepal_length`,
//!   `sepal_width`, and `petal_width` (arbitrary fixed dimension as target).
//! - **Classification** – predict `species` (categorical) from all four
//!   numeric features.
//!
//! Each test checks that predictions *vary* across the 150 samples, which
//! confirms the model learned meaningful structure rather than predicting a
//! constant for every row.

use estimators::api::Estimator;
use partition_tree::estimators::{PartitionForest, PartitionTree};
use polars::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Iris data loading
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct IrisRow {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

fn load_iris() -> Vec<IrisRow> {
    let url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv";
    let body = reqwest::blocking::get(url)
        .expect("failed to fetch iris dataset — check network access")
        .text()
        .expect("failed to read iris response body");

    let mut rdr = csv::Reader::from_reader(body.as_bytes());
    rdr.deserialize::<IrisRow>()
        .map(|r| r.expect("failed to parse iris row"))
        .collect()
}

// ---------------------------------------------------------------------------
// DataFrame builders
// ---------------------------------------------------------------------------

/// Regression: predict `petal_length` from the other three numeric features.
fn regression_frames(rows: &[IrisRow]) -> (DataFrame, DataFrame) {
    let sl: Vec<f64> = rows.iter().map(|r| r.sepal_length).collect();
    let sw: Vec<f64> = rows.iter().map(|r| r.sepal_width).collect();
    let pw: Vec<f64> = rows.iter().map(|r| r.petal_width).collect();
    let pl: Vec<f64> = rows.iter().map(|r| r.petal_length).collect();

    let x = DataFrame::new(vec![
        Column::new("sepal_length".into(), sl),
        Column::new("sepal_width".into(), sw),
        Column::new("petal_width".into(), pw),
    ])
    .unwrap();

    let y = DataFrame::new(vec![Column::new("petal_length".into(), pl)]).unwrap();

    (x, y)
}

/// Classification: predict `species` from all four numeric features.
///
/// Species are encoded as a fixed-category `Enum` Polars dtype so the
/// partition tree's categorical plugin handles them correctly.
fn classification_frames(rows: &[IrisRow]) -> (DataFrame, DataFrame) {
    let sl: Vec<f64> = rows.iter().map(|r| r.sepal_length).collect();
    let sw: Vec<f64> = rows.iter().map(|r| r.sepal_width).collect();
    let pl: Vec<f64> = rows.iter().map(|r| r.petal_length).collect();
    let pw: Vec<f64> = rows.iter().map(|r| r.petal_width).collect();
    let sp: Vec<&str> = rows.iter().map(|r| r.species.as_str()).collect();

    let x = DataFrame::new(vec![
        Column::new("sepal_length".into(), sl),
        Column::new("sepal_width".into(), sw),
        Column::new("petal_length".into(), pl),
        Column::new("petal_width".into(), pw),
    ])
    .unwrap();

    // Cast the string series to an Enum (fixed-category) type so that
    // `PolarsColumnView` recognises it as `LogicalDType::Categorical`.
    let sp_utf8 = Series::new("species".into(), sp);
    let cats = FrozenCategories::new(["setosa", "versicolor", "virginica"]).unwrap();
    let sp_cat: Series = sp_utf8
        .cast(&DataType::from_frozen_categories(cats))
        .expect("failed to cast species to Enum");

    let y = DataFrame::new(vec![sp_cat.into()]).unwrap();

    (x, y)
}

// ---------------------------------------------------------------------------
// Variation helpers
// ---------------------------------------------------------------------------

/// Returns `true` when a `Float64` column contains at least two distinct values.
fn f64_varies(col: &Column) -> bool {
    let ca = col.f64().expect("expected Float64 column");
    let first = ca.get(0).unwrap_or(f64::NAN);
    (1..ca.len()).any(|i| ca.get(i).map_or(false, |v| (v - first).abs() > 1e-10))
}

/// Returns `true` when a `String` column contains at least two distinct values.
fn str_varies(col: &Column) -> bool {
    let ca = col.str().expect("expected String column");
    let first = ca.get(0).unwrap_or("").to_string();
    (1..ca.len()).any(|i| ca.get(i).map_or(false, |v| v != first.as_str()))
}

// ---------------------------------------------------------------------------
// Error metric helpers
// ---------------------------------------------------------------------------

/// Mean squared error between a predicted `Float64` column and ground-truth values.
fn regression_mse(pred_col: &Column, true_vals: &[f64]) -> f64 {
    let ca = pred_col.f64().expect("expected Float64 column");
    let n = true_vals.len() as f64;
    (0..true_vals.len())
        .map(|i| {
            let p = ca.get(i).unwrap_or(f64::NAN);
            (p - true_vals[i]).powi(2)
        })
        .sum::<f64>()
        / n
}

/// Naive baseline MSE: always predict the training mean.
fn naive_regression_mse(true_vals: &[f64]) -> f64 {
    let n = true_vals.len() as f64;
    let mean = true_vals.iter().sum::<f64>() / n;
    true_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
}

/// Map iris species strings to the labels produced by `CategoricalPlugin`.
///
/// Since the plugin now uses real string labels extracted from the
/// categorical column, the predicted output will contain the actual
/// species name.
fn species_to_cat_label(species: &str) -> &'static str {
    match species {
        "setosa" => "setosa",
        "versicolor" => "versicolor",
        "virginica" => "virginica",
        _ => panic!("unknown species: {species}"),
    }
}

/// Fraction of predictions matching the supplied true labels.
fn classification_accuracy(pred_col: &Column, true_labels: &[&str]) -> f64 {
    let ca = pred_col.str().expect("expected String column");
    let correct = (0..true_labels.len())
        .filter(|&i| ca.get(i).map_or(false, |v| v == true_labels[i]))
        .count();
    correct as f64 / true_labels.len() as f64
}

// ---------------------------------------------------------------------------
// PartitionTree tests
// ---------------------------------------------------------------------------

mod tree_tests {
    use super::*;

    #[test]
    fn regression_predictions_vary() {
        let rows = load_iris();
        let (x, y) = regression_frames(&rows);

        let mut model = PartitionTree::with_defaults();
        let fitted = model.fit(&x, &y, None).expect("PartitionTree fit failed");
        let preds = fitted.predict(&x).expect("PartitionTree predict failed");

        assert_eq!(
            preds.height(),
            x.height(),
            "row count mismatch after predict"
        );

        let col = preds
            .column("petal_length")
            .expect("missing column 'petal_length' in predictions");

        assert!(
            f64_varies(col),
            "PartitionTree regression: all predictions are identical — model did not learn"
        );
    }

    #[test]
    fn classification_predictions_vary() {
        let rows = load_iris();
        let (x, y) = classification_frames(&rows);

        let mut model = PartitionTree::with_defaults();
        let fitted = model.fit(&x, &y, None).expect("PartitionTree fit failed");
        let preds = fitted.predict(&x).expect("PartitionTree predict failed");

        assert_eq!(
            preds.height(),
            x.height(),
            "row count mismatch after predict"
        );

        let col = preds
            .column("species")
            .expect("missing column 'species' in predictions");

        assert!(
            str_varies(col),
            "PartitionTree classification: all predictions are identical — model did not learn"
        );
    }

    #[test]
    fn regression_beats_naive_baseline() {
        let rows = load_iris();
        let (x, y) = regression_frames(&rows);
        let true_vals: Vec<f64> = rows.iter().map(|r| r.petal_length).collect();

        let mut model = PartitionTree::with_defaults();
        let fitted = model.fit(&x, &y, None).expect("PartitionTree fit failed");
        let preds = fitted.predict(&x).expect("PartitionTree predict failed");

        let col = preds
            .column("petal_length")
            .expect("missing column 'petal_length' in predictions");

        let model_mse = regression_mse(col, &true_vals);
        let baseline_mse = naive_regression_mse(&true_vals);

        assert!(
            model_mse < baseline_mse * 0.5,
            "PartitionTree MSE {model_mse:.4} should be less than 50% of naive baseline MSE {baseline_mse:.4}"
        );
    }

    #[test]
    fn classification_beats_naive_baseline() {
        let rows = load_iris();
        let (x, y) = classification_frames(&rows);
        let true_labels: Vec<&str> = rows
            .iter()
            .map(|r| species_to_cat_label(&r.species))
            .collect();

        let mut model = PartitionTree::with_defaults();
        let fitted = model.fit(&x, &y, None).expect("PartitionTree fit failed");
        let preds = fitted.predict(&x).expect("PartitionTree predict failed");

        let col = preds
            .column("species")
            .expect("missing column 'species' in predictions");

        let accuracy = classification_accuracy(col, &true_labels);
        // Iris has 3 balanced classes → naive (majority-class) accuracy = 1/3 ≈ 0.33.
        // A trained tree should comfortably exceed 0.70.
        assert!(
            accuracy > 0.70,
            "PartitionTree accuracy {accuracy:.2} should exceed 0.70 (naive baseline ≈ 0.33)"
        );
    }
}

// ---------------------------------------------------------------------------
// PartitionForest tests
// ---------------------------------------------------------------------------

mod forest_tests {
    use super::*;

    /// Small forest suited for fast tests: 10 trees, 20 leaves each.
    fn small_forest() -> PartitionForest {
        PartitionForest::new(
            /* n_estimators */ 10,
            /* max_leaves */ 20,
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
        )
    }

    #[test]
    fn regression_predictions_vary() {
        let rows = load_iris();
        let (x, y) = regression_frames(&rows);

        let mut model = small_forest();
        let fitted = model.fit(&x, &y, None).expect("PartitionForest fit failed");
        let preds = fitted.predict(&x).expect("PartitionForest predict failed");

        assert_eq!(
            preds.height(),
            x.height(),
            "row count mismatch after predict"
        );

        let col = preds
            .column("petal_length")
            .expect("missing column 'petal_length' in predictions");

        assert!(
            f64_varies(col),
            "PartitionForest regression: all predictions are identical — model did not learn"
        );
    }

    #[test]
    fn classification_predictions_vary() {
        let rows = load_iris();
        let (x, y) = classification_frames(&rows);

        let mut model = small_forest();
        let fitted = model.fit(&x, &y, None).expect("PartitionForest fit failed");
        let preds = fitted.predict(&x).expect("PartitionForest predict failed");

        assert_eq!(
            preds.height(),
            x.height(),
            "row count mismatch after predict"
        );

        let col = preds
            .column("species")
            .expect("missing column 'species' in predictions");

        assert!(
            str_varies(col),
            "PartitionForest classification: all predictions are identical — model did not learn"
        );
    }

    #[test]
    fn regression_beats_naive_baseline() {
        let rows = load_iris();
        let (x, y) = regression_frames(&rows);
        let true_vals: Vec<f64> = rows.iter().map(|r| r.petal_length).collect();

        let mut model = small_forest();
        let fitted = model.fit(&x, &y, None).expect("PartitionForest fit failed");
        let preds = fitted.predict(&x).expect("PartitionForest predict failed");

        let col = preds
            .column("petal_length")
            .expect("missing column 'petal_length' in predictions");

        let model_mse = regression_mse(col, &true_vals);
        let baseline_mse = naive_regression_mse(&true_vals);

        // small_forest uses only 20 leaves per tree (for test speed), so we use a
        // looser threshold (80%) compared to the full-capacity PartitionTree (50%).
        assert!(
            model_mse < baseline_mse * 0.80,
            "PartitionForest MSE {model_mse:.4} should be less than 80% of naive baseline MSE {baseline_mse:.4}"
        );
    }

    #[test]
    fn classification_beats_naive_baseline() {
        let rows = load_iris();
        let (x, y) = classification_frames(&rows);
        let true_labels: Vec<&str> = rows
            .iter()
            .map(|r| species_to_cat_label(&r.species))
            .collect();

        let mut model = small_forest();
        let fitted = model.fit(&x, &y, None).expect("PartitionForest fit failed");
        let preds = fitted.predict(&x).expect("PartitionForest predict failed");

        let col = preds
            .column("species")
            .expect("missing column 'species' in predictions");

        let accuracy = classification_accuracy(col, &true_labels);
        // Iris has 3 balanced classes → naive (majority-class) accuracy = 1/3 ≈ 0.33.
        // A trained forest should comfortably exceed 0.70.
        assert!(
            accuracy > 0.70,
            "PartitionForest accuracy {accuracy:.2} should exceed 0.70 (naive baseline ≈ 0.33)"
        );
    }
}
