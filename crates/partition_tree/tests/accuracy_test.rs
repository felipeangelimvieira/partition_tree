//! Accuracy tests for [`PartitionTree`] and [`PartitionForest`] on the
//! digits dataset (tests/data/X.csv, tests/data/y.csv).
//!
//! The digits dataset has 1797 samples, 64 features (8×8 pixel values),
//! and 10 classes (digits 0–9). Both estimators are trained and evaluated
//! on the same data. Accuracy must match known reference values.

use estimators::api::Estimator;
use partition_tree::estimators::{PartitionForest, PartitionTree};
use polars::prelude::*;

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// Load X and y DataFrames from the CSV files in tests/data/.
fn load_digits() -> (DataFrame, DataFrame) {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");

    // ── X ──────────────────────────────────────────────────────────────
    let x_path = base.join("X.csv");
    let x_full = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(x_path))
        .expect("failed to create X reader")
        .finish()
        .expect("failed to read X.csv");

    // Drop the unnamed row-index column (first column, header is empty string)
    let first_col = x_full.get_column_names()[0].to_string();
    let x = x_full
        .drop(&first_col)
        .expect("failed to drop index column");

    // ── y ──────────────────────────────────────────────────────────────
    let y_path = base.join("y.csv");
    let y_full = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(y_path))
        .expect("failed to create y reader")
        .finish()
        .expect("failed to read y.csv");

    // Drop the unnamed row-index column
    let first_col = y_full.get_column_names()[0].to_string();
    let y = y_full
        .drop(&first_col)
        .expect("failed to drop index column");

    (x, y)
}

/// Compute classification accuracy by rounding f64 predictions to the
/// nearest integer and comparing with the true integer labels.
fn accuracy(pred_col: &Column, true_col: &Column) -> f64 {
    let preds = pred_col.f64().expect("expected Float64 predictions");
    let trues = true_col.cast(&DataType::Int64).expect("cast y to i64");
    let trues = trues.i64().expect("expected Int64 true labels");

    let n = preds.len();
    let correct = (0..n)
        .filter(|&i| {
            if let (Some(p), Some(t)) = (preds.get(i), trues.get(i)) {
                p.round() as i64 == t
            } else {
                false
            }
        })
        .count();

    correct as f64 / n as f64
}

// ---------------------------------------------------------------------------
// PartitionTree test
// ---------------------------------------------------------------------------

#[test]
fn partition_tree_accuracy_above_threshold() {
    let (x, y) = load_digits();
    let target_name = y.get_column_names()[0].to_string();

    let mut model = PartitionTree::new(
        /* max_leaves */ 100000,
        /* boundaries_expansion_factor */ 0.0,
        /* min_samples_xy */ 1.0,
        /* min_samples_x */ 1.0,
        /* min_samples_y */ 1.0,
        /* min_gain */ 0.0,
        /* min_volume_fraction */ 0.0,
        /* max_depth */ 100000,
        /* min_samples_split */ 2.0,
        /* max_samples */ None,
        /* replace */ false,
        /* max_features */ None,
        /* loss */ None,
        /* seed */ Some(42),
        /* dtype_overrides */ std::collections::HashMap::new(),
    );

    let fitted = model.fit(&x, &y, None).expect("PartitionTree fit failed");
    let preds = fitted.predict(&x).expect("PartitionTree predict failed");

    assert_eq!(preds.height(), x.height(), "row count mismatch");

    let pred_col = preds
        .column(&target_name)
        .expect("missing target column in predictions");
    let true_col = y.column(&target_name).expect("missing target column in y");

    let acc = accuracy(pred_col, true_col);
    eprintln!("PartitionTree accuracy: {acc:.4}");

    let min_accuracy = 0.8358;
    assert!(
        acc >= min_accuracy,
        "PartitionTree accuracy {acc:.4} should be >= {min_accuracy:.4}"
    );
}

// ---------------------------------------------------------------------------
// PartitionForest test
// ---------------------------------------------------------------------------

#[test]
fn partition_forest_accuracy_above_threshold() {
    let (x, y) = load_digits();
    let target_name = y.get_column_names()[0].to_string();

    let mut model = PartitionForest::new(
        /* n_estimators */ 20,
        /* max_leaves */ 100000,
        /* boundaries_expansion_factor */ 0.0,
        /* min_samples_xy */ 0.0,
        /* min_samples_x */ 1.0,
        /* min_samples_y */ 1.0,
        /* min_gain */ 0.0,
        /* min_volume_fraction */ 0.0,
        /* max_depth */ 100000,
        /* min_samples_split */ 2.0,
        /* max_samples */ None,
        /* replace */ true,
        /* max_features */ None,
        /* loss */ None,
        /* seed */ Some(42),
    );

    let fitted = model.fit(&x, &y, None).expect("PartitionForest fit failed");
    let preds = fitted.predict(&x).expect("PartitionForest predict failed");

    assert_eq!(preds.height(), x.height(), "row count mismatch");

    let pred_col = preds
        .column(&target_name)
        .expect("missing target column in predictions");
    let true_col = y.column(&target_name).expect("missing target column in y");

    let acc = accuracy(pred_col, true_col);
    eprintln!("PartitionForest accuracy: {acc:.4}");

    let min_accuracy = 0.9794;
    assert!(
        acc >= min_accuracy,
        "PartitionForest accuracy {acc:.4} should be >= {min_accuracy:.4}"
    );
}
