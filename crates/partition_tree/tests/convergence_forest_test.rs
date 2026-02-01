use estimator_api::api::Estimator;
use partition_tree::estimator_forest::PartitionForest;
use polars::prelude::*;

fn generate_sample_dataframe(n_samples: usize, n_features: usize) -> DataFrame {
    let mut x1_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);
    let mut x2_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);
    let mut noise_cols: Vec<Vec<Option<f64>>> =
        vec![vec![None; n_samples]; n_features.saturating_sub(2)];
    let mut target_strs: Vec<Option<&str>> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x1 = if i % 4 < 2 { 1.0 } else { 2.0 };
        let x2 = if i % 3 == 0 { 1.0 } else { 2.0 };
        x1_vals.push(Some(x1));
        x2_vals.push(Some(x2));

        for (j, col) in noise_cols.iter_mut().enumerate() {
            let jj = j + 2;
            let noise_value = match jj % 4 {
                0 => ((i * 7 + jj * 3) % 100) as f64 / 10.0,
                1 => {
                    if (i + jj) % 5 == 0 {
                        10.0
                    } else {
                        5.0
                    }
                }
                2 => ((i * 13 + jj * 7) % 50) as f64 / 5.0,
                _ => ((i * 11 + jj * 5) % 30) as f64 / 3.0,
            };
            col[i] = Some(noise_value);
        }

        let target = if x1 > 1.7 { "B" } else { "A" };
        target_strs.push(Some(target));
    }

    let mut cols: Vec<Series> = Vec::new();
    cols.push(Series::new(PlSmallStr::from_static("x1"), x1_vals));
    if n_features >= 2 {
        cols.push(Series::new(PlSmallStr::from_static("x2"), x2_vals));
    }
    for (k, col) in noise_cols.into_iter().enumerate() {
        let name = format!("noise_{}", k);
        cols.push(Series::new(PlSmallStr::from_str(&name), col));
    }

    let target_utf8 = Series::new(PlSmallStr::from_static("target_a"), target_strs);
    let cats = FrozenCategories::new(["A", "B"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let target_enum = target_utf8.cast(&enum_dt).unwrap();
    cols.push(target_enum);

    let cols_as_columns: Vec<Column> = cols.into_iter().map(|s| s.into()).collect();
    DataFrame::new(cols_as_columns).unwrap()
}

fn build_classification_forest() -> PartitionForest {
    PartitionForest::new(1, 8, 2, 0, 0, 0, 0.0, 5, 1e-9, 0.0, f64::INFINITY, f64::INFINITY, 0.0, None, None, 0, None, Some(42))
}

fn generate_regression_dataframe(n_samples: usize, n_features: usize) -> DataFrame {
    let mut x1_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);
    let mut x2_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);
    let mut noise_cols: Vec<Vec<Option<f64>>> =
        vec![vec![None; n_samples]; n_features.saturating_sub(2)];
    let mut target_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x1 = if i % 4 < 2 { 1.0 } else { 2.0 };
        let x2 = if i % 3 == 0 { 1.0 } else { 2.0 };
        x1_vals.push(Some(x1));
        x2_vals.push(Some(x2));

        for (j, col) in noise_cols.iter_mut().enumerate() {
            let jj = j + 2;
            let noise_value = match jj % 4 {
                0 => ((i * 7 + jj * 3) % 100) as f64 / 10.0,
                1 => {
                    if (i + jj) % 5 == 0 {
                        10.0
                    } else {
                        5.0
                    }
                }
                2 => ((i * 13 + jj * 7) % 50) as f64 / 5.0,
                _ => ((i * 11 + jj * 5) % 30) as f64 / 3.0,
            };
            col[i] = Some(noise_value);
        }

        let y = 2.0 * x1;
        target_vals.push(Some(y));
    }

    let mut cols: Vec<Series> = Vec::new();
    cols.push(Series::new(PlSmallStr::from_static("x1"), x1_vals));
    if n_features >= 2 {
        cols.push(Series::new(PlSmallStr::from_static("x2"), x2_vals));
    }
    for (k, col) in noise_cols.into_iter().enumerate() {
        let name = format!("noise_{}", k);
        cols.push(Series::new(PlSmallStr::from_str(&name), col));
    }
    cols.push(Series::new(
        PlSmallStr::from_static("target_y"),
        target_vals,
    ));

    let cols_as_columns: Vec<Column> = cols.into_iter().map(|s| s.into()).collect();
    DataFrame::new(cols_as_columns).unwrap()
}

fn build_regression_forest() -> PartitionForest {
    PartitionForest::new(
        1,             // n_estimators
        6,             // max_iter
        2,             // min_samples_split
        0,             // min_samples_leaf_y
        0,             // min_samples_leaf_x
        0,             // min_samples_leaf
        0.0,           // min_target_volume
        6,             // max_depth
        0.0,           // min_split_gain
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        1.0,           // boundaries_expansion_factor
        None,          // max_features
        None,          // max_samples
        0,             // exploration_split_budget
        None,          // feature_split_fraction
        Some(1337),    // seed
    )
}

fn split_features_targets(df: &DataFrame) -> (DataFrame, DataFrame) {
    let mut feature_cols: Vec<PlSmallStr> = Vec::new();
    let mut target_cols: Vec<PlSmallStr> = Vec::new();

    for column in df.get_columns() {
        let name = column.name();
        if name.as_str().starts_with("target") {
            target_cols.push(name.clone());
        } else {
            feature_cols.push(name.clone());
        }
    }

    let features = df.select(feature_cols.clone()).expect("select features");
    let targets = df.select(target_cols.clone()).expect("select targets");

    (features, targets)
}

#[test]
fn partition_forest_convergence_zero_error() {
    let n_samples = 200;
    let n_features = 4;
    let df = generate_sample_dataframe(n_samples, n_features);
    let (x, y) = split_features_targets(&df);

    let mut forest = build_classification_forest();
    let forest = forest.fit(&x, &y, None).expect("forest fit");
    let preds = forest.predict(&x).expect("forest predict");

    let target_col_name = "target_a";
    let predicted_col_name = target_col_name
        .strip_prefix("target_")
        .unwrap_or(target_col_name);

    let target_col = df.column(target_col_name).expect("target column");
    let predicted_col = preds.column(predicted_col_name).expect("predicted column");

    let mismatches = (0..x.height())
        .filter(|&idx| {
            let actual = target_col.get(idx).expect("actual target").to_string();
            let predicted = predicted_col
                .get(idx)
                .expect("predicted target")
                .to_string();
            actual != predicted
        })
        .count();

    assert_eq!(mismatches, 0, "Expected zero classification error");
}

#[test]
fn partition_forest_regression_error_below_threshold() {
    let n_samples = 4;
    let n_features = 3;
    let df = generate_regression_dataframe(n_samples, n_features);
    let (x, y) = split_features_targets(&df);

    let mut forest = build_regression_forest();
    let forest = forest.fit(&x, &y, None).expect("forest fit");
    let preds = forest.predict(&x).expect("forest predict");

    let target_col_name = "target_y";
    let predicted_col_name = target_col_name
        .strip_prefix("target_")
        .unwrap_or(target_col_name);

    let actual = y
        .column(target_col_name)
        .expect("target_y column")
        .f64()
        .expect("target_y should be f64")
        .clone();
    let predicted = preds
        .column(predicted_col_name)
        .expect("predicted y column")
        .f64()
        .expect("predicted y should be f64")
        .clone();

    assert_eq!(
        predicted.null_count(),
        0,
        "Predictions must not contain nulls"
    );

    let mut total_abs_error = 0.0_f64;
    let mut total_abs_actual = 0.0_f64;
    for (idx, (a, p)) in actual
        .clone()
        .into_no_null_iter()
        .zip(predicted.clone().into_no_null_iter())
        .enumerate()
    {
        assert!(
            !p.is_nan(),
            "Prediction at index {} is NaN; expected finite value",
            idx
        );
        total_abs_error += (p - a).abs();
        total_abs_actual += a.abs();

        if (p - a).abs() > 1e-6 {
            println!(
                "Index {}: actual = {:.6}, predicted = {:.6}, abs error = {:.6}",
                idx,
                a,
                p,
                (p - a).abs()
            );
        }
    }

    let percentage_error = if total_abs_actual == 0.0 {
        100.0 * total_abs_error
    } else {
        (total_abs_error / total_abs_actual) * 100.0
    };

    // With sample-weighted trees, allow a small relative error rather than
    // requiring perfect interpolation on this tiny dataset.
    assert!(
        percentage_error < 40.0,
        "Relative error {:.6}% exceeded 40% threshold",
        percentage_error
    );
}
