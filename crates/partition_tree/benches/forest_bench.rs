//! Criterion benchmarks for PartitionForest training and prediction.
//!
//! Run with:
//!   cargo bench --bench forest_bench
//!
//! Results are stored in `target/criterion/` with HTML reports.
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use polars::prelude::*;
use polars::datatypes::FrozenCategories;

use partition_tree::{
    ConditionalLogLoss, DTypeRegistry, PartitionForest, PolarsDatasetView,
    TreeBuilder, TreeBuilderConfig,
};

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

/// Deterministic regression dataset: y ≈ x1 + 0.5*x2 + noise.
fn make_regression_data(n_rows: usize, n_features: usize) -> (DataFrame, DataFrame) {
    let mut cols: Vec<Column> = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let vals: Vec<f64> = (0..n_rows)
            .map(|i| {
                let base = (i as f64) / (n_rows as f64) * 10.0;
                // Deterministic pseudo-noise via modular arithmetic
                let noise = ((i * 7 + j * 13) % 97) as f64 / 97.0 - 0.5;
                base + noise
            })
            .collect();
        cols.push(Column::new(
            PlSmallStr::from_string(format!("x{}", j + 1)),
            vals,
        ));
    }
    let x = DataFrame::new(cols).unwrap();

    let y_vals: Vec<f64> = (0..n_rows)
        .map(|i| {
            let x1 = (i as f64) / (n_rows as f64) * 10.0;
            let x2 = (i as f64) / (n_rows as f64) * 5.0;
            let noise = ((i * 11 + 3) % 53) as f64 / 53.0 - 0.5;
            x1 + 0.5 * x2 + noise
        })
        .collect();
    let y = DataFrame::new(vec![Column::new(PlSmallStr::from_static("y"), y_vals)]).unwrap();

    (x, y)
}

/// Deterministic classification dataset: 3 classes based on x1/x2 regions.
fn make_classification_data(n_rows: usize, n_features: usize) -> (DataFrame, DataFrame) {
    let mut cols: Vec<Column> = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let vals: Vec<f64> = (0..n_rows)
            .map(|i| {
                let base = (i as f64) / (n_rows as f64) * 10.0;
                let noise = ((i * 7 + j * 13) % 97) as f64 / 97.0 - 0.5;
                base + noise
            })
            .collect();
        cols.push(Column::new(
            PlSmallStr::from_string(format!("x{}", j + 1)),
            vals,
        ));
    }
    let x = DataFrame::new(cols).unwrap();

    let target_vals: Vec<&str> = (0..n_rows)
        .map(|i| {
            let x1 = (i as f64) / (n_rows as f64) * 10.0;
            if x1 < 3.3 {
                "A"
            } else if x1 < 6.6 {
                "B"
            } else {
                "C"
            }
        })
        .collect();

    let cats = FrozenCategories::new(["A", "B", "C"]).unwrap();
    let series = Series::new(PlSmallStr::from_static("label"), target_vals)
        .cast(&DataType::from_frozen_categories(cats))
        .unwrap();
    let y = DataFrame::new(vec![series.into()]).unwrap();

    (x, y)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_single_tree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_tree_build");
    group.sample_size(10);

    for &n_rows in &[500, 2_000, 10_000] {
        let (x, y) = make_regression_data(n_rows, 5);

        // Prepare combined XY dataset the way TreeBuilder expects
        let y_prefixed = y
            .clone()
            .lazy()
            .rename(["y"], ["target__y"], true)
            .collect()
            .unwrap();
        let xy =
            polars::functions::concat_df_horizontal(&[x.clone(), y_prefixed.clone()], true)
                .unwrap();
        let dataset = PolarsDatasetView::new(&xy);

        group.bench_with_input(
            BenchmarkId::new("regression", n_rows),
            &n_rows,
            |b, _| {
                b.iter(|| {
                    let config = TreeBuilderConfig {
                        max_leaves: 31,
                        seed: Some(42),
                        ..Default::default()
                    };
                    let loss: Box<dyn partition_tree::LossFunc> = Box::new(ConditionalLogLoss);
                    let registry = Arc::new(DTypeRegistry::default());
                    let builder = TreeBuilder::new(config, loss, registry);
                    builder.build(&dataset)
                });
            },
        );
    }

    group.finish();
}

fn bench_forest_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_fit");
    group.sample_size(10);

    for &(n_rows, n_trees) in &[(1_000, 10), (1_000, 50), (5_000, 10)] {
        let (x, y) = make_regression_data(n_rows, 5);

        group.bench_with_input(
            BenchmarkId::new(format!("rows={n_rows}"), n_trees),
            &n_trees,
            |b, &n_trees| {
                b.iter(|| {
                    let mut forest = PartitionForest {
                        n_estimators: n_trees,
                        max_leaves: 31,
                        seed: Some(42),
                        ..PartitionForest::with_defaults()
                    };
                    use estimators::api::Estimator;
                    forest.fit(&x, &y, None).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_forest_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_predict");
    group.sample_size(10);

    for &(n_rows, n_trees) in &[(1_000, 10), (1_000, 50), (1_000, 100)] {
        let (x_train, y_train) = make_regression_data(n_rows, 5);
        let (x_test, _) = make_regression_data(500, 5);

        let mut forest = PartitionForest {
            n_estimators: n_trees,
            max_leaves: 31,
            seed: Some(42),
            ..PartitionForest::with_defaults()
        };
        use estimators::api::Estimator;
        let fitted = forest.fit(&x_train, &y_train, None).unwrap();

        group.bench_with_input(
            BenchmarkId::new("predict", n_trees),
            &n_trees,
            |b, _| {
                b.iter(|| fitted.predict(&x_test).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("predict_proba", n_trees),
            &n_trees,
            |b, _| {
                b.iter(|| fitted.predict_proba(&x_test).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_forest_predict_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_predict_classification");
    group.sample_size(10);

    for &n_trees in &[10, 50, 100] {
        let (x_train, y_train) = make_classification_data(1_000, 5);
        let (x_test, _) = make_classification_data(500, 5);

        let mut forest = PartitionForest {
            n_estimators: n_trees,
            max_leaves: 31,
            seed: Some(42),
            ..PartitionForest::with_defaults()
        };
        use estimators::api::Estimator;
        let fitted = forest.fit(&x_train, &y_train, None).unwrap();

        group.bench_with_input(
            BenchmarkId::new("predict", n_trees),
            &n_trees,
            |b, _| {
                b.iter(|| fitted.predict(&x_test).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("predict_proba", n_trees),
            &n_trees,
            |b, _| {
                b.iter(|| fitted.predict_proba(&x_test).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_tree_build,
    bench_forest_fit,
    bench_forest_predict,
    bench_forest_predict_classification,
);
criterion_main!(benches);
