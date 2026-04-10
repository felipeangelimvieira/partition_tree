//! Helper binary for the determinism test.
//!
//! Fits a small forest with a fixed seed and prints the prediction values
//! to stdout, one f64 per line. The test runner invokes this binary twice
//! in separate processes and asserts identical output.

use estimators::api::Estimator;
use partition_tree::PartitionForest;
use polars::prelude::*;

fn main() {
    let n_rows = 100;
    let n_x_cols = 20; // many columns to increase HashMap ordering variance

    let mut x_cols: Vec<Column> = Vec::new();
    for c in 0..n_x_cols {
        let vals: Vec<Option<f64>> = (0..n_rows)
            .map(|i| Some(((i * (c + 1)) as f64).sin()))
            .collect();
        x_cols.push(Column::new(PlSmallStr::from_string(format!("x{c}")), vals));
    }
    let y: Vec<Option<f64>> = (0..n_rows)
        .map(|i| Some(if i % 4 < 2 { 2.0 } else { 4.0 }))
        .collect();

    let x = DataFrame::new(x_cols).unwrap();
    let y_df = DataFrame::new(vec![Column::new(PlSmallStr::from_static("y"), y)]).unwrap();

    let mut model = PartitionForest::new(
        /* n_estimators */ 10,
        /* max_leaves */ 16,
        /* boundaries_expansion_factor */ 0.0,
        /* min_samples_xy */ 0.0,
        /* min_samples_x */ 0.0,
        /* min_samples_y */ 0.0,
        /* min_gain */ 0.0,
        /* min_volume_fraction */ 0.0,
        /* max_depth */ 8,
        /* min_samples_split */ 2.0,
        /* max_samples */ None,
        /* replace */ true,
        /* max_features */ None,
        /* loss */ None,
        /* seed */ Some(42),
    );

    let fitted = model.fit(&x, &y_df, None).expect("fit failed");
    let preds = fitted.predict(&x).expect("predict failed");
    let col = preds.column("y").unwrap().f64().unwrap();

    for i in 0..preds.height() {
        println!("{:.17e}", col.get(i).unwrap());
    }

    // Also output the per-tree probability distributions for deeper comparison
    let dists = fitted.predict_proba(&x).unwrap();
    for d in &dists {
        let mv = d.mean_vector();
        let mut keys: Vec<_> = mv.keys().collect();
        keys.sort();
        for k in keys {
            let vals = &mv[k];
            for v in vals {
                println!("{:.17e}", v);
            }
        }
    }
}
