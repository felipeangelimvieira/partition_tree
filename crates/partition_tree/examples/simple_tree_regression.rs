use partition_tree::tree::{LeafReason, Tree};
use polars::prelude::*;
use std::time::Instant;

// Build a synthetic regression dataset where the target is continuous (Float64)
fn generate_sample_dataframe(n_samples: usize, n_features: usize) -> DataFrame {
    // Core numeric features x1, x2
    let mut x1_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);
    let mut x2_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);

    // Additional noisy numeric features
    let mut noise_cols: Vec<Vec<Option<f64>>> =
        vec![vec![None; n_samples]; n_features.saturating_sub(2)];

    // Continuous target
    let mut target_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Make x1 and x2 have simple discrete structure but still numeric
        let x1 = if i % 4 < 2 { 1.0 } else { 2.0 }; // toggles every 2
        let x2 = if i % 3 == 0 { 1.0 } else { 2.0 }; // every 3
        x1_vals.push(Some(x1));
        x2_vals.push(Some(x2));

        // Deterministic "noise" without RNG dependency
        // Range roughly in [-0.5, 0.5)
        let base_noise = ((i * 13 + 7) % 10) as f64 / 10.0 - 0.5;

        // Populate extra noise feature columns
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

        // Continuous target: simple linear function of x1 and x2 plus noise
        // y = 2*x1 - 1*x2 + noise
        let y = 2.0 * x1;
        target_vals.push(Some(y));
    }

    // Build columns
    let mut cols: Vec<Series> = Vec::new();
    cols.push(Series::new(PlSmallStr::from_static("x1"), x1_vals));
    if n_features >= 2 {
        cols.push(Series::new(PlSmallStr::from_static("x2"), x2_vals));
    }
    for (k, col) in noise_cols.into_iter().enumerate() {
        let name = format!("noise_{}", k);
        cols.push(Series::new(PlSmallStr::from_str(&name), col));
    }

    // Continuous target column name must start with "target" so the library treats it as Y
    cols.push(Series::new(
        PlSmallStr::from_static("target_y"),
        target_vals,
    ));

    let cols_as_columns: Vec<Column> = cols.into_iter().map(|s| s.into()).collect();
    DataFrame::new(cols_as_columns).unwrap()
}

fn main() {
    println!("=== Simple Tree Regression (Polars) ===");

    let n_samples = 20000;
    let n_features = 80; // x1, x2, and a few noise features
    let df = generate_sample_dataframe(n_samples, n_features);

    println!(
        "DataFrame shape: {{ rows: {}, cols: {} }}",
        df.height(),
        df.width()
    );
    println!("Columns: {:?}", df.get_column_names());

    // Tree hyperparameters (kept permissive for demo)
    let mut tree = Tree::new(
        12,            // max_iter
        2,             // min_samples_split
        0,             // min_samples_leaf_y
        0,             // min_samples_leaf_x
        0,             // min_samples_leaf
        6,             // max_depth
        0.0,           // min_target_volume
        1e-8,          // min_split_gain
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        0.0,
        None,
        None,
        0,
        None,       // feature_split_fraction
        Some(42),   // seed
    );

    println!("\nFitting tree...");
    let start_time = Instant::now();

    let sample_weights =
        Float64Chunked::full(PlSmallStr::from_static("sample_weights"), 1.0, df.height());
    tree.fit(&df, Some(sample_weights));
    let duration = start_time.elapsed();
    println!("Tree fitting took: {:?}", duration);

    println!("\n=== Results ===");
    println!("Total nodes: {}", tree.num_nodes());
    println!("Number of leaves: {}", tree.get_leaves().len());
    println!("Leaves: {:?}", tree.get_leaves());

    // Split history
    println!("\nSplit history (parent -> [left, right]) and split points:");
    for (i, rec) in tree.get_split_history().iter().enumerate() {
        let value = rec.split_result.split_value();
        let gain = rec.split_result.gain();
        println!(
            "  {:>2}. parent={} col='{}' value={:.6} gain={:.6} -> [{} , {}]",
            i,
            rec.parent_index,
            rec.column_name,
            value,
            gain,
            rec.left_child_index,
            rec.right_child_index
        );
    }

    // Leaf details
    println!("\nLeaf details:");
    for (i, &leaf_idx) in tree.get_leaves().iter().enumerate() {
        if let Some(leaf) = tree.get_node(leaf_idx) {
            let reason_str = match tree.get_leaf_reasons().get(&leaf_idx) {
                Some(LeafReason::MinSamplesSplit(n)) => format!("MinSamplesSplit(n={})", n),
                Some(LeafReason::MinSamplesLeaf(n)) => format!("MinSamplesLeaf(n={})", n),
                Some(LeafReason::MaxDepth(d)) => format!("MaxDepth(depth={})", d),
                Some(LeafReason::NoValidSplit(msg)) => format!("NoValidSplit(msg={})", msg),
                Some(LeafReason::IterationLimit) => "IterationLimit".to_string(),
                None => "Unknown".to_string(),
            };
            println!(
                "Leaf {} (node {}): {} samples at depth {} - reason: {}",
                i,
                leaf_idx,
                leaf.indices_xy.len(),
                leaf.depth,
                reason_str,
            );
        }
    }

    // Match leaves by X-only columns and show first few rows
    println!("\nX-only matches for first 10 rows:");
    let matches = tree.match_leaves_given_x(&df);
    for i in 0..std::cmp::min(10, matches.len()) {
        println!("row {} -> leaf_idxs {:?}", i, matches[i]);
    }

    // predict_mean is a placeholder in the current implementation; ensure it runs
    let preds_df = tree.predict_mean(&df);
    let pdf_values = tree.predict_mass(&df);
    println!(
        "\nPredictions DataFrame (placeholder): shape = {{ rows: {}, cols: {} }}",
        preds_df.height(),
        preds_df.width()
    );

    println!("\nPredictions vs true target (first 10 rows):");
    let max_rows = std::cmp::min(10, df.height());
    let actual = df
        .column("target_y")
        .expect("target_y column present")
        .f64()
        .expect("target_y should be f64")
        .clone();
    let predicted = preds_df
        .column("target_y")
        .expect("predicted target_y column present")
        .f64()
        .expect("predicted target_y should be f64")
        .clone();
    for idx in 0..max_rows {
        let actual_val = actual.get(idx).unwrap_or(f64::NAN);
        let predicted_val = predicted.get(idx).unwrap_or(f64::NAN);
        let pdf_val = pdf_values.get(idx).copied().unwrap_or(f64::NAN);
        println!(
            "  row {:>3}: predicted = {:>8.4} | actual = {:>8.4} | pdf = {:>8.4}",
            idx, predicted_val, actual_val, pdf_val
        );
    }
}
