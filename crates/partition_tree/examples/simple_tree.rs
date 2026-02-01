use partition_tree::tree::{LeafReason, Tree};
use polars::prelude::*;
use std::time::Instant;

fn generate_sample_dataframe(n_samples: usize, n_features: usize) -> DataFrame {
    // Build meaningful numeric features x1, x2
    let mut x1_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);
    let mut x2_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);

    // Additional noisy numeric features
    let mut noise_cols: Vec<Vec<Option<f64>>> =
        vec![vec![None; n_samples]; n_features.saturating_sub(2)];

    // Target strings to cast to Enum
    let mut target_strs: Vec<Option<&str>> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x1 = if i % 4 < 2 { 1.0 } else { 2.0 };
        let x2 = if i % 3 == 0 { 1.0 } else { 2.0 };
        x1_vals.push(Some(x1));
        x2_vals.push(Some(x2));

        // Noisy features
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

        // Simple rule for target
        let target = if x1 > 1.7 { "B" } else { "A" };
        target_strs.push(Some(target));
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

    // Categorical/Enum target
    let target_utf8 = Series::new(PlSmallStr::from_static("target_a"), target_strs);
    let cats = FrozenCategories::new(["A", "B"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let target_enum = target_utf8.cast(&enum_dt).unwrap();
    cols.push(target_enum);

    let cols_as_columns: Vec<Column> = cols.into_iter().map(|s| s.into()).collect();
    DataFrame::new(cols_as_columns).unwrap()
}

fn main() {
    println!("=== Simple Tree (Polars) ===");

    let n_samples = 200;
    let n_features = 4; // x1, x2, and a couple of noise features
    let df = generate_sample_dataframe(n_samples, n_features);

    println!(
        "DataFrame shape: {{ rows: {}, cols: {} }}",
        df.height(),
        df.width()
    );
    println!("Columns: {:?}", df.get_column_names());

    // Tree hyperparameters adapted to current implementation
    let mut tree = Tree::new(
        8,             // max_iter
        2,             // min_samples_split
        0,             // min_samples_leaf_y
        0,             // min_samples_leaf_x
        0,             // min_samples_leaf
        5,             // max_depth
        0.0,           // min_target_volume (unused in current splits)
        0.001,         // min_split_gain (allow any)
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

    println!("DataFrame: {:?}", df);

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

    let preds_df = tree.predict_mean(&df);
    let pdf_values = tree.predict_pdf(&df);
    println!(
        "\nPredictions DataFrame: shape = {{ rows: {}, cols: {} }}",
        preds_df.height(),
        preds_df.width()
    );

    println!("\nPredictions vs true target (first 10 rows):");
    let max_rows = std::cmp::min(10, df.height());
    for idx in 0..max_rows {
        let actual = df
            .column("target_a")
            .expect("target_a column")
            .get(idx)
            .expect("target value")
            .to_string();
        let predicted = preds_df
            .column("target_a")
            .expect("predicted target_a column")
            .get(idx)
            .expect("predicted value")
            .to_string();
        let pdf_value = pdf_values.get(idx).copied().unwrap_or(f64::NAN);
        println!(
            "  row {:>3}: predicted = {:<6} | actual = {} | pdf = {:.6}",
            idx, predicted, actual, pdf_value
        );
    }
}
