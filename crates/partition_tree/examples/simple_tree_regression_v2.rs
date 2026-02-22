/// V2 equivalent of simple_tree_regression.rs — continuous target (Float64)
///
/// Expected to produce the same split structure as the v1 example:
///   Split 0: target_y (gain ≈ 0.693147)
///   Split 1: x1       (gain ≈ 0.346574)
///   Split 2: x1       (gain ≈ 0.346574)
///   → 7 nodes, 4 leaves
///
/// Also runs v1 tree on the same data and compares predictions side-by-side.
use std::sync::Arc;
use std::time::Instant;

use partition_tree::tree::{LeafReason, Tree as V1Tree};
use partition_tree::v2::{
    ConditionalLogLoss, DTypeRegistry, PolarsDatasetView, SplitRestrictions, Tree, TreeBuilder,
    TreeBuilderConfig,
};
use partition_tree::v2::dataset_view::DatasetView;
use polars::prelude::*;

// ---- Data generation (identical to v1 example) ----

fn generate_sample_dataframe(n_samples: usize, n_features: usize) -> DataFrame {
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

// ---- main ----

fn main() {
    println!("=== Simple Tree Regression V2 (Polars) ===");

    let n_samples = 20_000;
    let n_features = 80;
    let df = generate_sample_dataframe(n_samples, n_features);

    println!(
        "DataFrame shape: {{ rows: {}, cols: {} }}",
        df.height(),
        df.width()
    );

    // ------------------------------------------------------------------
    // V2 tree
    // ------------------------------------------------------------------
    let config = TreeBuilderConfig {
        max_leaves: 12 + 1,
        boundaries_expansion_factor: 0.0,
        restrictions: SplitRestrictions {
            min_samples_xy: 0.0,
            min_samples_x: 0.0,
            min_samples_y: 0.0,
            min_gain: 1e-8,
            min_volume: 0.0,
            max_depth: 6,
            min_samples_split: 2.0,
        },
    };

    let dataset = PolarsDatasetView::new(&df);
    let n = dataset.n_rows() as f64;

    let builder = TreeBuilder::new(
        config,
        Box::new(ConditionalLogLoss::new(n)),
        Arc::new(DTypeRegistry::default()),
    );

    println!("\nFitting v2 tree...");
    let start_time = Instant::now();
    let v2_tree: Tree = builder.build(&dataset);
    let v2_duration = start_time.elapsed();
    println!("V2 tree fitting took: {:?}", v2_duration);

    print_v2_results(&v2_tree);

    // ------------------------------------------------------------------
    // V1 tree (same hyperparameters)
    // ------------------------------------------------------------------
    let mut v1_tree = V1Tree::new(
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

    println!("\n\nFitting v1 tree...");
    let start_time = Instant::now();
    let sample_weights =
        Float64Chunked::full(PlSmallStr::from_static("sample_weights"), 1.0, df.height());
    v1_tree.fit(&df, Some(sample_weights));
    let v1_duration = start_time.elapsed();
    println!("V1 tree fitting took: {:?}", v1_duration);

    print_v1_results(&v1_tree);

    // ------------------------------------------------------------------
    // Compare predictions
    // ------------------------------------------------------------------
    compare_predictions(&v1_tree, &v2_tree, &df, &dataset);
}

// ---- V2 result printing ----

fn print_v2_results(tree: &Tree) {
    println!("\n=== V2 Results ===");
    println!("Total nodes: {}", tree.n_nodes());
    println!("Number of leaves: {}", tree.n_leaves());
    println!("Leaves: {:?}", tree.leaves);

    println!("\nSplit history (parent -> [left, right]) and split points:");
    for (i, rec) in tree.split_history.iter().enumerate() {
        let value = match &tree.nodes[rec.left_child_index].cell.rules.get(&rec.col_name) {
            Some(rule) => {
                if let Some(ci) = rule.as_any().downcast_ref::<partition_tree::rules::ContinuousInterval>() {
                    ci.high
                } else if let Some(bt) = rule.as_any().downcast_ref::<partition_tree::rules::BelongsTo>() {
                    bt.values.len() as f64
                } else {
                    f64::NAN
                }
            }
            None => f64::NAN,
        };
        println!(
            "  {:>2}. parent={} col='{}' value={:.6} gain={:.6} -> [{} , {}]",
            i,
            rec.parent_index,
            rec.col_name,
            value,
            rec.gain,
            rec.left_child_index,
            rec.right_child_index
        );
    }

    println!("\nLeaf details:");
    for (i, &leaf_idx) in tree.leaves.iter().enumerate() {
        let node = &tree.nodes[leaf_idx];
        println!(
            "Leaf {} (node {}): w_xy={:.0} at depth {}",
            i, leaf_idx, node.w_xy, node.depth,
        );
    }

    println!("\n{}", tree);
}

// ---- V1 result printing ----

fn print_v1_results(tree: &V1Tree) {
    println!("\n=== V1 Results ===");
    println!("Total nodes: {}", tree.num_nodes());
    println!("Number of leaves: {}", tree.get_leaves().len());

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
}

// ---- Prediction comparison ----

fn compare_predictions(v1_tree: &V1Tree, v2_tree: &Tree, df: &DataFrame, dataset: &PolarsDatasetView) {
    println!("\n{:=^60}", "");
    println!("=== Prediction Comparison: V1 vs V2 ===");
    println!("{:=^60}", "");

    // V1 predictions
    let v1_start = Instant::now();
    let v1_preds = v1_tree.predict_mean(df);
    let v1_pred_time = v1_start.elapsed();

    // V2 predictions
    let v2_start = Instant::now();
    let v2_preds = v2_tree.predict_mean(dataset);
    let v2_pred_time = v2_start.elapsed();

    println!("\nPrediction time — V1: {:?}  |  V2: {:?}", v1_pred_time, v2_pred_time);

    let v1_col = v1_preds
        .column("target_y")
        .expect("v1 target_y")
        .f64()
        .expect("v1 f64")
        .clone();

    let v2_col = v2_preds
        .column("target_y")
        .expect("v2 target_y")
        .f64()
        .expect("v2 f64")
        .clone();

    let actual = df
        .column("target_y")
        .expect("target_y")
        .f64()
        .expect("f64")
        .clone();

    // Print first 20 rows
    let max_rows = std::cmp::min(20, df.height());
    println!(
        "\n{:>5} | {:>12} | {:>12} | {:>12} | {:>10}",
        "row", "actual", "v1_pred", "v2_pred", "v1==v2?"
    );
    println!("{:-^65}", "");

    let mut n_match = 0usize;
    let mut n_total = 0usize;
    let mut max_diff: f64 = 0.0;
    let mut sum_sq_diff_v1: f64 = 0.0;
    let mut sum_sq_diff_v2: f64 = 0.0;

    for idx in 0..df.height() {
        let a = actual.get(idx).unwrap_or(f64::NAN);
        let p1 = v1_col.get(idx).unwrap_or(f64::NAN);
        let p2 = v2_col.get(idx).unwrap_or(f64::NAN);
        let diff = (p1 - p2).abs();
        let match_str = if diff < 1e-10 { "✓" } else { "✗" };

        if diff < 1e-10 {
            n_match += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
        n_total += 1;
        sum_sq_diff_v1 += (a - p1).powi(2);
        sum_sq_diff_v2 += (a - p2).powi(2);

        if idx < max_rows {
            println!(
                "{:>5} | {:>12.6} | {:>12.6} | {:>12.6} | {:>10}",
                idx, a, p1, p2, match_str
            );
        }
    }

    let mse_v1 = sum_sq_diff_v1 / n_total as f64;
    let mse_v2 = sum_sq_diff_v2 / n_total as f64;

    println!("\n{:=^65}", " Summary ");
    println!("Total rows:              {}", n_total);
    println!("Exact matches (|Δ|<1e-10): {} / {} ({:.2}%)",
        n_match, n_total, 100.0 * n_match as f64 / n_total as f64);
    println!("Max |v1 - v2| diff:      {:.2e}", max_diff);
    println!("V1 MSE (vs actual):      {:.6e}", mse_v1);
    println!("V2 MSE (vs actual):      {:.6e}", mse_v2);

    // V2 distribution details for first 5 rows
    println!("\n--- V2 distribution details (first 5 rows) ---");
    let distributions = v2_tree.predict_distributions(dataset);
    for idx in 0..std::cmp::min(5, distributions.len()) {
        let dist = &distributions[idx];
        let mv = dist.mean_vector();
        let mass = dist.total_mass();
        println!(
            "  row {}: mean_vector={:?}  total_mass={:.4}",
            idx, mv, mass
        );
    }
}
