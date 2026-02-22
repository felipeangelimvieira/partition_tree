/// V2 equivalent of simple_tree.rs — categorical target (Enum)
///
/// Expected to produce the same split structure as the v1 example:
///   Split 0: target_a (gain ≈ 0.693147)
///   Split 1: x1       (gain ≈ 0.346574)
///   Split 2: x1       (gain ≈ 0.346574)
///   → 7 nodes, 4 leaves
use std::sync::Arc;
use std::time::Instant;

use partition_tree::{
    ConditionalLogLoss, DTypeRegistry, PolarsDatasetView, SplitRestrictions, Tree, TreeBuilder,
    TreeBuilderConfig,
};
use partition_tree::dataset_view::DatasetView;
use polars::prelude::*;

// ---- Data generation (identical to v1 example) ----

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

// ---- main ----

fn main() {
    println!("=== Simple Tree V2 (Polars) ===");

    let n_samples = 200;
    let n_features = 4;
    let df = generate_sample_dataframe(n_samples, n_features);

    println!(
        "DataFrame shape: {{ rows: {}, cols: {} }}",
        df.height(),
        df.width()
    );
    println!("Columns: {:?}", df.get_column_names());

    // Map v1 params → v2
    //   v1: max_iter=8, min_samples_split=2, min_samples_leaf_y=0,
    //       min_samples_leaf_x=0, min_samples_leaf=0, max_depth=5,
    //       min_target_volume=0, min_split_gain=0.001, expansion=0.0
    let config = TreeBuilderConfig {
        max_leaves: 8 + 1, // max_iter + 1
        boundaries_expansion_factor: 0.0,
        restrictions: SplitRestrictions {
            min_samples_xy: 0.0,
            min_samples_x: 0.0,
            min_samples_y: 0.0,
            min_gain: 0.001,
            min_volume: 0.0,
            max_depth: 5,
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

    println!("\nFitting tree (v2)...");
    let start_time = Instant::now();
    let tree: Tree = builder.build(&dataset);
    let duration = start_time.elapsed();
    println!("Tree fitting took: {:?}", duration);

    // ---- Print results in the same format as v1 ----
    print_results(&tree);
}

fn print_results(tree: &Tree) {
    println!("\n=== Results ===");
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
