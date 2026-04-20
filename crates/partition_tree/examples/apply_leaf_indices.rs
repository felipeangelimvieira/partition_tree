//! Example: using `apply` to get leaf indices.
//!
//! Demonstrates two scenarios for both `PartitionTree` and `PartitionForest`:
//!   1. Calling `apply` with **all** feature columns → each row maps to
//!      exactly one leaf (single-element inner `Vec`).
//!   2. Calling `apply` with a **subset** of feature columns → when a
//!      split column is absent the tree explores both children, so a row
//!      may reach multiple leaves.
//!
//! For `PartitionForest`, `apply` returns `Vec<Vec<Vec<usize>>>`:
//!   outer = per-tree, middle = per-row, inner = leaf indices.

use estimators::api::Estimator;
use partition_tree::{PartitionForest, PartitionTree};
use polars::prelude::*;

fn main() {
    // ── 1. Build a dataset where the target depends on *both* x1 and x2 ─
    //
    //  x1 ∈ {1, 2, 3, 4}  (cycles every 4 rows)
    //  x2 ∈ {10, 20, 30}   (cycles every 3 rows)
    //  y  = x1 * 10 + x2   → the tree must split on both features.
    let n = 300;
    let x1: Vec<Option<f64>> = (0..n).map(|i| Some((i % 4 + 1) as f64)).collect();
    let x2: Vec<Option<f64>> = (0..n).map(|i| Some(((i % 3 + 1) * 10) as f64)).collect();
    let y: Vec<Option<f64>> = (0..n)
        .map(|i| Some((i % 4 + 1) as f64 * 10.0 + ((i % 3 + 1) * 10) as f64))
        .collect();

    let x = DataFrame::new(vec![
        Column::new(PlSmallStr::from_static("x1"), x1),
        Column::new(PlSmallStr::from_static("x2"), x2),
    ])
    .unwrap();

    let y_df = DataFrame::new(vec![Column::new(PlSmallStr::from_static("y"), y)]).unwrap();

    // ── 2. Fit a PartitionTree ──────────────────────────────────────────
    let mut model = PartitionTree::new(
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
        /* replace */ false,
        /* max_features */ None,
        /* loss */ None,
        /* seed */ Some(42),
    );

    let fitted = model.fit(&x, &y_df, None).expect("fit failed");

    // Show the tree structure so we can see which columns are split on.
    println!("{}\n", fitted.tree.as_ref().unwrap());

    // ── 3. Apply with all features ──────────────────────────────────────
    // Every feature column the tree may split on is present, so each row
    // resolves to exactly one leaf.
    let leaf_indices_full = fitted.apply(&x).expect("apply failed");

    println!("=== apply with all feature columns ===");
    println!("Number of rows: {}", leaf_indices_full.len());
    for (i, leaves) in leaf_indices_full.iter().enumerate().take(10) {
        println!("  row {i}: leaf indices = {leaves:?}");
    }

    // Each row should reach exactly one leaf when every feature is present.
    let all_single = leaf_indices_full.iter().all(|v| v.len() == 1);
    println!("All rows reach exactly one leaf: {all_single}\n");

    // ── 4. Apply with a subset of columns ───────────────────────────────
    // Drop "x1" so that any split on x1 cannot be evaluated.
    // The tree will explore both children at those splits, potentially
    // returning multiple leaf indices per row.
    let x_subset = x.select(["x2"]).unwrap();

    let leaf_indices_subset = fitted.apply(&x_subset).expect("apply with subset failed");

    println!("=== apply with subset of feature columns (only x2) ===");
    println!("Number of rows: {}", leaf_indices_subset.len());
    for (i, leaves) in leaf_indices_subset.iter().enumerate().take(10) {
        println!("  row {i}: leaf indices = {leaves:?}");
    }

    let any_multi = leaf_indices_subset.iter().any(|v| v.len() > 1);
    println!("Some rows reach multiple leaves (due to missing split column): {any_multi}");

    // ══════════════════════════════════════════════════════════════════════
    // PartitionForest — apply
    // ══════════════════════════════════════════════════════════════════════
    //
    // PartitionForest::apply returns Vec<Vec<Vec<usize>>>:
    //   [tree_index][row_index] -> Vec<leaf_indices>

    println!("\n\n=== PartitionForest: apply ===\n");

    // ── 5. Fit a PartitionForest ────────────────────────────────────────
    let mut forest = PartitionForest::new(
        /* n_estimators */ 5,
        /* max_leaves */ 16,
        /* boundaries_expansion_factor */ 0.0,
        /* min_samples_xy */ 0.0,
        /* min_samples_x */ 0.0,
        /* min_samples_y */ 0.0,
        /* min_gain */ 0.0,
        /* min_volume_fraction */ 0.0,
        /* max_depth */ 8,
        /* min_samples_split */ 2.0,
        /* max_samples */ Some(0.8),
        /* replace */ true,
        /* max_features */ None,
        /* loss */ None,
        /* seed */ Some(42),
    );

    let fitted_forest = forest.fit(&x, &y_df, None).expect("forest fit failed");

    // ── 6. Forest apply with all features ──────────────────────────────
    // Returns one Vec<Vec<usize>> per tree, each inner Vec has one entry
    // per row (single leaf per row when every split column is present).
    let forest_leaves_full = fitted_forest.apply(&x).expect("forest apply failed");

    println!("=== forest apply with all feature columns ===");
    println!("Number of trees: {}", forest_leaves_full.len());
    for (tree_idx, tree_leaves) in forest_leaves_full.iter().enumerate() {
        println!("  tree {tree_idx} — first 5 rows: {:?}", &tree_leaves[..5]);
    }

    let all_single_forest = forest_leaves_full
        .iter()
        .all(|tree| tree.iter().all(|v| v.len() == 1));
    println!("All rows reach exactly one leaf per tree: {all_single_forest}\n");

    // ── 7. Forest apply with a subset of columns ────────────────────────
    // Dropping "x1" causes splits on x1 to fan out into multiple leaves.
    let forest_leaves_subset = fitted_forest
        .apply(&x_subset)
        .expect("forest apply with subset failed");

    println!("=== forest apply with subset of feature columns (only x2) ===");
    for (tree_idx, tree_leaves) in forest_leaves_subset.iter().enumerate() {
        let multi_count = tree_leaves.iter().filter(|v| v.len() > 1).count();
        println!(
            "  tree {tree_idx} — rows with multiple leaves: {multi_count}/{} — first 5 rows: {:?}",
            tree_leaves.len(),
            &tree_leaves[..5]
        );
    }
}
