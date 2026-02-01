use partition_tree::tree::Tree;
use polars::prelude::*;

/// Generate a simple dataset for testing determinism
fn generate_test_df(n_samples: usize, n_features: usize) -> DataFrame {
    let mut cols: Vec<Series> = Vec::new();

    // Feature columns (simple deterministic pattern)
    for f in 0..n_features {
        let values: Vec<Option<f64>> = (0..n_samples)
            .map(|i| Some(((i * 7 + f * 13) % 100) as f64 / 10.0))
            .collect();
        let series = Series::new(PlSmallStr::from(format!("x_{}", f)), values);
        cols.push(series);
    }

    // Target column (continuous)
    let target_values: Vec<Option<f64>> = (0..n_samples)
        .map(|i| Some(((i * 11) % 100) as f64 / 10.0))
        .collect();
    let target = Series::new(PlSmallStr::from_static("target_y"), target_values);
    cols.push(target);

    let columns: Vec<Column> = cols.into_iter().map(|s| s.into()).collect();
    DataFrame::new(columns).unwrap()
}

#[test]
fn test_tree_determinism_with_same_seed() {
    let df = generate_test_df(100, 5);

    // First tree fit
    let mut tree1 = Tree::new(
        10,            // max_iter
        2,             // min_samples_split
        1,             // min_samples_leaf_y
        1,             // min_samples_leaf_x
        1,             // min_samples_leaf
        5,             // max_depth
        0.0,           // min_target_volume
        0.0,           // min_split_gain
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        0.0,           // boundaries_expansion_factor
        None,          // max_samples
        None,          // max_features
        0,             // exploration_split_budget
        None,          // feature_split_fraction
        Some(42),      // seed
    );
    tree1.fit(&df, None);

    // Second tree fit with the same seed
    let mut tree2 = Tree::new(
        10,            // max_iter
        2,             // min_samples_split
        1,             // min_samples_leaf_y
        1,             // min_samples_leaf_x
        1,             // min_samples_leaf
        5,             // max_depth
        0.0,           // min_target_volume
        0.0,           // min_split_gain
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        0.0,           // boundaries_expansion_factor
        None,          // max_samples
        None,          // max_features
        0,             // exploration_split_budget
        None,          // feature_split_fraction
        Some(42),      // seed
    );
    tree2.fit(&df, None);

    // Compare tree structure (number of nodes)
    assert_eq!(
        tree1.num_nodes(),
        tree2.num_nodes(),
        "Trees should have same number of nodes with same seed"
    );

    // Compare leaves
    assert_eq!(
        tree1.get_leaves().len(),
        tree2.get_leaves().len(),
        "Trees should have same number of leaves with same seed"
    );
}

#[test]
fn test_tree_determinism_multiple_runs() {
    let df = generate_test_df(200, 10);

    let mut node_counts: Vec<usize> = Vec::new();
    let mut leaf_counts: Vec<usize> = Vec::new();

    for _ in 0..5 {
        let mut tree = Tree::new(
            20,            // max_iter
            2,             // min_samples_split
            1,             // min_samples_leaf_y
            1,             // min_samples_leaf_x
            1,             // min_samples_leaf
            8,             // max_depth
            0.0,           // min_target_volume
            0.0,           // min_split_gain
            0.0,           // min_density_value
            f64::INFINITY, // max_density_value
            f64::INFINITY, // max_measure_value
            0.0,           // boundaries_expansion_factor
            None,          // max_samples
            None,          // max_features
            0,             // exploration_split_budget
            None,          // feature_split_fraction
            Some(42),      // seed
        );
        tree.fit(&df, None);

        node_counts.push(tree.num_nodes());
        leaf_counts.push(tree.get_leaves().len());
    }

    // All runs should produce the same tree structure
    for i in 1..node_counts.len() {
        assert_eq!(
            node_counts[0], node_counts[i],
            "Run 0 and run {} should have same node count",
            i
        );
        assert_eq!(
            leaf_counts[0], leaf_counts[i],
            "Run 0 and run {} should have same leaf count",
            i
        );
    }
}

#[test]
fn test_tree_determinism_with_many_features() {
    // More features increases chance of parallel processing non-determinism
    let df = generate_test_df(500, 50);

    let mut node_counts: Vec<usize> = Vec::new();
    let mut leaf_counts: Vec<usize> = Vec::new();

    for _ in 0..10 {
        let mut tree = Tree::new(
            50,            // max_iter
            2,             // min_samples_split
            1,             // min_samples_leaf_y
            1,             // min_samples_leaf_x
            1,             // min_samples_leaf
            10,            // max_depth
            0.0,           // min_target_volume
            0.0,           // min_split_gain
            0.0,           // min_density_value
            f64::INFINITY, // max_density_value
            f64::INFINITY, // max_measure_value
            0.0,           // boundaries_expansion_factor
            None,          // max_samples
            None,          // max_features
            0,             // exploration_split_budget
            None,          // feature_split_fraction
            Some(42),      // seed
        );
        tree.fit(&df, None);

        node_counts.push(tree.num_nodes());
        leaf_counts.push(tree.get_leaves().len());
    }

    // All runs should produce the same tree structure
    for i in 1..node_counts.len() {
        assert_eq!(
            node_counts[0], node_counts[i],
            "Run 0 and run {} should have same node count (many features)",
            i
        );
        assert_eq!(
            leaf_counts[0], leaf_counts[i],
            "Run 0 and run {} should have same leaf count (many features)",
            i
        );
    }
}
