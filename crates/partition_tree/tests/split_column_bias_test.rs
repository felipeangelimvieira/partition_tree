use partition_tree::node::Node;
use partition_tree::split::{SplitRestrictions, find_best_split_column};
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

fn uniform_sample_weights(df: &DataFrame) -> Float64Chunked {
    Float64Chunked::full(PlSmallStr::from_static("sample_weights"), 1.0, df.height())
}

#[test]
fn test_find_best_split_column_no_first_column_bias() {
    // Create a simple dataset where multiple columns have identical splits
    let data = df! {
        "feature_a" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature_b" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // Identical to feature_a
        "feature_c" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // Identical to feature_a
        "target_density" => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    .expect("Failed to create test dataframe");

    // Create a node using the standard method
    let mut rng = StdRng::seed_from_u64(42);
    let node = Node::default_from_dataframe(&data, 0.0, None, &mut rng);
    let sample_weights = uniform_sample_weights(&data);

    let restrictions = SplitRestrictions::default();

    // Run the function multiple times and collect results
    let mut column_counts = HashMap::new();
    for _ in 0..10 {
        let (best_column, split_result) =
            find_best_split_column(&node, &data, &restrictions, None, &mut rng, &sample_weights, false);

        // Only count valid splits
        if split_result.is_valid() {
            *column_counts.entry(best_column).or_insert(0) += 1;
        }
    }

    println!("Column selection counts: {:?}", column_counts);

    // Since all features are identical, the selection should be deterministic
    // but not always favor the first column. The hash-based tie breaking
    // should distribute selections based on column name hashes.

    // At minimum, we should not have selected only "feature_a" every time
    // (which would indicate first-column bias)
    let feature_a_count = column_counts.get("feature_a").unwrap_or(&0);
    let total_selections = column_counts.values().sum::<i32>();

    if total_selections > 0 {
        // If we always selected feature_a, that would indicate bias
        assert!(
            *feature_a_count < total_selections,
            "Expected hash-based tie breaking to not always select the first column, but feature_a was selected {} out of {} times",
            feature_a_count,
            total_selections
        );
    }
}

#[test]
fn test_find_best_split_column_deterministic() {
    // Test that with identical inputs, we get identical outputs (deterministic)
    let data = df! {
        "col1" => [1.0, 2.0, 3.0, 4.0],
        "col2" => [1.0, 2.0, 3.0, 4.0], // Identical values
        "target_density" => [1.0, 1.0, 1.0, 1.0],
    }
    .expect("Failed to create test dataframe");

    let mut rng = StdRng::seed_from_u64(42);
    let node = Node::default_from_dataframe(&data, 0.0, None, &mut rng);
    let sample_weights = uniform_sample_weights(&data);

    let restrictions = SplitRestrictions::default();

    let mut rng = StdRng::seed_from_u64(42);
    // Run multiple times and ensure we get the same result
    let (first_column, first_result) =
        find_best_split_column(&node, &data, &restrictions, None, &mut rng, &sample_weights, false);

    for _ in 0..5 {
        rng = StdRng::seed_from_u64(42);
        let (column, result) =
            find_best_split_column(&node, &data, &restrictions, None, &mut rng, &sample_weights, false);
        assert_eq!(
            column, first_column,
            "Column selection should be deterministic"
        );
        assert_eq!(
            result.gain(),
            first_result.gain(),
            "Split gain should be deterministic"
        );
    }
}
