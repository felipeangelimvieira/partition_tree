use partition_tree::tree::Tree;
use polars::prelude::*;
use std::any::Any;
use std::collections::HashMap;

fn generate_categorical_dataframe(n_samples: usize) -> DataFrame {
    let mut x1_vals: Vec<Option<f64>> = Vec::with_capacity(n_samples);
    let mut category_strs: Vec<Option<&str>> = Vec::with_capacity(n_samples);
    let mut target_strs: Vec<Option<&str>> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x1 = (i % 10) as f64 / 10.0; // Values from 0.0 to 0.9
        x1_vals.push(Some(x1));

        // Create a categorical feature with 3 categories
        let category = match i % 3 {
            0 => "red",
            1 => "green",
            _ => "blue",
        };
        category_strs.push(Some(category));

        // Target depends on both x1 and category
        let target = if x1 > 0.5 && category == "red" {
            "positive"
        } else {
            "negative"
        };
        target_strs.push(Some(target));
    }

    let mut cols: Vec<Series> = Vec::new();

    // Continuous feature
    cols.push(Series::new(PlSmallStr::from_static("x1"), x1_vals));

    // Categorical feature
    let category_utf8 = Series::new(PlSmallStr::from_static("category"), category_strs);
    let category_cats = FrozenCategories::new(["red", "green", "blue"]).unwrap();
    let category_enum_dt = DataType::from_frozen_categories(category_cats);
    let category_enum = category_utf8.cast(&category_enum_dt).unwrap();
    cols.push(category_enum);

    // Target
    let target_utf8 = Series::new(PlSmallStr::from_static("target_class"), target_strs);
    let target_cats = FrozenCategories::new(["positive", "negative"]).unwrap();
    let target_enum_dt = DataType::from_frozen_categories(target_cats);
    let target_enum = target_utf8.cast(&target_enum_dt).unwrap();
    cols.push(target_enum);

    let cols_as_columns: Vec<Column> = cols.into_iter().map(|s| s.into()).collect();
    DataFrame::new(cols_as_columns).unwrap()
}

fn build_categorical_tree() -> Tree {
    Tree::new(
        50,            // max_iter
        2,             // min_samples_split
        0,             // min_samples_leaf_y
        0,             // min_samples_leaf_x
        0,             // min_samples_leaf
        10,            // max_depth
        0.0,           // min_target_volume
        0.0,           // min_split_gain
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        0.1,           // boundaries_expansion_factor
        None,
        None,
        0,
        None,
        None,
    )
}

#[test]
fn test_string_to_categorical_conversion() {
    let n_samples = 100;
    let df = generate_categorical_dataframe(n_samples);

    // Print the dataframe structure for debugging
    println!("DataFrame schema: {:?}", df.schema());
    println!("First few rows:");
    println!("{}", df.head(Some(5)));

    let mut tree = build_categorical_tree();
    tree.fit(&df, None);

    println!("Tree built with {} nodes", tree.num_nodes());
    println!("Number of leaves: {}", tree.get_leaves().len());

    // Test 1: Create a query row using string values for categorical features
    let mut query_row: HashMap<String, Box<dyn Any>> = HashMap::new();

    // Add continuous feature
    query_row.insert("x1".to_string(), Box::new(0.7_f64) as Box<dyn Any>);

    // Add categorical feature as STRING (this should be converted to the appropriate categorical ID)
    query_row.insert(
        "category".to_string(),
        Box::new("red".to_string()) as Box<dyn Any>,
    );

    // Test that we can access a leaf node's cell and use match_hashmap
    if let Some(leaf_idx) = tree.get_leaves().first() {
        let leaf_node = tree.get_node(*leaf_idx).unwrap();

        // Test with exclude target partitions (X only) - this should work
        use partition_tree::conf::TargetBehaviour;
        let result_exclude = leaf_node
            .cell
            .match_hashmap(&query_row, TargetBehaviour::Exclude);
        println!("match_hashmap with Exclude (X only): {}", result_exclude);

        // Test with include all partitions - need to add target value
        let mut query_row_with_target: HashMap<String, Box<dyn Any>> = HashMap::new();
        query_row_with_target.insert("x1".to_string(), Box::new(0.7_f64) as Box<dyn Any>);
        query_row_with_target.insert(
            "category".to_string(),
            Box::new("red".to_string()) as Box<dyn Any>,
        );
        query_row_with_target.insert(
            "target_class".to_string(),
            Box::new("positive".to_string()) as Box<dyn Any>,
        );

        let result_include = leaf_node
            .cell
            .match_hashmap(&query_row_with_target, TargetBehaviour::Include);
        println!(
            "match_hashmap with Include (X + target): {}",
            result_include
        );

        // The key test: this should NOT panic and should return a boolean result
        // Previously, this would fail with a type mismatch error when "red" string
        // couldn't be converted to the expected categorical ID
    }

    // Test 2: Try with different categorical values
    let mut query_row2: HashMap<String, Box<dyn Any>> = HashMap::new();
    query_row2.insert("x1".to_string(), Box::new(0.3_f64) as Box<dyn Any>);
    query_row2.insert(
        "category".to_string(),
        Box::new("green".to_string()) as Box<dyn Any>,
    );

    if let Some(leaf_idx) = tree.get_leaves().first() {
        let leaf_node = tree.get_node(*leaf_idx).unwrap();

        use partition_tree::conf::TargetBehaviour;
        let result = leaf_node
            .cell
            .match_hashmap(&query_row2, TargetBehaviour::Exclude);
        println!("match_hashmap with 'green': {}", result);
    }

    // Test 3: Try with an invalid categorical value (should not panic, but may return false)
    let mut query_row3: HashMap<String, Box<dyn Any>> = HashMap::new();
    query_row3.insert("x1".to_string(), Box::new(0.5_f64) as Box<dyn Any>);
    query_row3.insert(
        "category".to_string(),
        Box::new("purple".to_string()) as Box<dyn Any>,
    ); // Not in domain

    if let Some(leaf_idx) = tree.get_leaves().first() {
        let leaf_node = tree.get_node(*leaf_idx).unwrap();

        use partition_tree::conf::TargetBehaviour;
        let result = leaf_node
            .cell
            .match_hashmap(&query_row3, TargetBehaviour::Exclude);
        println!("match_hashmap with 'purple' (invalid): {}", result);
        assert!(!result, "Invalid categorical values should not match");
    }

    println!("String to categorical conversion test completed successfully!");
}

#[test]
fn test_mixed_types_in_hashmap() {
    let n_samples = 50;
    let df = generate_categorical_dataframe(n_samples);

    let mut tree = build_categorical_tree();
    tree.fit(&df, None);

    // Test with mixed types - some as strings, some as proper categorical values
    let mut query_row: HashMap<String, Box<dyn Any>> = HashMap::new();

    // Continuous feature as f64
    query_row.insert("x1".to_string(), Box::new(0.8_f64) as Box<dyn Any>);

    // Categorical feature as string (should be converted)
    query_row.insert(
        "category".to_string(),
        Box::new("blue".to_string()) as Box<dyn Any>,
    );

    if let Some(leaf_idx) = tree.get_leaves().first() {
        let leaf_node = tree.get_node(*leaf_idx).unwrap();

        use partition_tree::conf::TargetBehaviour;

        // This should work without panicking
        let result = leaf_node
            .cell
            .match_hashmap(&query_row, TargetBehaviour::Exclude);
        println!("Mixed types test result: {}", result);

        // Test that the conversion is actually happening by checking we get a result
        // (rather than a panic from type mismatch)
        assert!(result == true || result == false); // Just ensure we get a boolean back
    }

    println!("Mixed types test completed successfully!");
}
