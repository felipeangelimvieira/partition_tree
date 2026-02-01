use partition_tree::conf::TargetBehaviour;
use partition_tree::rules::Rule;
use partition_tree::split::SplitResult;
use partition_tree::tree::{Tree, TreeBuilderStatus};
use polars::prelude::*;

fn build_df() -> DataFrame {
    // x: numeric with a None
    let x = Series::new(
        PlSmallStr::from_static("x"),
        &[Some(0.0_f64), Some(5.0), None, Some(10.0)],
    );

    // categorical: Enum
    let cat_utf8 = Series::new(
        PlSmallStr::from_static("cat"),
        &[Some("a"), Some("b"), Some("b"), Some("a")],
    );
    let cats = FrozenCategories::new(["a", "b"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let cat = cat_utf8.cast(&enum_dt).unwrap();

    // target column (prefix "target")
    let target_a = Series::new(
        PlSmallStr::from_static("target_a"),
        &[Some(1.0_f64), Some(2.0), Some(3.0), Some(4.0)],
    );

    DataFrame::new(vec![x.into(), cat.into(), target_a.into()]).unwrap()
}

fn build_df_with_enum_target() -> DataFrame {
    // x: numeric with a None
    let x = Series::new(
        PlSmallStr::from_static("x"),
        &[Some(0.0_f64), Some(5.0), None, Some(10.0)],
    );

    // categorical feature: Enum
    let cat_utf8 = Series::new(
        PlSmallStr::from_static("cat"),
        &[Some("a"), Some("b"), Some("b"), Some("a")],
    );
    let cats = FrozenCategories::new(["a", "b"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let cat = cat_utf8.cast(&enum_dt).unwrap();

    // numeric target column
    let target_a = Series::new(
        PlSmallStr::from_static("target_a"),
        &[Some(10.0_f64), Some(20.0), Some(30.0), Some(40.0)],
    );

    // categorical target column
    let target_cat_utf8 = Series::new(
        PlSmallStr::from_static("target_cat"),
        &[
            Some("apple"),
            Some("banana"),
            Some("banana"),
            Some("banana"),
        ],
    );
    let target_cat_enum = target_cat_utf8
        .cast(&DataType::from_frozen_categories(
            FrozenCategories::new(["apple", "banana"]).unwrap(),
        ))
        .unwrap();

    DataFrame::new(vec![
        x.into(),
        cat.into(),
        target_a.into(),
        target_cat_enum.into(),
    ])
    .unwrap()
}

fn build_df_categorical_feature_only() -> DataFrame {
    // Single categorical feature (Enum)
    let cat_utf8 = Series::new(
        PlSmallStr::from_static("cat"),
        &[Some("a"), Some("b"), Some("b"), Some("c")],
    );
    let cats = FrozenCategories::new(["a", "b", "c"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let cat = cat_utf8.cast(&enum_dt).unwrap();

    // Numeric target
    let target_a = Series::new(
        PlSmallStr::from_static("target_a"),
        &[Some(1.0_f64), Some(2.0), Some(3.0), Some(4.0)],
    );

    DataFrame::new(vec![cat.into(), target_a.into()]).unwrap()
}

fn build_df_with_unused_enum_categories() -> DataFrame {
    // Numeric feature
    let x = Series::new(
        PlSmallStr::from_static("x"),
        &[Some(0.0_f64), Some(1.0), Some(2.0), Some(3.0)],
    );

    // Categorical feature with an unused category in the mapping
    let cat_utf8 = Series::new(
        PlSmallStr::from_static("cat"),
        &[Some("a"), Some("b"), Some("a"), Some("b")],
    );
    let cat_enum = cat_utf8
        .cast(&DataType::from_frozen_categories(
            FrozenCategories::new(["a", "b", "c"]).unwrap(),
        ))
        .unwrap();

    // Numeric target
    let target_num = Series::new(
        PlSmallStr::from_static("target_num"),
        &[Some(1.0_f64), Some(2.0), Some(3.0), Some(4.0)],
    );

    // Categorical target with an unused category in the mapping
    let target_cat_utf8 = Series::new(
        PlSmallStr::from_static("target_cat"),
        &[Some("apple"), Some("banana"), Some("apple"), Some("banana")],
    );
    let target_cat_enum = target_cat_utf8
        .cast(&DataType::from_frozen_categories(
            FrozenCategories::new(["apple", "banana", "cherry"]).unwrap(),
        ))
        .unwrap();

    DataFrame::new(vec![
        x.into(),
        cat_enum.into(),
        target_num.into(),
        target_cat_enum.into(),
    ])
    .unwrap()
}

fn build_classification_df_no_split() -> DataFrame {
    // Simple numeric feature
    let x = Series::new(
        PlSmallStr::from_static("x"),
        &[Some(0.0_f64), Some(1.0), Some(2.0), Some(3.0)],
    );

    // Single categorical target column (Enum)
    let target_utf8 = Series::new(
        PlSmallStr::from_static("target_class"),
        &[Some("apple"), Some("banana"), Some("apple"), Some("banana")],
    );
    let cats = FrozenCategories::new(["apple", "banana"]).unwrap();
    let target_enum = target_utf8
        .cast(&DataType::from_frozen_categories(cats))
        .unwrap();

    DataFrame::new(vec![x.into(), target_enum.into()]).unwrap()
}

fn default_tree() -> Tree {
    Tree::new(
        6,                 // max_iter
        1,                 // min_samples_split
        0,                 // min_samples_leaf_y
        0,                 // min_samples_leaf_x
        0,                 // min_samples_leaf
        5,                 // max_depth
        0.0,               // min_target_volume
        f64::NEG_INFINITY, // min_split_gain (allow any)
        0.0,               // min_density_value
        f64::INFINITY,     // max_density_value
        f64::INFINITY,     // max_measure_value
        0.0,
        None,
        None,
        0,
        None,
        None,
    )
}

#[test]
fn tree_fit_and_structure() {
    let df = build_df();
    let mut tree = default_tree();
    tree.fit(&df, None);

    assert!(tree.num_nodes() >= 1);
    assert!(!tree.get_leaves().is_empty());
    assert!(tree.get_node(0).is_some());

    let history = tree.get_split_history();
    if !history.is_empty() {
        let first = &history[0];
        assert_eq!(first.parent_index, 0);
        assert!(first.left_child_index < tree.num_nodes());
        assert!(first.right_child_index < tree.num_nodes());

        // Column exists in df
        let mut cols = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        cols.sort();
        assert!(cols.contains(&first.column_name));

        // Split should not be InvalidSplit
        use partition_tree::split::SplitResult;
        assert!(!matches!(first.split_result, SplitResult::InvalidSplit(_)));
    }
}

#[test]
fn tree_match_leaves_given_x_basic() {
    let df = build_df();
    let mut tree = default_tree();
    tree.fit(&df, None);

    let matches = tree.match_leaves_given_x(&df);
    assert_eq!(matches.len(), df.height());
    let num_leaves = tree.get_leaves().len();
    for per_row in matches.iter() {
        assert!(!per_row.is_empty());
        for &lid in per_row {
            assert!(lid < num_leaves);
        }
    }
}

#[test]
fn tree_predict_mean_numeric_targets_returns_global_mean() {
    let df = build_df();
    let mut tree = default_tree();
    tree.fit(&df, None);

    let predictions = tree.predict_mean(&df);
    assert_eq!(predictions.height(), df.height());
    assert_eq!(predictions.get_column_names(), vec!["target_a"]);

    let probas = tree.predict_proba(&df);
    let root_cell = &tree.get_node(0).expect("root node").cell;
    let expected: Vec<f64> = probas
        .iter()
        .map(|dist| {
            let mean_map = dist.mean();
            let encoded = mean_map.get("target_a").expect("mean contains target_a");
            root_cell
                .get_rule_as_continuous_interval("target_a")
                .inverse_one_hot(encoded)
        })
        .collect();

    let predicted_col = predictions
        .column("target_a")
        .expect("target_a column present")
        .f64()
        .expect("target_a output is f64")
        .clone();

    assert_eq!(predicted_col.null_count(), 0);
    let predicted_values: Vec<f64> = predicted_col.into_no_null_iter().collect();
    assert_eq!(predicted_values, expected);
}

#[test]
fn tree_predict_mean_handles_multiple_target_types() {
    let df = build_df_with_enum_target();
    let mut tree = default_tree();
    tree.fit(&df, None);

    let predictions = tree.predict_mean(&df);
    assert_eq!(predictions.height(), df.height());
    assert_eq!(
        predictions.get_column_names(),
        vec!["target_a", "target_cat"]
    );

    let probas = tree.predict_proba(&df);
    let root_cell = &tree.get_node(0).expect("root node").cell;

    let expected_numeric: Vec<f64> = probas
        .iter()
        .map(|dist| {
            let mean_map = dist.mean();
            let encoded = mean_map.get("target_a").expect("mean contains target_a");
            root_cell
                .get_rule_as_continuous_interval("target_a")
                .inverse_one_hot(encoded)
        })
        .collect();

    let numeric_pred = predictions
        .column("target_a")
        .expect("numeric target column present")
        .f64()
        .expect("numeric predictions are f64")
        .clone();
    assert_eq!(numeric_pred.null_count(), 0);
    let numeric_values: Vec<f64> = numeric_pred.into_no_null_iter().collect();
    assert_eq!(numeric_values, expected_numeric);

    let cat_col = df
        .column("target_cat")
        .expect("original categorical column");
    let categorical_mapping = match cat_col.dtype() {
        DataType::Enum(_, mapping) => mapping,
        other => panic!("Unexpected dtype for target_cat: {:?}", other),
    };

    let expected_strings: Vec<String> = probas
        .iter()
        .map(|dist| {
            let mean_map = dist.mean();
            let encoded = mean_map
                .get("target_cat")
                .expect("mean contains target_cat");
            let code = root_cell
                .get_rule_as_belongs_to_u32("target_cat")
                .inverse_one_hot(encoded);
            categorical_mapping
                .cat_to_str(code)
                .expect("category exists")
                .to_string()
        })
        .collect();

    let categorical_pred = predictions
        .column("target_cat")
        .expect("categorical target column present");
    assert_eq!(categorical_pred.null_count(), 0);
    let predicted_strings: Vec<String> = (0..categorical_pred.len())
        .map(
            |idx| match categorical_pred.get(idx).expect("value present") {
                AnyValue::String(s) => s.to_string(),
                AnyValue::StringOwned(s) => s.to_string(),
                other => panic!("Unexpected categorical prediction value: {:?}", other),
            },
        )
        .collect();
    assert_eq!(predicted_strings, expected_strings);
}

#[test]
fn tree_exploration_splits_largest_cell() {
    // Use strict min_samples_split so the regular gain-based splitter marks root as leaf
    let mut tree = Tree::new(
        6,  // max_iter
        10, // min_samples_split (larger than dataset)
        0,
        0,
        0,
        5,
        0.0,
        f64::NEG_INFINITY,
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        0.0,
        None,
        None,
        1,
        None,
        None,
    );

    let df = build_df();
    tree.fit(&df, None);

    assert_eq!(tree.get_build_status(), &TreeBuilderStatus::SUCCESS);
    // Exploration should introduce at least one split
    assert!(tree.num_nodes() >= 3);
    assert!(tree.get_split_history().iter().any(|rec| matches!(
        rec.split_result,
        SplitResult::ContinuousSplit(..) | SplitResult::CategoricalSplit(..)
    )));
}

#[test]
fn tree_exploration_can_split_categorical() {
    // Force exploration and ensure it operates on categorical-only features
    let mut tree = Tree::new(
        4,  // max_iter
        10, // min_samples_split (larger than dataset)
        0,
        0,
        0,
        3,
        0.0,
        f64::NEG_INFINITY,
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        0.0,
        None,
        None,
        1,
        None,
        None,
    );

    let df = build_df_categorical_feature_only();
    tree.fit(&df, None);

    assert_eq!(tree.get_build_status(), &TreeBuilderStatus::SUCCESS);
    assert!(tree.num_nodes() >= 3);
    assert!(
        tree.get_split_history()
            .iter()
            .any(|rec| matches!(rec.split_result, SplitResult::CategoricalSplit(..)))
    );
}

#[test]
fn tree_handles_enum_categories_not_seen_in_training_rows() {
    let df_train = build_df_with_unused_enum_categories();
    let mut tree = default_tree();

    // Should build without panicking even though the category map includes unused values
    tree.fit(&df_train, None);
    assert_eq!(tree.get_build_status(), &TreeBuilderStatus::SUCCESS);

    // Build a prediction dataframe that contains a category ("c") not seen in training rows
    let predict_cat_utf8 = Series::new(
        PlSmallStr::from_static("cat"),
        &[Some("c"), Some("b"), Some("a"), Some("c")],
    );
    let predict_cat = predict_cat_utf8
        .cast(&DataType::from_frozen_categories(
            FrozenCategories::new(["a", "b", "c"]).unwrap(),
        ))
        .unwrap();

    let predict_df = DataFrame::new(vec![
        df_train.column("x").unwrap().clone(),
        predict_cat.into(),
        df_train.column("target_num").unwrap().clone(),
        df_train.column("target_cat").unwrap().clone(),
    ])
    .unwrap();

    // Predict on the dataframe containing the unseen-but-mapped category
    let predictions = tree.predict_mean(&predict_df);
    assert_eq!(predictions.height(), predict_df.height());
    assert_eq!(
        predictions.get_column_names(),
        vec!["target_num", "target_cat"]
    );

    let cat_pred = predictions
        .column("target_cat")
        .expect("categorical target present");
    assert_eq!(cat_pred.null_count(), 0);

    // All predicted categories should be among the observed training categories (no crash on unseen mapping entries)
    let predicted_strings: Vec<String> = (0..cat_pred.len())
        .map(|idx| match cat_pred.get(idx).expect("value present") {
            AnyValue::String(s) => s.to_string(),
            AnyValue::StringOwned(s) => s.to_string(),
            other => panic!("Unexpected categorical prediction value: {:?}", other),
        })
        .collect();

    for value in predicted_strings {
        assert!(value == "apple" || value == "banana");
    }
}

#[test]
fn tree_predict_proba_returns_valid_distributions() {
    let df = build_df();
    let mut tree = default_tree();
    tree.fit(&df, None);

    let probas = tree.predict_proba(&df);
    assert_eq!(probas.len(), df.height());

    for (row_idx, dist) in probas.iter().enumerate() {
        assert_eq!(dist.cells.len(), dist.masses().len());

        let mass_sum: f64 = dist.masses().iter().sum();
        assert!(mass_sum.is_finite() && mass_sum > 0.0);

        let idx = UInt32Chunked::from_slice(PlSmallStr::from_static("idx"), &[row_idx as u32]);
        let row = df.take(&idx).expect("failed to take row dataframe");

        let pdf = dist.pdf(&row);
        assert_eq!(pdf.len(), 1);

        let masses = dist.mass(&row);
        assert_eq!(masses.len(), 1);

        // Recompute pdf and mass manually using the cell matching logic to ensure
        // predict_proba wiring is consistent with PiecewiseConstantDistribution
        // internals.
        let subset = UInt32Chunked::new(PlSmallStr::from_static("idx"), 0u32..1u32);
        let mut manual_pdf = 0.0;
        let mut manual_mass = 0.0;
        for (cell, &weight) in dist.cells.iter().zip(dist.masses()) {
            let matches = cell
                .match_dataframe_return_mask(&row, &subset, TargetBehaviour::Only)
                .expect("mask for row");
            if matches.into_iter().next().unwrap_or(false) {
                manual_pdf += weight / (cell.target_volume() * mass_sum);
                manual_mass += weight / mass_sum;
            }
        }

        assert!((pdf[0] - manual_pdf).abs() < 1e-12);
        assert!((masses[0] - manual_mass).abs() < 1e-12);
    }
}

#[test]
fn tree_predict_proba_distributions_differ_between_samples() {
    let df = build_df();
    let mut tree = default_tree();
    tree.fit(&df, None);

    let probas = tree.predict_proba(&df);
    assert!(probas.len() >= 2, "need at least two samples to compare");

    let first = &probas[0];
    let second = &probas[1];

    // If the number of cells differs, distributions are already distinct.
    if first.cells.len() != second.cells.len() {
        return;
    }

    // Otherwise, compare masses; require at least one differing weight
    let mass_diff: f64 = first
        .masses()
        .iter()
        .zip(second.masses())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(mass_diff > 1e-9, "distributions have identical masses");
}

#[test]
fn tree_root_becomes_leaf_when_no_valid_split_due_to_gain() {
    let df = build_classification_df_no_split();

    // Force the root to have no valid split by setting an extremely high min_split_gain.
    let mut tree = Tree::new(
        3,             // max_iter
        1,             // min_samples_split
        1,             // min_samples_leaf_y
        1,             // min_samples_leaf_x
        1,             // min_samples_leaf
        5,             // max_depth
        0.0,           // min_target_volume
        1e6,           // min_split_gain (unreachable)
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        0.0,           // boundaries_expansion_factor
        None,          // max_samples
        None,          // max_features
        0,
        None,      // feature_split_fraction
        Some(0),   // seed
    );

    tree.fit(&df, None);

    // The build should succeed with a single root leaf and no split history.
    assert_eq!(tree.get_build_status(), &TreeBuilderStatus::SUCCESS);
    assert_eq!(tree.get_leaves().len(), 1);
    assert_eq!(tree.get_leaves()[0], 0);
    assert!(tree.get_split_history().is_empty());

    // Predictions should still be well-formed for every row.
    let probas = tree.predict_proba(&df);
    assert_eq!(probas.len(), df.height());
    for dist in probas.iter() {
        assert_eq!(dist.cells.len(), 1);
        let mean_map = dist.mean();
        assert!(mean_map.contains_key("target_class"));
    }

    let preds = tree.predict_mean(&df);
    assert_eq!(preds.height(), df.height());
    assert_eq!(preds.get_column_names(), vec!["target_class"]);
    assert_eq!(preds.column("target_class").unwrap().null_count(), 0);
}
