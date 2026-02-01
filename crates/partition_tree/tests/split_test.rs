use partition_tree::node::Node;
use partition_tree::split::{
    SplitRestrictions, SplitResult, find_best_split_column, search_split_column,
};
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn df_xy_simple() -> DataFrame {
    // Build a small DataFrame with one numeric feature `x`, a categorical `c` (Enum), and a target `target_y`
    let x = Series::new(PlSmallStr::from_static("x"), &[1.0_f64, 2.0, 3.0, 4.0]);

    let c_utf8 = Series::new(PlSmallStr::from_static("c"), &["a", "b", "a", "b"]);
    let cats = FrozenCategories::new(["a", "b"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let c_enum = c_utf8.cast(&enum_dt).unwrap();

    let y = Series::new(
        PlSmallStr::from_static("target_y"),
        &[10.0_f64, 20.0, 30.0, 40.0],
    );

    DataFrame::new(vec![x.into(), c_enum.into(), y.into()]).unwrap()
}

fn root_node_for(df: &DataFrame) -> Node {
    let mut rng = StdRng::seed_from_u64(42);
    Node::default_from_dataframe(&df, 0.0, None, &mut rng)
}

fn uniform_sample_weights(df: &DataFrame) -> Float64Chunked {
    Float64Chunked::full(PlSmallStr::from_static("sample_weights"), 1.0, df.height())
}

#[test]
fn split_continuous_column_basic() {
    let df = df_xy_simple();
    let node = root_node_for(&df);
    let sample_weights = uniform_sample_weights(&df);

    let restrictions = SplitRestrictions {
        min_samples_split: 2,
        min_samples_leaf: 0,
        min_samples_leaf_x: 0,
        min_samples_leaf_y: 0,
        max_depth: 10,
        min_target_volume: 0.0,
        min_split_gain: f64::NEG_INFINITY,
        total_target_volume: 1.0,
        min_density_value: 0.0,
        max_density_value: f64::INFINITY,
        max_measure_value: f64::INFINITY,
        dataset_size: df.height() as f64,
    };

    let res = search_split_column(&node, &df, "x", &restrictions, &sample_weights);
    match res {
        SplitResult::ContinuousSplit(value, gain, _) => {
            assert!(
                value > 1.0 && value < 4.0,
                "split value in range, got {value}"
            );
            assert!(gain.is_finite(), "gain should be finite, got {gain}");
        }
        other => panic!("expected continuous split, got {other:?}"),
    }
}

#[test]
fn split_categorical_enum_basic() {
    let df = df_xy_simple();
    let node = root_node_for(&df);
    let sample_weights = uniform_sample_weights(&df);

    let restrictions = SplitRestrictions {
        min_samples_split: 2,
        min_samples_leaf: 1,
        min_samples_leaf_x: 1,
        min_samples_leaf_y: 1,
        max_depth: 10,
        min_target_volume: 0.0,
        min_split_gain: f64::NEG_INFINITY,
        total_target_volume: 1.0,
        min_density_value: 0.0,
        max_density_value: f64::INFINITY,
        max_measure_value: f64::INFINITY,
        dataset_size: df.height() as f64,
    };

    let res = search_split_column(&node, &df, "c", &restrictions, &sample_weights);
    match res {
        SplitResult::CategoricalSplit(_, gain, _) => {
            assert!(gain.is_finite(), "gain should be finite");
        }
        other => panic!("expected categorical split, got {other:?}"),
    }
}

#[test]
fn best_split_picks_max_gain() {
    let df = df_xy_simple();
    let node = root_node_for(&df);
    let sample_weights = uniform_sample_weights(&df);

    let restrictions = SplitRestrictions {
        min_samples_split: 2,
        min_samples_leaf: 1,
        min_samples_leaf_x: 1,
        min_samples_leaf_y: 1,
        max_depth: 10,
        min_target_volume: 0.0,
        min_split_gain: f64::NEG_INFINITY,
        total_target_volume: 1.0,
        min_density_value: 0.0,
        max_density_value: f64::INFINITY,
        max_measure_value: f64::INFINITY,
        dataset_size: df.height() as f64,
    };

    let mut rng = StdRng::seed_from_u64(42);
    let (col, res) = find_best_split_column(
        &node,
        &df,
        &restrictions,
        None,
        &mut rng,
        &sample_weights,
        false,
    );
    assert!(!col.is_empty());
    assert!(matches!(
        res,
        SplitResult::ContinuousSplit(..) | SplitResult::CategoricalSplit(..)
    ));
}

#[test]
fn split_categorical_enum_with_nulls_none_side_decision() {
    // Build data with nulls in categorical
    let c_utf8 = Series::new(
        PlSmallStr::from_static("c"),
        &[Some("a"), None, Some("b"), Some("a"), None, Some("b")],
    );
    let cats = FrozenCategories::new(["a", "b"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let c_enum = c_utf8.cast(&enum_dt).unwrap();

    let x = Series::new(
        PlSmallStr::from_static("x"),
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let y = Series::new(
        PlSmallStr::from_static("target_y"),
        &[10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0],
    );
    let df = DataFrame::new(vec![x.into(), c_enum.into(), y.into()]).unwrap();

    let node = root_node_for(&df);
    let sample_weights = uniform_sample_weights(&df);
    let restrictions = SplitRestrictions {
        min_samples_split: 2,
        min_samples_leaf: 1,
        min_samples_leaf_x: 1,
        min_samples_leaf_y: 1,
        max_depth: 10,
        min_target_volume: 0.0,
        min_split_gain: f64::NEG_INFINITY,
        total_target_volume: 1.0,
        min_density_value: 0.0,
        max_density_value: f64::INFINITY,
        max_measure_value: f64::INFINITY,
        dataset_size: df.height() as f64,
    };

    let res = search_split_column(&node, &df, "c", &restrictions, &sample_weights);
    match res {
        SplitResult::CategoricalSplit(_, gain, _none_to_left) => {
            assert!(gain.is_finite());
        }
        other => panic!("expected categorical split, got {other:?}"),
    }
}

#[test]
fn point_valid_returns_reason_when_invalid() {
    let restrictions = SplitRestrictions {
        min_samples_split: 5,
        ..SplitRestrictions::default()
    };

    let (is_valid, reason) = restrictions.is_point_valid(
        &1.0,  // xy
        &1.0,  // x
        &1.0,  // y
        &4.0,  // total_xy
        &4.0,  // total_x
        &4.0,  // total_y
        &1.0,  // target_volume
        &1.0,  // target_volume_left
        &1.0,  // target_volume_right
        &-1.0, // gain
    );

    assert!(!is_valid);
    assert_eq!(reason, "gain (-1) < min_split_gain (0)");
}

#[test]
fn point_valid_returns_ok_when_valid() {
    let restrictions = SplitRestrictions {
        min_samples_split: 2,
        min_samples_leaf: 1,
        min_samples_leaf_x: 1,
        min_samples_leaf_y: 1,
        min_target_volume: 0.0,
        min_split_gain: 0.1,
        ..SplitRestrictions::default()
    };

    let (is_valid, reason) = restrictions.is_point_valid(
        &2.0, &2.0, &2.0, &10.0, &10.0, &10.0, &1.0, &1.0, &1.0, &0.5,
    );

    assert!(is_valid);
    assert_eq!(reason, "ok");
}
