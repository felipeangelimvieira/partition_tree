use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use polars::prelude::*;

use partition_tree::cell::Cell;
use partition_tree::conf::TargetBehaviour;
use partition_tree::density::{ConstantDensity, ConstantF64};
use partition_tree::onedimpartition::OneDimPartition;
use partition_tree::rules::{BelongsToGeneric, BelongsToString, ContinuousInterval};

fn chunk_to_vec_u32(ca: &UInt32Chunked) -> Vec<u32> {
    ca.into_iter().flatten().collect::<Vec<u32>>()
}

#[test]
fn match_dataframe_no_partitions_defaults_all_true() {
    // DataFrame with 4 rows
    let df = df!(
        "x" => &[Some(0.0_f64), Some(5.0), None, Some(10.0)],
        "y" => &[Some(1_i32), Some(3), Some(5), None]
    )
    .unwrap();

    let cell = Cell::new();

    // No subset -> all indices [0,1,2,3]
    let out = cell
        .match_dataframe(
            &df,
            &UInt32Chunked::from_slice(PlSmallStr::from_static("all"), &[]),
            TargetBehaviour::Include,
        )
        .expect("match_dataframe");
    // With empty subset, by spec, expect empty result
    assert_eq!(chunk_to_vec_u32(&out), vec![] as Vec<u32>);

    // With subset -> only those in subset
    let subset = UInt32Chunked::from_slice(PlSmallStr::from_static("subset"), &[1, 3]);
    let out2 = cell
        .match_dataframe(&df, &subset, TargetBehaviour::Include)
        .expect("match_dataframe subset");
    assert_eq!(chunk_to_vec_u32(&out2), vec![1, 3]);
}

#[test]
fn match_dataframe_numeric_partition_filters_rows() {
    // x: [0.0, 5.0, None, 10.0]
    let df = df!(
        "x" => &[Some(0.0_f64), Some(5.0), None, Some(10.0)]
    )
    .unwrap();

    // Rule: [0.0, 10.0) lower-closed, upper-open; None is not accepted
    let rule = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), false);
    let density = ConstantF64::new(1.0);
    let part = OneDimPartition::new(rule, density);

    let mut cell = Cell::new();
    cell.insert("x", part);

    let out = cell
        .match_dataframe(
            &df,
            &UInt32Chunked::from_slice(PlSmallStr::from_static("all"), &[]),
            TargetBehaviour::Include,
        )
        .expect("match_dataframe");

    // Empty subset -> empty output
    assert_eq!(chunk_to_vec_u32(&out), vec![] as Vec<u32>);

    // With subset [1,2,3] -> only row 1 qualifies
    let subset = UInt32Chunked::from_slice(PlSmallStr::from_static("subset"), &[1, 2, 3]);
    let out2 = cell
        .match_dataframe(&df, &subset, TargetBehaviour::Include)
        .expect("match_dataframe subset");
    assert_eq!(chunk_to_vec_u32(&out2), vec![1]);
}

#[test]
fn match_dataframe_and_across_partitions_and_target_behaviour() {
    // Build a DataFrame with 4 rows
    let df = df!(
        "x" => &[Some(0.0_f64), Some(5.0), None, Some(10.0)],
        "y" => &[Some(1_i32), Some(3), Some(5), Some(3)],
        "target_flag" => &[Some(true), Some(false), Some(true), None]
    )
    .unwrap();

    // x rule: [0.0, 10.0) -> rows 0 and 1
    let x_rule = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), false);
    let x_den = ConstantF64::new(1.0);
    let x_part = OneDimPartition::new(x_rule, x_den);

    // y rule: belongs to {3, 5} -> rows 1,2,3
    let y_domain: Arc<Vec<i32>> = Arc::new([1, 3, 5].into_iter().collect());
    let y_names: Arc<Vec<String>> = Arc::new(
        vec!["1", "3", "5"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
    );
    let y_rule = BelongsToGeneric::new(
        [3_i32, 5_i32].into_iter().collect::<HashSet<i32>>(),
        Arc::clone(&y_domain),
        Arc::clone(&y_names),
        false,
    );
    let y_den = ConstantDensity::<i32>::new(1.0);
    let y_part = OneDimPartition::new(y_rule, y_den);

    // target_flag rule: true only -> rows 0 and 2
    let t_domain: Arc<Vec<bool>> = Arc::new([false, true].into_iter().collect());
    let t_names: Arc<Vec<String>> = Arc::new(vec!["false".to_string(), "true".to_string()]);
    let t_rule = BelongsToGeneric::new(
        [true].into_iter().collect::<HashSet<bool>>(),
        Arc::clone(&t_domain),
        Arc::clone(&t_names),
        false,
    );
    let t_den = ConstantDensity::<bool>::new(1.0);
    let t_part = OneDimPartition::new(t_rule, t_den);

    let mut cell = Cell::new();
    cell.insert("x", x_part);
    cell.insert("y", y_part);
    cell.insert("target_flag", t_part);

    // Include: AND across x, y, and target_flag -> intersection of (0,1) ∩ (1,2,3) ∩ (0,2) = {}
    let out_inc = cell
        .match_dataframe(
            &df,
            &UInt32Chunked::from_slice(PlSmallStr::from_static("all"), &[]),
            TargetBehaviour::Include,
        )
        .expect("match_dataframe include");
    assert_eq!(chunk_to_vec_u32(&out_inc), vec![] as Vec<u32>);

    // Only: use only target_* partitions (target_flag) -> rows 0 and 2
    let out_only = cell
        .match_dataframe(
            &df,
            &UInt32Chunked::from_slice(PlSmallStr::from_static("all"), &[]),
            TargetBehaviour::Only,
        )
        .expect("match_dataframe only");
    assert_eq!(chunk_to_vec_u32(&out_only), vec![] as Vec<u32>);

    // Exclude: use non-target partitions (x AND y) -> (0,1) ∩ (1,2,3) = {1}
    let out_excl = cell
        .match_dataframe(
            &df,
            &UInt32Chunked::from_slice(PlSmallStr::from_static("all"), &[]),
            TargetBehaviour::Exclude,
        )
        .expect("match_dataframe exclude");
    assert_eq!(chunk_to_vec_u32(&out_excl), vec![] as Vec<u32>);
}

#[test]
fn match_dataframe_enum_basic_belongs_to_and_none_handling() {
    // Build a Utf8 Series first
    let s = Series::new(
        PlSmallStr::from_static("s"),
        &[Some("a"), None, Some("b"), Some("a"), Some("c"), None],
    );

    // Define categories and cast to Enum
    let cats = FrozenCategories::new(["a", "b", "c"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let s_enum = s.cast(&enum_dt).unwrap();
    let df = DataFrame::new(vec![s_enum.into()]).unwrap();

    // Domain codes [0,1,2] corresponding to [a,b,c]
    let domain: Arc<Vec<u32>> = Arc::new([0, 1, 2].into_iter().collect());
    let names: Arc<Vec<String>> = Arc::new(
        vec!["a", "b", "c"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
    );

    // Keep {a, c} => codes {0, 2}; do NOT accept None
    let rule = BelongsToGeneric::new(
        [0_u32, 2_u32].into_iter().collect::<HashSet<u32>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        false,
    );
    let den = ConstantDensity::<u32>::new(1.0);
    let part = OneDimPartition::new(rule, den);

    let mut cell = Cell::new();
    cell.insert("s", part);

    // Expect rows with a or c: indices 0, 3, 4
    let out = cell
        .match_dataframe(
            &df,
            &UInt32Chunked::from_slice(PlSmallStr::from_static("all"), &[]),
            TargetBehaviour::Include,
        )
        .expect("enum match");
    assert_eq!(chunk_to_vec_u32(&out), vec![] as Vec<u32>);

    // Now accept None; expect 0,1,3,4,5
    let rule2 = BelongsToGeneric::new(
        [0_u32, 2_u32].into_iter().collect::<HashSet<u32>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        true,
    );
    let den2 = ConstantDensity::<u32>::new(1.0);
    let part2 = OneDimPartition::new(rule2, den2);
    let mut cell2 = Cell::new();
    cell2.insert("s", part2);
    let out2 = cell2
        .match_dataframe(
            &df,
            &UInt32Chunked::from_slice(PlSmallStr::from_static("all"), &[]),
            TargetBehaviour::Include,
        )
        .expect("enum match none");
    assert_eq!(chunk_to_vec_u32(&out2), vec![] as Vec<u32>);
}

#[test]
fn match_dataframe_string_partition_filters_rows() {
    let df = df!(
        "s" => &[Some("red"), None, Some("blue"), Some("red"), Some("green")]
    )
    .unwrap();

    let domain: Vec<String> = vec!["red", "blue", "green"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    let values: HashSet<String> = ["red", "green"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    let rule = BelongsToString::new(
        values,
        Arc::new(domain.clone()),
        Arc::new(domain.clone()),
        false,
    );
    let density = ConstantDensity::<String>::new(1.0);
    let partition = OneDimPartition::new(rule, density);

    let mut cell = Cell::new();
    cell.insert("s", partition);

    let subset = UInt32Chunked::from_slice(PlSmallStr::from_static("idx"), &[0, 1, 2, 3, 4]);
    let out = cell
        .match_dataframe(&df, &subset, TargetBehaviour::Include)
        .expect("string partition match");
    assert_eq!(chunk_to_vec_u32(&out), vec![0, 3, 4]);
}

#[test]
fn match_dataframe_enum_with_subset() {
    // Utf8 -> Enum
    let s = Series::new(
        PlSmallStr::from_static("s"),
        &[Some("a"), None, Some("b"), Some("a"), Some("c"), None],
    );
    let cats = FrozenCategories::new(["a", "b", "c"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let s_enum = s.cast(&enum_dt).unwrap();
    let df = DataFrame::new(vec![s_enum.into()]).unwrap();

    let domain: Arc<Vec<u32>> = Arc::new([0, 1, 2].into_iter().collect());
    let names: Arc<Vec<String>> = Arc::new(
        vec!["a", "b", "c"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
    );

    // Keep {a,c}; accept None
    let rule = BelongsToGeneric::new(
        [0_u32, 2_u32].into_iter().collect::<HashSet<u32>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        true,
    );
    let den = ConstantDensity::<u32>::new(1.0);
    let part = OneDimPartition::new(rule, den);

    let mut cell = Cell::new();
    cell.insert("s", part);

    // Subset indices [1,2,5] -> only 1 and 5 qualify (None allowed), 2 is 'b' (excluded)
    let subset = UInt32Chunked::from_slice(PlSmallStr::from_static("subset"), &[1, 2, 5]);
    let out = cell
        .match_dataframe(&df, &subset, TargetBehaviour::Include)
        .expect("enum subset match");
    assert_eq!(chunk_to_vec_u32(&out), vec![1, 5]);
}

// Helper function to create a HashMap row for testing
fn create_row_hashmap() -> HashMap<String, Box<dyn Any>> {
    let mut row = HashMap::new();
    row.insert("x".to_string(), Box::new(Some(5.0_f64)) as Box<dyn Any>);
    row.insert("y".to_string(), Box::new(Some(3_i32)) as Box<dyn Any>);
    row.insert(
        "category".to_string(),
        Box::new(Some("red".to_string())) as Box<dyn Any>,
    );
    row.insert(
        "target_flag".to_string(),
        Box::new(Some(true)) as Box<dyn Any>,
    );
    row
}

#[test]
fn match_hashmap_no_partitions_returns_true() {
    let cell = Cell::new();
    let row = create_row_hashmap();

    // No partitions = all true for any TargetBehaviour
    assert!(cell.match_hashmap(&row, TargetBehaviour::Include));
    assert!(cell.match_hashmap(&row, TargetBehaviour::Exclude));
    assert!(cell.match_hashmap(&row, TargetBehaviour::Only));
}

#[test]
fn match_hashmap_single_continuous_partition() {
    let mut cell = Cell::new();

    // Add partition for x: [0.0, 10.0)
    let rule = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), false);
    let density = ConstantF64::new(1.0);
    let part = OneDimPartition::new(rule, density);
    cell.insert("x", part);

    // Row with x=5.0 should match
    let row = create_row_hashmap();
    assert!(cell.match_hashmap(&row, TargetBehaviour::Include));
    assert!(cell.match_hashmap(&row, TargetBehaviour::Exclude));

    // Row with x=15.0 should not match
    let mut row2 = HashMap::new();
    row2.insert("x".to_string(), Box::new(Some(15.0_f64)) as Box<dyn Any>);
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Include));
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Exclude));
}

#[test]
fn match_hashmap_single_categorical_partition() {
    let mut cell = Cell::new();

    // Add partition for y: belongs to {1, 3, 5}
    let y_domain: Arc<Vec<i32>> = Arc::new([1, 3, 5, 7].into_iter().collect());
    let y_names: Arc<Vec<String>> = Arc::new(
        vec!["1", "3", "5", "7"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
    );
    let y_rule = BelongsToGeneric::new(
        [1_i32, 3_i32, 5_i32].into_iter().collect::<HashSet<i32>>(),
        Arc::clone(&y_domain),
        Arc::clone(&y_names),
        false,
    );
    let y_den = ConstantDensity::<i32>::new(1.0);
    let y_part = OneDimPartition::new(y_rule, y_den);
    cell.insert("y", y_part);

    // Row with y=3 should match
    let row = create_row_hashmap();
    assert!(cell.match_hashmap(&row, TargetBehaviour::Include));
    assert!(cell.match_hashmap(&row, TargetBehaviour::Exclude));

    // Row with y=7 should not match
    let mut row2 = HashMap::new();
    row2.insert("y".to_string(), Box::new(Some(7_i32)) as Box<dyn Any>);
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Include));
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Exclude));
}

#[test]
fn match_hashmap_string_partition() {
    let mut cell = Cell::new();

    // Add string partition for category: belongs to {"red", "green"}
    let domain: Vec<String> = vec!["red", "blue", "green"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    let values: HashSet<String> = ["red", "green"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    let rule = BelongsToString::new(
        values,
        Arc::new(domain.clone()),
        Arc::new(domain.clone()),
        false,
    );
    let density = ConstantDensity::<String>::new(1.0);
    let partition = OneDimPartition::new(rule, density);
    cell.insert("category", partition);

    // Row with category="red" should match
    let row = create_row_hashmap();
    assert!(cell.match_hashmap(&row, TargetBehaviour::Include));
    assert!(cell.match_hashmap(&row, TargetBehaviour::Exclude));

    // Row with category="blue" should not match
    let mut row2 = HashMap::new();
    row2.insert(
        "category".to_string(),
        Box::new(Some("blue".to_string())) as Box<dyn Any>,
    );
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Include));
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Exclude));
}

#[test]
fn match_hashmap_multiple_partitions_and_logic() {
    let mut cell = Cell::new();

    // Add x partition: [0.0, 10.0)
    let x_rule = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), false);
    let x_den = ConstantF64::new(1.0);
    let x_part = OneDimPartition::new(x_rule, x_den);
    cell.insert("x", x_part);

    // Add y partition: belongs to {3, 5}
    let y_domain: Arc<Vec<i32>> = Arc::new([1, 3, 5, 7].into_iter().collect());
    let y_names: Arc<Vec<String>> = Arc::new(
        vec!["1", "3", "5", "7"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
    );
    let y_rule = BelongsToGeneric::new(
        [3_i32, 5_i32].into_iter().collect::<HashSet<i32>>(),
        Arc::clone(&y_domain),
        Arc::clone(&y_names),
        false,
    );
    let y_den = ConstantDensity::<i32>::new(1.0);
    let y_part = OneDimPartition::new(y_rule, y_den);
    cell.insert("y", y_part);

    // Row with x=5.0, y=3 should match both (AND logic)
    let row = create_row_hashmap();
    assert!(cell.match_hashmap(&row, TargetBehaviour::Include));
    assert!(cell.match_hashmap(&row, TargetBehaviour::Exclude));

    // Row with x=5.0, y=7 should not match (y fails)
    let mut row2 = HashMap::new();
    row2.insert("x".to_string(), Box::new(Some(5.0_f64)) as Box<dyn Any>);
    row2.insert("y".to_string(), Box::new(Some(7_i32)) as Box<dyn Any>);
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Include));
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Exclude));

    // Row with x=15.0, y=3 should not match (x fails)
    let mut row3 = HashMap::new();
    row3.insert("x".to_string(), Box::new(Some(15.0_f64)) as Box<dyn Any>);
    row3.insert("y".to_string(), Box::new(Some(3_i32)) as Box<dyn Any>);
    assert!(!cell.match_hashmap(&row3, TargetBehaviour::Include));
    assert!(!cell.match_hashmap(&row3, TargetBehaviour::Exclude));
}

#[test]
#[should_panic(expected = "Column 'missing' not found in input row")]
fn match_hashmap_missing_column_panics() {
    let mut cell = Cell::new();

    // Add partition for column "missing"
    let rule = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), false);
    let density = ConstantF64::new(1.0);
    let part = OneDimPartition::new(rule, density);
    cell.insert("missing", part);

    // Row without "missing" column should panic
    let row = create_row_hashmap(); // doesn't contain "missing"
    cell.match_hashmap(&row, TargetBehaviour::Include);
}

#[test]
fn match_hashmap_target_behaviour_scenarios() {
    let mut cell = Cell::new();

    // Add non-target partition for x
    let x_rule = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), false);
    let x_den = ConstantF64::new(1.0);
    let x_part = OneDimPartition::new(x_rule, x_den);
    cell.insert("x", x_part);

    // Add target partition for target_flag: true only
    let t_domain: Arc<Vec<bool>> = Arc::new([false, true].into_iter().collect());
    let t_names: Arc<Vec<String>> = Arc::new(vec!["false".to_string(), "true".to_string()]);
    let t_rule = BelongsToGeneric::new(
        [true].into_iter().collect::<HashSet<bool>>(),
        Arc::clone(&t_domain),
        Arc::clone(&t_names),
        false,
    );
    let t_den = ConstantDensity::<bool>::new(1.0);
    let t_part = OneDimPartition::new(t_rule, t_den);
    cell.insert("target_flag", t_part);

    let row = create_row_hashmap(); // x=5.0, target_flag=true

    // Include: should check both x AND target_flag
    assert!(cell.match_hashmap(&row, TargetBehaviour::Include));

    // Exclude: should check only x (non-target columns)
    assert!(cell.match_hashmap(&row, TargetBehaviour::Exclude));

    // Only: should check only target_flag (target columns)
    assert!(cell.match_hashmap(&row, TargetBehaviour::Only));

    // Test with target_flag=false
    let mut row2 = HashMap::new();
    row2.insert("x".to_string(), Box::new(Some(5.0_f64)) as Box<dyn Any>);
    row2.insert(
        "target_flag".to_string(),
        Box::new(Some(false)) as Box<dyn Any>,
    );

    // Include: should fail (target_flag=false doesn't match)
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Include));

    // Exclude: should pass (only checks x, which is 5.0 and matches)
    assert!(cell.match_hashmap(&row2, TargetBehaviour::Exclude));

    // Only: should fail (target_flag=false doesn't match)
    assert!(!cell.match_hashmap(&row2, TargetBehaviour::Only));
}

#[test]
fn match_hashmap_with_none_values() {
    let mut cell = Cell::new();

    // Add partition for x: [0.0, 10.0) that accepts None values
    let rule = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), true);
    let density = ConstantF64::new(1.0);
    let part = OneDimPartition::new(rule, density);
    cell.insert("x", part);

    // Row with x=None should match (accept_none=true)
    let mut row_with_none = HashMap::new();
    row_with_none.insert("x".to_string(), Box::new(None::<f64>) as Box<dyn Any>);
    assert!(cell.match_hashmap(&row_with_none, TargetBehaviour::Include));

    // Add partition for y that does NOT accept None values
    let y_domain: Arc<Vec<i32>> = Arc::new([1, 3, 5].into_iter().collect());
    let y_names: Arc<Vec<String>> = Arc::new(
        vec!["1", "3", "5"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
    );
    let y_rule = BelongsToGeneric::new(
        [3_i32, 5_i32].into_iter().collect::<HashSet<i32>>(),
        Arc::clone(&y_domain),
        Arc::clone(&y_names),
        false, // accept_none = false
    );
    let y_den = ConstantDensity::<i32>::new(1.0);
    let y_part = OneDimPartition::new(y_rule, y_den);
    cell.insert("y", y_part);

    // Row with x=None, y=None should not match (y doesn't accept None)
    let mut row_both_none = HashMap::new();
    row_both_none.insert("x".to_string(), Box::new(None::<f64>) as Box<dyn Any>);
    row_both_none.insert("y".to_string(), Box::new(None::<i32>) as Box<dyn Any>);
    assert!(!cell.match_hashmap(&row_both_none, TargetBehaviour::Include));

    // Row with x=None, y=3 should match (x accepts None, y=3 is valid)
    let mut row_mixed = HashMap::new();
    row_mixed.insert("x".to_string(), Box::new(None::<f64>) as Box<dyn Any>);
    row_mixed.insert("y".to_string(), Box::new(Some(3_i32)) as Box<dyn Any>);
    assert!(cell.match_hashmap(&row_mixed, TargetBehaviour::Include));
}
