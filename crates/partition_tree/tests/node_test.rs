use polars::prelude::*;

use partition_tree::density::{ConstantDensity, ConstantF64, ConstantU32};
use partition_tree::node::*;
use partition_tree::rules::*;
use rand::rngs::StdRng;
use rand::{SeedableRng, rng};

fn chunk_to_vec_u32(ca: &UInt32Chunked) -> Vec<u32> {
    ca.into_iter().flatten().collect::<Vec<u32>>()
}

#[test]
fn default_from_dataframe_builds_expected_partitions_and_indices() {
    // Float non-target, Float target_*, and Enum categorical
    let x = Series::new(
        PlSmallStr::from_static("x"),
        &[Some(-5.0_f64), None, Some(1.5), Some(10.0), Some(0.0)],
    );

    let target_y = Series::new(
        PlSmallStr::from_static("target_y"),
        &[Some(0.0_f64), Some(2.0), None, Some(10.0), Some(4.0)],
    );

    // Utf8 -> Enum with categories [A, B, C]
    let s = Series::new(
        PlSmallStr::from_static("c"),
        &[Some("A"), Some("B"), None, Some("C"), Some("A")],
    );
    let cats = FrozenCategories::new(["A", "B", "C"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let c_enum = s.cast(&enum_dt).unwrap();

    let df = DataFrame::new(vec![
        x.clone().into(),
        target_y.clone().into(),
        c_enum.into(),
    ])
    .unwrap();

    let mut rng = StdRng::seed_from_u64(42);
    let node = Node::default_from_dataframe(&df, 0.0, None, &mut rng);

    // Root properties and indices
    assert!(node.is_root());
    let expected_idx: Vec<u32> = (0u32..df.height() as u32).collect();
    assert_eq!(chunk_to_vec_u32(&node.indices_xy), expected_idx);
    assert_eq!(chunk_to_vec_u32(&node.indices_x), expected_idx);
    assert_eq!(chunk_to_vec_u32(&node.indices_y), expected_idx);

    // Partitions present for all 3 columns
    assert_eq!(node.cell.partitions.len(), 3);
    assert!(node.cell.partitions.contains_key("x"));
    assert!(node.cell.partitions.contains_key("target_y"));
    assert!(node.cell.partitions.contains_key("c"));

    // Non-target float column uses ContinuousInterval with infinite bounds
    let p_x = node.cell.partitions.get("x").unwrap();
    let rule_x = p_x
        .rule_any()
        .downcast_ref::<ContinuousInterval>()
        .expect("x rule is ContinuousInterval");
    assert!(rule_x.low.is_infinite() && rule_x.low.is_sign_negative());
    assert!(rule_x.high.is_infinite() && rule_x.high.is_sign_positive());
    assert!(rule_x.lower_closed && rule_x.upper_closed);
    assert!(rule_x.accept_none);

    let den_x = p_x
        .density_any()
        .downcast_ref::<ConstantF64>()
        .expect("x density is ConstantF64");
    // 1.0 / (inf) == 0.0
    assert_eq!(den_x.c, 0.0);

    // Target float column uses min/max bounds from data
    let p_ty = node.cell.partitions.get("target_y").unwrap();
    let rule_ty = p_ty
        .rule_any()
        .downcast_ref::<ContinuousInterval>()
        .expect("target_y rule is ContinuousInterval");
    let exp_low = 0.0_f64; // min over non-null values
    let exp_high = 10.0_f64; // max over non-null values
    assert!((rule_ty.low - exp_low).abs() < f64::EPSILON);
    assert!((rule_ty.high - exp_high).abs() < f64::EPSILON);
    assert!(rule_ty.lower_closed && rule_ty.upper_closed);
    assert!(rule_ty.accept_none);

    let den_ty = p_ty
        .density_any()
        .downcast_ref::<ConstantF64>()
        .expect("target_y density is ConstantF64");
    let expected_c = 1.0 / (exp_high - exp_low);
    assert!((den_ty.c - expected_c).abs() < f64::EPSILON);

    // Enum column becomes BelongsToU32 over present codes, None accepted
    let p_c = node.cell.partitions.get("c").unwrap();
    let rule_c = p_c
        .rule_any()
        .downcast_ref::<BelongsToU32>()
        .expect("c rule is BelongsToU32");
    // Expect codes {0,1,2} for categories [A,B,C]
    let expected_codes: Vec<u32> = [0_u32, 1_u32, 2_u32].into_iter().collect();
    assert_eq!(rule_c.domain.len(), expected_codes.len());
    assert_eq!(rule_c.values.len(), expected_codes.len());
    assert!(expected_codes.iter().all(|v| rule_c.domain.contains(v)));
    assert!(expected_codes.iter().all(|v| rule_c.values.contains(v)));
    assert!(rule_c.accept_none);

    let den_c = p_c
        .density_any()
        .downcast_ref::<ConstantU32>()
        .expect("c density is ConstantU32");
    let expected_c_den = 1.0 / rule_c.volume();
    assert!((den_c.c - expected_c_den).abs() < f64::EPSILON);
}

#[test]
fn default_from_dataframe_handles_string_columns() {
    let s = Series::new(
        PlSmallStr::from_static("s"),
        &[Some("red"), None, Some("blue"), Some("red"), Some("green")],
    );

    let df = DataFrame::new(vec![s.into()]).unwrap();

    let mut rng = StdRng::seed_from_u64(42);
    let node = Node::default_from_dataframe(&df, 0.0, None, &mut rng);

    let partition = node
        .cell
        .partitions
        .get("s")
        .expect("string column partition present");

    let rule = partition
        .rule_any()
        .downcast_ref::<BelongsToString>()
        .expect("string column uses BelongsToString rule");

    assert_eq!(
        rule.domain_as_vec(),
        vec!["red".to_string(), "blue".to_string(), "green".to_string()]
    );
    assert_eq!(rule.values.len(), 3);
    let mut actual_values: Vec<String> = rule.values.iter().cloned().collect();
    actual_values.sort();
    let mut expected_values = vec!["blue".to_string(), "green".to_string(), "red".to_string()];
    expected_values.sort();
    assert_eq!(actual_values, expected_values);
    assert!(rule.accept_none);

    let density = partition
        .density_any()
        .downcast_ref::<ConstantDensity<String>>()
        .expect("string column uses ConstantDensity<String>");
    assert!((density.c - (1.0 / rule.volume())).abs() < f64::EPSILON);
}
