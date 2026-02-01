use polars::prelude::*;
// Bring extension traits into scope
use partition_tree::dataframe::PTreeDataFrameExt;

fn uniform_sample_weights(df: &DataFrame) -> Float64Chunked {
    Float64Chunked::full(PlSmallStr::from_static("sample_weights"), 1.0, df.height())
}

#[test]
fn data_frame_sorted_numeric_values() {
    let df = df!("a" => &[Some(3.0_f64), Some(1.0), Some(2.0)]).unwrap();
    let idx = UInt32Chunked::from_slice(PlSmallStr::from_static("idx"), &[0_u32, 1, 2]);
    let sample_weights = uniform_sample_weights(&df);

    let res = df
        .sorted_numeric_values::<Float64Type>("a", &idx, true, &sample_weights)
        .expect("df numeric sort");

    assert_eq!(res.none_count, 0);
    assert_eq!(res.sorted_values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn column_and_target_names() {
    let df = df!(
        "target_y" => &[1.0_f64],
        "x" => &[2.0_f64],
        "target_score" => &[3.0_f64]
    )
    .unwrap();

    let mut names = df.column_names_vec();
    names.sort();
    assert_eq!(names, vec!["target_score", "target_y", "x"]);

    let mut targets = df.target_column_names();
    targets.sort();
    assert_eq!(targets, vec!["target_score", "target_y"]);
}

#[test]
fn get_row_as_strings_basic() {
    let df = df!(
        "a" => &[Some(1.1_f64), None],
        "b" => &[Some(2_i32), Some(3_i32)]
    )
    .unwrap();

    let row0 = df.get_row_as_strings(0).expect("row 0");
    assert_eq!(row0.get("a").unwrap(), "1.1");
    assert_eq!(row0.get("b").unwrap(), "2");

    let row1 = df.get_row_as_strings(1).expect("row 1");
    assert_eq!(row1.get("a").unwrap(), "None");
    assert_eq!(row1.get("b").unwrap(), "3");

    assert!(df.get_row_as_strings(2).is_none());
}

#[test]
#[should_panic]
fn sorted_numeric_values_on_non_numeric_panics() {
    // Utf8 column should panic
    let df = df!("s" => &[Some("a"), None, Some("b")]).unwrap();
    let idx = UInt32Chunked::from_slice(PlSmallStr::from_static("idx"), &[0_u32, 1, 2]);
    let sample_weights = uniform_sample_weights(&df);
    let _ = df
        .sorted_numeric_values::<Float64Type>("s", &idx, true, &sample_weights)
        .unwrap();
}

#[test]
fn target_prefix_constant_present() {
    assert!(partition_tree::conf::TARGET_PREFIX.starts_with("target"));
}

#[test]
#[should_panic]
fn count_categories_on_string_column() {
    // Build a string column with some nulls
    let df = df!(
        "s" => &[Some("a"), None, Some("b"), Some("a"), None, Some("b"), Some("c")]
    )
    .unwrap();
    let sample_weights = uniform_sample_weights(&df);

    // All indices
    let idx = UInt32Chunked::from_slice(PlSmallStr::from_static("idx"), &[0_u32, 1, 2, 3, 4, 5, 6]);

    // Use Enum generic parameter, the implementation handles String as well
    let _ = df.count_categories("s", &idx, &sample_weights).unwrap();
}

#[test]
fn count_categories_on_enum() {
    // Build a Utf8 Series first
    let s = Series::new(
        PlSmallStr::from_static("s"),
        &[
            Some("a"),
            None,
            Some("b"),
            Some("a"),
            None,
            Some("b"),
            Some("c"),
        ],
    );

    // Define allowed categories and build Enum dtype
    let cats = FrozenCategories::new(["a", "b", "c"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);

    // Cast the Utf8 series to Enum
    let s_enum = s.cast(&enum_dt).unwrap();

    // Build a DataFrame with that Enum column
    let df = DataFrame::new(vec![s_enum.into()]).unwrap();
    let sample_weights = uniform_sample_weights(&df);

    // All indices
    let idx = UInt32Chunked::from_slice("idx".into(), &[0_u32, 1, 2, 3, 4, 5, 6]);

    // Now call your extension method
    let out = df.count_categories("s", &idx, &sample_weights).unwrap();

    // You can assert on counts, unique categories, etc.
    println!("{:?}", out.unique_categories);
    println!("{:?}", out.counts);
    assert_eq!(out.none_count, 2);
}
