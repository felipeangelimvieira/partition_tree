use estimator_api::api::Estimator;
use partition_tree::cell::Cell;
use partition_tree::conf::TARGET_PREFIX;
use partition_tree::density::{ConstantDensity, ConstantF64};
use partition_tree::estimator::PartitionTree;
use partition_tree::onedimpartition::OneDimPartition;
use partition_tree::predict::probability::PiecewiseConstantDistribution;
use partition_tree::rules::BelongsTo;
use partition_tree::rules::ContinuousInterval;
use polars::prelude::FrozenCategories;
use polars::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;

fn make_cell(low: f64, high: f64) -> Cell {
    let mut cell = Cell::new();
    let rule = ContinuousInterval::new(low, high, true, true, None, false);
    let density = ConstantF64::new(1.0);
    cell.insert(
        format!("{TARGET_PREFIX}_0"),
        OneDimPartition::new(rule, density),
    );
    cell
}

#[test]
fn pdf_with_intervals_returns_values_and_bounds() {
    // Two cells with different target spans
    let cells = vec![make_cell(0.0, 1.0), make_cell(1.0, 3.0)];
    let masses = vec![0.2, 0.8];

    let dist = PiecewiseConstantDistribution::new(cells, masses);
    let parts = dist.pdf_with_intervals();

    assert_eq!(parts.len(), 2);
    let (pdf1, interval1) = &parts[0];
    let (pdf2, interval2) = &parts[1];

    // Check bounds (low, high, lower_closed, upper_closed)
    assert_eq!(interval1.0, 0.0);
    assert_eq!(interval1.1, 1.0);
    assert!(interval1.2); // lower_closed
    assert!(interval1.3); // upper_closed

    assert_eq!(interval2.0, 1.0);
    assert_eq!(interval2.1, 3.0);
    assert!(interval2.2); // lower_closed
    assert!(interval2.3); // upper_closed

    // pdf = mass / (volume * total_mass); volumes: 1.0 and 2.0
    assert!((*pdf1 - 0.2).abs() < 1e-12);
    assert!((*pdf2 - 0.4).abs() < 1e-12);
}

#[test]
fn pdf_with_intervals_handles_zero_total_mass_uniformly() {
    let cells = vec![make_cell(0.0, 2.0), make_cell(2.0, 4.0)];
    let masses = vec![0.0, 0.0];

    let dist = PiecewiseConstantDistribution::new(cells, masses);
    let parts = dist.pdf_with_intervals();

    // total mass zero => uniform across cells: 1 / (volume * n_cells)
    assert_eq!(parts.len(), 2);
    let (pdf1, interval1) = &parts[0];
    let (pdf2, interval2) = &parts[1];

    // Check bounds (low, high, lower_closed, upper_closed)
    assert_eq!(interval1.0, 0.0);
    assert_eq!(interval1.1, 2.0);
    assert_eq!(interval2.0, 2.0);
    assert_eq!(interval2.1, 4.0);

    assert!((*pdf1 - 1.0 / (2.0 * 2.0)).abs() < 1e-12);
    assert!((*pdf2 - 1.0 / (2.0 * 2.0)).abs() < 1e-12);
}

#[test]
fn pdf_and_mass_match_expected_on_dataframe() {
    let cells = vec![make_cell(0.0, 1.0), make_cell(1.0, 3.0)];
    let masses = vec![0.25, 0.75];

    let dist = PiecewiseConstantDistribution::new(cells, masses);

    let target_values = Series::new(PlSmallStr::from_static("target_0"), &[0.5_f64, 1.5, 2.5]);
    let df = DataFrame::new(vec![target_values.into()]).expect("dataframe builds");

    let pdf = dist.pdf(&df);
    assert_eq!(pdf.len(), 3);
    assert!((pdf[0] - 0.25).abs() < 1e-12); // 0.25 / 1.0
    assert!((pdf[1] - 0.375).abs() < 1e-12); // 0.75 / 2.0
    assert!((pdf[2] - 0.375).abs() < 1e-12);

    let masses = dist.mass(&df);
    assert_eq!(masses.len(), 3);
    assert!((masses[0] - 0.25).abs() < 1e-12);
    assert!((masses[1] - 0.75).abs() < 1e-12);
    assert!((masses[2] - 0.75).abs() < 1e-12);
}

fn make_categorical_cell(
    active: &[usize],
    domain: Arc<Vec<usize>>,
    domain_names: Arc<Vec<String>>,
) -> Cell {
    let mut cell = Cell::new();
    let rule = BelongsTo::new(
        HashSet::from_iter(active.iter().copied()),
        Arc::clone(&domain),
        Arc::clone(&domain_names),
        false,
    );
    let density = ConstantDensity::<usize>::new(1.0);
    cell.insert("cat", OneDimPartition::new(rule, density));
    cell
}

#[test]
fn masses_with_categories_returns_string_labels() {
    let domain = Arc::new(vec![0usize, 1usize, 2usize]);
    let domain_names = Arc::new(vec![
        "red".to_string(),
        "blue".to_string(),
        "green".to_string(),
    ]);

    let cell1 = make_categorical_cell(&[0, 2], Arc::clone(&domain), Arc::clone(&domain_names));
    let cell2 = make_categorical_cell(&[1], Arc::clone(&domain), Arc::clone(&domain_names));
    let masses = vec![0.3_f64, 0.7_f64];
    let dist = PiecewiseConstantDistribution::new(vec![cell1, cell2], masses);

    let result = dist.masses_with_categories();
    assert_eq!(result.len(), 2);

    let (m1, cats1) = &result[0];
    let (m2, cats2) = &result[1];

    assert!((*m1 - 0.3).abs() < 1e-12);
    assert!((*m2 - 0.7).abs() < 1e-12);

    let names1 = cats1.get("cat").expect("cat partition present in cell1");
    let names2 = cats2.get("cat").expect("cat partition present in cell2");

    assert_eq!(names1, &vec!["red".to_string(), "green".to_string()]);
    assert_eq!(names2, &vec!["blue".to_string()]);
}

#[test]
fn predict_categorical_masses_returns_per_row_labels() {
    // Simple dataset with one numeric feature and one categorical target
    let x = Series::new(PlSmallStr::from_static("x"), &[0.0_f64, 1.0, 2.0]);
    let color_utf8 = Series::new(PlSmallStr::from_static("color"), &["red", "blue", "red"]);
    let color_cats = FrozenCategories::new(["red", "blue"]).unwrap();
    let color = color_utf8
        .cast(&DataType::from_frozen_categories(color_cats))
        .expect("cast to enum");

    let x_df = DataFrame::new(vec![x.into()]).expect("x df");
    let y_df = DataFrame::new(vec![color.into()]).expect("y df");

    let tree = PartitionTree::default()
        .fit(&x_df, &y_df, None)
        .expect("fit succeeds");

    let per_row = tree
        .predict_categorical_masses(&x_df)
        .expect("predict categorical masses");

    assert_eq!(per_row.len(), x_df.height());

    for (row_idx, row_masses) in per_row.iter().enumerate() {
        eprintln!("Row {}: {:?}", row_idx, row_masses);
        assert!(!row_masses.is_empty());
        let total_mass: f64 = row_masses.iter().map(|(m, _)| *m).sum();
        eprintln!("Row {} total_mass: {}", row_idx, total_mass);
        assert!(
            (total_mass - 1.0).abs() < 1e-12,
            "Row {} total mass {} != 1.0",
            row_idx,
            total_mass
        );

        for (mass, cats) in row_masses {
            assert!(*mass > 0.0);
            let labels = cats
                .get("target_color")
                .expect("target categorical labels present");
            assert!(!labels.is_empty());
            for lbl in labels {
                assert!(lbl == "red" || lbl == "blue");
            }
        }
    }
}
