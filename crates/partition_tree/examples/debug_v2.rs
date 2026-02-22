use partition_tree::v2::dataset_view::{DatasetView, PolarsDatasetView, ColumnView};
use partition_tree::v2::cell::Cell;
use partition_tree::v2::node::Node;
use partition_tree::v2::split_result::{SplitKind, SplitRestrictions};
use partition_tree::v2::loss::ConditionalLogLoss;
use partition_tree::v2::column_split::{CategoricalColumnSplitSearcher, ContinuousColumnSplitSearcher, ColumnSplitSearcher};
use partition_tree::v2::dtype_plugin::{DTypeRegistry, DTypePlugin};
use polars::prelude::*;

fn main() {
    let df = DataFrame::new(vec![
        Column::new("x1".into(), &[1.0_f64, 1.0, 2.0, 2.0]),
        Column::new("x2".into(), &[1.0_f64, 2.0, 2.0, 1.0]),
    ]).unwrap();

    let target_strs = Series::new(PlSmallStr::from_static("target_a"), &[Some("A"), Some("A"), Some("B"), Some("B")]);
    let cats = FrozenCategories::new(["A", "B"]).unwrap();
    let enum_dt = DataType::from_frozen_categories(cats);
    let target_enum = target_strs.cast(&enum_dt).unwrap();
    let df = df.hstack(&[target_enum.into()]).unwrap();

    println!("DataFrame: {:?}", df);

    let dataset = PolarsDatasetView::new(&df);
    println!("n_rows={} n_cols={}", dataset.n_rows(), dataset.n_columns());

    for col in dataset.columns() {
        println!("  col='{}' dtype={:?} is_target={}", col.name(), col.logical_dtype(), col.is_target());
        for i in 0..4 {
            println!("    [{}] f64={:?} cat={:?} null={}", i, col.get_f64(i), col.get_cat(i), col.is_null(i));
        }
    }

    let registry = DTypeRegistry::default();
    let mut cell = Cell::new();
    for col in dataset.columns() {
        let dtype = col.logical_dtype();
        let plugin = registry.get_or_panic(dtype);
        let rule = plugin.default_rule(col, 0.0);
        println!("Default rule for '{}': {:?}", col.name(), rule);
        cell.set_rule(col.name(), rule);
    }

    println!("\nCell volume={:.6} target_vol={:.6}", cell.volume(), cell.target_volume());

    let node = Node::root(&dataset, cell);
    println!("Root: w_xy={} w_x={} w_y={}", node.w_xy, node.w_x, node.w_y);

    for (col_name, indices) in &node.sorted.sorted_xy {
        println!("sorted_xy[{}] = {:?}", col_name, indices);
    }

    let loss = ConditionalLogLoss::new(4.0);
    let restrictions = SplitRestrictions {
        min_samples_xy: 0.0,
        min_samples_x: 0.0,
        min_samples_y: 0.0,
        min_gain: 0.001,
        min_volume: 0.0,
        max_depth: 5,
        min_samples_split: 2.0,
    };

    let target_col = dataset.column("target_a").unwrap();
    println!("\nSearching categorical split on target_a (YSplit)...");
    let cat_searcher = CategoricalColumnSplitSearcher;
    let result = cat_searcher.search(&node, &node.cell, target_col, SplitKind::YSplit, &dataset, &loss, &restrictions);
    match &result {
        Some(s) => println!("Cat split: gain={:.6} col={} op={:?}", s.gain, s.col_name, s.op),
        None => println!("Cat split: NONE"),
    }

    let x1_col = dataset.column("x1").unwrap();
    println!("\nSearching continuous split on x1 (XSplit)...");
    let cont_searcher = ContinuousColumnSplitSearcher;
    let result2 = cont_searcher.search(&node, &node.cell, x1_col, SplitKind::XSplit, &dataset, &loss, &restrictions);
    match &result2 {
        Some(s) => println!("Cont split: gain={:.6} col={} op={:?}", s.gain, s.col_name, s.op),
        None => println!("Cont split: NONE"),
    }
}
