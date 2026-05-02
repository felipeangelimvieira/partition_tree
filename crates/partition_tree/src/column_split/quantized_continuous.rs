//! Quantized-continuous column split searcher.
//!
//! Handles numeric columns by snapping values to bins centered on
//! `resolution * i` and using integer-style split thresholds on the bin indices.
use super::{ColumnSplitSearcher, clip_candidate_positions, cumsum, split_nulls};
use crate::cell::Cell;
use crate::dataset_view::{ColumnView, DatasetView, LogicalDType};
use crate::loss::{CellStats, LossFunc};
use crate::node::Node;
use crate::split_result::{
    QuantizedContinuousSplitOp, SplitKind, SplitPoint, SplitRestrictions,
};

/// Presorted-scan split search for quantized-continuous columns.
#[derive(Debug, Clone)]
pub struct QuantizedContinuousColumnSplitSearcher;

impl ColumnSplitSearcher for QuantizedContinuousColumnSplitSearcher {
    fn search(
        &self,
        node: &Node,
        cell: &Cell,
        col: &dyn ColumnView,
        split_kind: SplitKind,
        dataset: &dyn DatasetView,
        loss: &dyn LossFunc,
        restrictions: &SplitRestrictions,
        max_candidate_split_points: Option<usize>,
        dataset_size: f64,
    ) -> Option<SplitPoint> {
        let spec = match col.logical_dtype() {
            LogicalDType::QuantizedContinuous(spec) => spec,
            dtype => panic!(
                "QuantizedContinuousColumnSplitSearcher requires a quantized column, got {dtype:?}"
            ),
        };
        let resolution = spec.resolution();
        let quantize = |value: f64| {
            crate::rules::QuantizedContinuousInterval::quantize_value(value, resolution)
                .unwrap_or_else(|_| {
                    panic!(
                        "column '{}' contains value {} that could not be quantized at resolution {}",
                        col.name(),
                        value,
                        resolution
                    )
                })
        };

        let col_name = col.name();
        let weights_xy = dataset.weights_xy();
        let weights_x = dataset.weights_x();
        let weights_y = dataset.weights_y();

        let sx = node.sorted.sorted_x.get(col_name)?;
        let sy = node.sorted.sorted_y.get(col_name)?;
        let sxy = node.sorted.sorted_xy.get(col_name)?;

        let (sx_some, wx_null) = split_nulls(sx, col, weights_x);
        let (sy_some, wy_null) = split_nulls(sy, col, weights_y);
        let (sxy_some, wxy_null) = split_nulls(sxy, col, weights_xy);

        let w_x_parent: f64 = sx_some.iter().map(|&i| weights_x[i as usize]).sum::<f64>() + wx_null;
        let w_y_parent: f64 = sy_some.iter().map(|&i| weights_y[i as usize]).sum::<f64>() + wy_null;
        let w_xy_parent: f64 = sxy_some
            .iter()
            .map(|&i| weights_xy[i as usize])
            .sum::<f64>()
            + wxy_null;

        let vol_parent = cell.target_volume();
        let parent_stats = CellStats::new(w_xy_parent, w_x_parent, w_y_parent, vol_parent);

        let candidates = match split_kind {
            SplitKind::XSplit => &sx_some,
            SplitKind::YSplit => &sy_some,
        };

        if candidates.len() < 2 {
            return None;
        }

        let valid_candidate_positions: Vec<usize> = (0..candidates.len() - 1)
            .filter(|&k| {
                let i0 = candidates[k] as usize;
                let i1 = candidates[k + 1] as usize;

                matches!((col.get_f64(i0), col.get_f64(i1)), (Some(v0), Some(v1)) if quantize(v0) != quantize(v1))
            })
            .collect();
        let candidate_positions =
            clip_candidate_positions(&valid_candidate_positions, max_candidate_split_points);

        if candidate_positions.is_empty() {
            return None;
        }

        let candidate_weights: Vec<f64> = match split_kind {
            SplitKind::XSplit => candidates.iter().map(|&i| weights_x[i as usize]).collect(),
            SplitKind::YSplit => candidates.iter().map(|&i| weights_y[i as usize]).collect(),
        };
        let prefix_candidate = cumsum(&candidate_weights);

        let sxy_weights: Vec<f64> = sxy_some.iter().map(|&i| weights_xy[i as usize]).collect();
        let prefix_wxy = cumsum(&sxy_weights);

        let mut best: Option<SplitPoint> = None;
        let mut best_gain = f64::NEG_INFINITY;
        let tie_break_start = (candidates.len() - 1) / 2;

        for &none_to_left in &[true, false] {
            let mut p_xy: usize = 0;

            for &k in &candidate_positions {
                let i0 = candidates[k] as usize;
                let i1 = candidates[k + 1] as usize;

                let v0 = match col.get_f64(i0) {
                    Some(v) => quantize(v),
                    None => continue,
                };
                let v1 = match col.get_f64(i1) {
                    Some(v) => quantize(v),
                    None => continue,
                };

                if v0 == v1 {
                    continue;
                }

                let threshold_idx = v1;
                let w_candidate_left = prefix_candidate[k];

                let (w_x_left, w_x_right, w_y_left, w_y_right) = match split_kind {
                    SplitKind::XSplit => {
                        let wxl = w_candidate_left + if none_to_left { wx_null } else { 0.0 };
                        let wxr = w_x_parent - wxl;
                        (wxl, wxr, w_y_parent, w_y_parent)
                    }
                    SplitKind::YSplit => {
                        let wyl = w_candidate_left + if none_to_left { wy_null } else { 0.0 };
                        let wyr = w_y_parent - wyl;
                        (w_x_parent, w_x_parent, wyl, wyr)
                    }
                };

                while p_xy < sxy_some.len() {
                    match col.get_f64(sxy_some[p_xy] as usize) {
                        Some(v) if quantize(v) < threshold_idx => p_xy += 1,
                        _ => break,
                    }
                }
                let w_xy_left_some = if p_xy > 0 { prefix_wxy[p_xy - 1] } else { 0.0 };
                let w_xy_left = w_xy_left_some + if none_to_left { wxy_null } else { 0.0 };
                let w_xy_right = w_xy_parent - w_xy_left;

                let quantized_op = QuantizedContinuousSplitOp {
                    threshold_idx,
                    resolution,
                    k_candidate: k,
                    p_xy,
                };
                let (vol_left, vol_right) =
                    cell.child_target_volumes(col_name, &quantized_op, none_to_left);

                let left_stats = CellStats::new(w_xy_left, w_x_left, w_y_left, vol_left);
                let right_stats = CellStats::new(w_xy_right, w_x_right, w_y_right, vol_right);

                if !restrictions.is_valid_children(
                    &left_stats,
                    &right_stats,
                    node.depth,
                    cell.target_continuous_domain_volume(),
                    cell.child_target_continuous_volumes(col_name, &quantized_op, none_to_left),
                ) {
                    continue;
                }

                let gain = loss.gain(&parent_stats, &left_stats, &right_stats, dataset_size);
                if gain < restrictions.min_gain {
                    continue;
                }

                if gain > best_gain
                    || ((gain - best_gain).abs() < f64::EPSILON && k < tie_break_start)
                {
                    best_gain = gain;
                    best = Some(SplitPoint {
                        col_name: col_name.to_string(),
                        split_kind,
                        none_to_left,
                        gain,
                        left_stats,
                        right_stats,
                        op: Box::new(quantized_op),
                    });
                }
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset_view::{LogicalDType, PolarsDatasetView};
    use crate::loss::ConditionalLogLoss;
    use crate::rule::DynRule;
    use crate::rules::QuantizedContinuousInterval;
    use polars::prelude::*;

    fn make_test_dataset() -> PolarsDatasetView {
        let df = DataFrame::new(vec![
            Column::new("x1".into(), &[0.0_f64, 0.5, 1.0, 1.5, 2.0]),
            Column::new("target__y1".into(), &[10.0_f64, 10.5, 20.0, 20.5, 21.0]),
        ])
        .unwrap();

        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "target__y1".to_string(),
            LogicalDType::quantized_continuous(0.5).unwrap(),
        );

        PolarsDatasetView::try_with_dtype_overrides(&df, &overrides).unwrap()
    }

    fn make_root_node(dataset: &dyn DatasetView) -> Node {
        let cell = Cell::new().with_rule(
            "target__y1",
            Box::new(QuantizedContinuousInterval::new(20, 42, 0.5, Some((20, 42)), true))
                as Box<dyn DynRule>,
        );
        Node::root(dataset, cell)
    }

    #[test]
    fn quantized_continuous_split_finds_valid_split() {
        let dataset = make_test_dataset();
        let node = make_root_node(&dataset);
        let col = dataset.column("target__y1").unwrap();
        let loss = ConditionalLogLoss;
        let restrictions = SplitRestrictions::default();

        let searcher = QuantizedContinuousColumnSplitSearcher;
        let result = searcher.search(
            &node,
            &node.cell,
            col,
            SplitKind::YSplit,
            &dataset,
            &loss,
            &restrictions,
            None,
            5.0,
        );

        assert!(result.is_some(), "should find a valid split");
        let split = result.unwrap();
        assert!(split.gain > 0.0, "gain should be positive");
    }
}