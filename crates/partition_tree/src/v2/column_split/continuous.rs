//! Continuous column split searcher.
//!
//! Handles `Float64`, `Float32` and similar continuous numeric columns
//! using a presorted scan with a moving XY pointer.
use super::super::cell::Cell;
use super::super::dataset_view::{ColumnView, DatasetView};
use super::super::loss::{CellStats, LossFunc};
use super::super::node::Node;
use super::super::split_result::{ContinuousSplitOp, SplitKind, SplitPoint, SplitRestrictions};
use super::{ColumnSplitSearcher, cumsum, split_nulls};

// ---------------------------------------------------------------------------
// ContinuousColumnSplitSearcher
// ---------------------------------------------------------------------------

/// Presorted-scan split search for continuous columns.
///
/// ## Algorithm
///
/// 1. Extract the node-local presorted lists for the split column and
///    separate null from non-null indices.
/// 2. Choose the **candidate list** based on split kind:
///    `sorted_x` for `XSplit`, `sorted_y` for `YSplit`.
/// 3. Build prefix sums on the candidate list weights.
/// 4. Use a **moving pointer** on `sorted_xy` to compute `w_xy_left` at
///    each candidate midpoint threshold.
/// 5. Evaluate gain at each midpoint; return the best.
///
/// Both `none_to_left = true` and `false` are tried.
///
/// ## Complexity
///
/// $O(n \log n)$ for the initial sort (done once at dataset construction),
/// $O(n)$ per split search where $n$ is the node's sample count.
#[derive(Debug, Clone)]
pub struct ContinuousColumnSplitSearcher;

impl ColumnSplitSearcher for ContinuousColumnSplitSearcher {
    fn search(
        &self,
        node: &Node,
        cell: &Cell,
        col: &dyn ColumnView,
        split_kind: SplitKind,
        dataset: &dyn DatasetView,
        loss: &dyn LossFunc,
        restrictions: &SplitRestrictions,
    ) -> Option<SplitPoint> {
        let col_name = col.name();
        let weights_xy = dataset.weights_xy();
        let weights_x = dataset.weights_x();
        let weights_y = dataset.weights_y();

        let sx = node.sorted.sorted_x.get(col_name)?;
        let sy = node.sorted.sorted_y.get(col_name)?;
        let sxy = node.sorted.sorted_xy.get(col_name)?;

        // Separate nulls and compute their weights
        let (sx_some, wx_null) = split_nulls(sx, col, weights_x);
        let (sy_some, wy_null) = split_nulls(sy, col, weights_y);
        let (sxy_some, wxy_null) = split_nulls(sxy, col, weights_xy);

        // Total weights at parent
        let w_x_parent: f64 = sx_some.iter().map(|&i| weights_x[i as usize]).sum::<f64>() + wx_null;
        let w_y_parent: f64 = sy_some.iter().map(|&i| weights_y[i as usize]).sum::<f64>() + wy_null;
        let w_xy_parent: f64 = sxy_some
            .iter()
            .map(|&i| weights_xy[i as usize])
            .sum::<f64>()
            + wxy_null;

        let vol_parent = cell.target_volume();
        let parent_stats = CellStats::new(w_xy_parent, w_x_parent, w_y_parent, vol_parent);

        // Choose candidate list based on split kind
        let candidates = match split_kind {
            SplitKind::XSplit => &sx_some,
            SplitKind::YSplit => &sy_some,
        };

        if candidates.len() < 2 {
            return None;
        }

        // Prefix sums on candidate weights
        let candidate_weights: Vec<f64> = match split_kind {
            SplitKind::XSplit => candidates.iter().map(|&i| weights_x[i as usize]).collect(),
            SplitKind::YSplit => candidates.iter().map(|&i| weights_y[i as usize]).collect(),
        };
        let prefix_candidate = cumsum(&candidate_weights);

        // Prefix sums on sxy_some weights
        let sxy_weights: Vec<f64> = sxy_some.iter().map(|&i| weights_xy[i as usize]).collect();
        let prefix_wxy = cumsum(&sxy_weights);

        let mut best: Option<SplitPoint> = None;
        let mut best_gain = f64::NEG_INFINITY;

        // Try both none_to_left options
        for &none_to_left in &[true, false] {
            let mut p_xy: usize = 0;

            for k in 0..candidates.len() - 1 {
                let i0 = candidates[k] as usize;
                let i1 = candidates[k + 1] as usize;

                let v0 = match col.get_f64(i0) {
                    Some(v) => v,
                    None => continue,
                };
                let v1 = match col.get_f64(i1) {
                    Some(v) => v,
                    None => continue,
                };

                if (v0 - v1).abs() < f64::EPSILON {
                    continue;
                }

                let t = 0.5 * (v0 + v1);

                // Candidate-measure left weights
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

                // Move pointer on sxy_some to compute w_xy_left
                while p_xy < sxy_some.len() {
                    match col.get_f64(sxy_some[p_xy] as usize) {
                        Some(v) if v < t => p_xy += 1,
                        _ => break,
                    }
                }
                let w_xy_left_some = if p_xy > 0 { prefix_wxy[p_xy - 1] } else { 0.0 };
                let w_xy_left = w_xy_left_some + if none_to_left { wxy_null } else { 0.0 };
                let w_xy_right = w_xy_parent - w_xy_left;

                // Child volumes
                let cont_op = ContinuousSplitOp {
                    threshold: t,
                    k_candidate: k,
                    p_xy,
                };
                let (vol_left, vol_right) =
                    cell.child_target_volumes(col_name, &cont_op, none_to_left);

                let left_stats = CellStats::new(w_xy_left, w_x_left, w_y_left, vol_left);
                let right_stats = CellStats::new(w_xy_right, w_x_right, w_y_right, vol_right);

                // Validate restrictions
                if !restrictions.is_valid_children(&left_stats, &right_stats, node.depth) {
                    continue;
                }

                // Compute gain
                let gain = loss.gain(&parent_stats, &left_stats, &right_stats);
                if gain < restrictions.min_gain {
                    continue;
                }

                if gain > best_gain {
                    best_gain = gain;
                    best = Some(SplitPoint {
                        col_name: col_name.to_string(),
                        split_kind,
                        none_to_left,
                        gain,
                        left_stats,
                        right_stats,
                        op: Box::new(cont_op),
                    });
                }
            }
        }

        best
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::ContinuousInterval;
    use crate::v2::dataset_view::PolarsDatasetView;
    use crate::v2::loss::ConditionalLogLoss;
    use crate::v2::rule::DynRule;
    use polars::prelude::*;

    fn make_test_dataset() -> PolarsDatasetView {
        // Craft data so that an X-split at x1=2.5 is informative:
        // rows with x1<2.5 are associated with low target values,
        // rows with x1>2.5 with high target values.
        // This creates a non-uniform density w_xy/(w_x*vol) across children.
        let df = DataFrame::new(vec![
            Column::new("x1".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0]),
            Column::new("target__y1".into(), &[10.0_f64, 11.0, 40.0, 41.0, 42.0]),
        ])
        .unwrap();
        PolarsDatasetView::new(&df)
    }

    fn make_root_node(dataset: &dyn DatasetView) -> Node {
        let cell = Cell::new()
            .with_rule(
                "x1",
                Box::new(ContinuousInterval::new(
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    true,
                    true,
                    Some((f64::NEG_INFINITY, f64::INFINITY)),
                    true,
                )) as Box<dyn DynRule>,
            )
            .with_rule(
                "target__y1",
                Box::new(ContinuousInterval::new(
                    0.0,
                    60.0,
                    true,
                    true,
                    Some((0.0, 60.0)),
                    true,
                )) as Box<dyn DynRule>,
            );
        Node::root(dataset, cell)
    }

    #[test]
    fn continuous_split_finds_valid_split() {
        let dataset = make_test_dataset();
        let node = make_root_node(&dataset);
        // Search on the target column (YSplit) — this changes volume and produces
        // positive gain under ConditionalLogLoss.
        let col = dataset.column("target__y1").unwrap();
        let loss = ConditionalLogLoss::new(5.0);
        let restrictions = SplitRestrictions::default();

        let searcher = ContinuousColumnSplitSearcher;
        let result = searcher.search(
            &node,
            &node.cell,
            col,
            SplitKind::YSplit,
            &dataset,
            &loss,
            &restrictions,
        );

        assert!(result.is_some(), "should find a valid split");
        let split = result.unwrap();
        assert_eq!(split.col_name, "target__y1");
        assert!(split.gain > 0.0, "gain should be positive");
    }

    #[test]
    fn continuous_split_on_target_column() {
        let dataset = make_test_dataset();
        let node = make_root_node(&dataset);
        let col = dataset.column("target__y1").unwrap();
        let loss = ConditionalLogLoss::new(5.0);
        let restrictions = SplitRestrictions::default();

        let searcher = ContinuousColumnSplitSearcher;
        let result = searcher.search(
            &node,
            &node.cell,
            col,
            SplitKind::YSplit,
            &dataset,
            &loss,
            &restrictions,
        );

        assert!(result.is_some(), "should find a valid target split");
        let split = result.unwrap();
        assert_eq!(split.split_kind, SplitKind::YSplit);
    }
}
