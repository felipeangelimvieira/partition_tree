//! Integer column split searcher.
//!
//! Handles `Int32` and `Int64` Polars columns using a presorted scan
//! analogous to the continuous searcher but with integer thresholds.
use super::{ColumnSplitSearcher, cumsum, split_nulls};
use crate::cell::Cell;
use crate::dataset_view::{ColumnView, DatasetView};
use crate::loss::{CellStats, LossFunc};
use crate::node::Node;
use crate::split_result::{IntegerSplitOp, SplitKind, SplitPoint, SplitRestrictions};

// ---------------------------------------------------------------------------
// IntegerColumnSplitSearcher
// ---------------------------------------------------------------------------

/// Presorted-scan split search for integer columns.
///
/// ## Algorithm
///
/// Analogous to [`ContinuousColumnSplitSearcher`](super::ContinuousColumnSplitSearcher)
/// but operates on `i64` values.
/// At each adjacent pair of distinct values $(v_0, v_1)$, the split threshold
/// is $v_1$ (values $< v_1$ go left). This produces the correct discrete
/// partition because all values are integers.
///
/// Both `none_to_left = true` and `false` are tried.
///
/// ## Complexity
///
/// $O(n \log n)$ for the initial sort (done once at dataset construction),
/// $O(n)$ per split search where $n$ is the node's sample count.
#[derive(Debug, Clone)]
pub struct IntegerColumnSplitSearcher;

impl ColumnSplitSearcher for IntegerColumnSplitSearcher {
    fn search(
        &self,
        node: &Node,
        cell: &Cell,
        col: &dyn ColumnView,
        split_kind: SplitKind,
        dataset: &dyn DatasetView,
        loss: &dyn LossFunc,
        restrictions: &SplitRestrictions,
        dataset_size: f64,
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
        let tie_break_start = (candidates.len() - 1) / 2;

        // Try both none_to_left options
        for &none_to_left in &[true, false] {
            let mut p_xy: usize = 0;

            for k in 0..candidates.len() - 1 {
                let i0 = candidates[k] as usize;
                let i1 = candidates[k + 1] as usize;

                let v0 = match col.get_i64(i0) {
                    Some(v) => v,
                    None => continue,
                };
                let v1 = match col.get_i64(i1) {
                    Some(v) => v,
                    None => continue,
                };

                if v0 == v1 {
                    continue;
                }

                // Use v1 as the threshold: values < v1 go left.
                let t = v1;

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
                    match col.get_i64(sxy_some[p_xy] as usize) {
                        Some(v) if v < t => p_xy += 1,
                        _ => break,
                    }
                }
                let w_xy_left_some = if p_xy > 0 { prefix_wxy[p_xy - 1] } else { 0.0 };
                let w_xy_left = w_xy_left_some + if none_to_left { wxy_null } else { 0.0 };
                let w_xy_right = w_xy_parent - w_xy_left;

                // Child volumes
                let int_op = IntegerSplitOp {
                    threshold: t,
                    k_candidate: k,
                    p_xy,
                };
                let (vol_left, vol_right) =
                    cell.child_target_volumes(col_name, &int_op, none_to_left);

                let left_stats = CellStats::new(w_xy_left, w_x_left, w_y_left, vol_left);
                let right_stats = CellStats::new(w_xy_right, w_x_right, w_y_right, vol_right);

                // Validate restrictions
                if !restrictions.is_valid_children(
                    &left_stats,
                    &right_stats,
                    node.depth,
                    cell.target_continuous_domain_volume(),
                    cell.child_target_continuous_volumes(col_name, &int_op, none_to_left),
                ) {
                    continue;
                }

                // Compute gain
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
                        op: Box::new(int_op),
                    });
                }
            }
        }

        best
    }
}
