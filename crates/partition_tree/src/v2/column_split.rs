//! Per-column split search algorithms.
//!
//! The [`ColumnSplitSearcher`] trait defines the contract for finding the best
//! split point on a single column. Two built-in implementations are provided:
//!
//! | Searcher                           | Algorithm                              |
//! |------------------------------------|----------------------------------------|
//! | [`ContinuousColumnSplitSearcher`]  | Presorted scan with moving XY pointer  |
//! | [`CategoricalColumnSplitSearcher`] | Sort-by-score prefix scan              |
//!
//! ## Extension
//!
//! To support a new dtype, implement [`ColumnSplitSearcher`] and register it
//! via a custom [`DTypePlugin`](super::dtype_plugin::DTypePlugin).
use std::collections::HashSet;

use super::cell::Cell;
use super::dataset_view::{ColumnView, DatasetView, LogicalDType};
use super::loss::{CellStats, LossFunc};
use super::node::Node;
use super::split_result::{
    CategoricalSplitOp, ContinuousSplitOp, IntegerSplitOp, SplitKind, SplitPoint, SplitRestrictions,
};

// ---------------------------------------------------------------------------
// ColumnSplitSearcher trait
// ---------------------------------------------------------------------------

/// Contract for finding the best split point on a single column.
///
/// Implementations receive:
///
/// - The current [`Node`] (with presorted indices and weights).
/// - The node’s [`Cell`] (for volume computation).
/// - The [`ColumnView`] to split on.
/// - A [`SplitKind`] indicating feature vs target split.
/// - The full [`DatasetView`] (for weight arrays).
/// - A [`LossFunc`] to evaluate gain.
/// - [`SplitRestrictions`] to prune invalid candidates.
///
/// # Thread Safety
///
/// Must be `Send + Sync` because the [`SplitSearcher`](super::split_searcher::SplitSearcher)
/// dispatches column searches in parallel via `rayon`.
pub trait ColumnSplitSearcher: Send + Sync {
    /// Search for the best split point on `col` given the current `node`.
    ///
    /// Returns `None` if no valid split satisfying the restrictions is found.
    fn search(
        &self,
        node: &Node,
        cell: &Cell,
        col: &dyn ColumnView,
        split_kind: SplitKind,
        dataset: &dyn DatasetView,
        loss: &dyn LossFunc,
        restrictions: &SplitRestrictions,
    ) -> Option<SplitPoint>;
}

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
/// $O(n)$ per split search where $n$ is the node’s sample count.
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
        let (sx_some, wx_null) = split_nulls_continuous(sx, col, weights_x);
        let (sy_some, wy_null) = split_nulls_continuous(sy, col, weights_y);
        let (sxy_some, wxy_null) = split_nulls_continuous(sxy, col, weights_xy);

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
// CategoricalColumnSplitSearcher
// ---------------------------------------------------------------------------

/// Sort-and-scan split search for categorical columns.
///
/// ## Algorithm
///
/// 1. Accumulate per-category statistics $(a_c, b_c)$ from the node’s
///    `sorted_xy` and `sorted_x` (or `sorted_y`) indices.
/// 2. Compute the score $r_c = a_c / b_c$ for each active category.
/// 3. Sort categories by $r_c$ ascending.
/// 4. Scan **prefix splits**: for $t = 0 \ldots m-2$, the left child gets
///    categories $[0 \ldots t]$ and the right gets $[t+1 \ldots m-1]$.
/// 5. Evaluate gain at each prefix; return the best.
///
/// Both `none_to_left = true` and `false` are tried.
///
/// ## Complexity
///
/// $O(n + m \log m)$ where $n$ is the node’s sample count and $m$ is the
/// number of active categories.
#[derive(Debug, Clone)]
pub struct CategoricalColumnSplitSearcher;

impl ColumnSplitSearcher for CategoricalColumnSplitSearcher {
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

        let sxy = node.sorted.sorted_xy.get(col_name)?;

        // 0) Accumulate per-category stats from node indices
        let mut map_a: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();

        for &idx in sxy {
            if let Some(cat) = col.get_cat(idx as usize) {
                *map_a.entry(cat).or_insert(0.0) += weights_xy[idx as usize];
            }
        }

        let mut map_b: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();

        match split_kind {
            SplitKind::XSplit => {
                let sx = node.sorted.sorted_x.get(col_name)?;
                for &idx in sx {
                    if let Some(cat) = col.get_cat(idx as usize) {
                        *map_b.entry(cat).or_insert(0.0) += weights_x[idx as usize];
                    }
                }
            }
            SplitKind::YSplit => {
                let sy = node.sorted.sorted_y.get(col_name)?;
                for &idx in sy {
                    if let Some(cat) = col.get_cat(idx as usize) {
                        *map_b.entry(cat).or_insert(0.0) += weights_y[idx as usize];
                    }
                }
            }
        }

        // Build active category list with (cat, a_c, b_c, r_c)
        let mut cats: Vec<(usize, f64, f64, f64)> = Vec::new();
        for (&cat, &b) in &map_b {
            if b <= 0.0 {
                continue;
            }
            let a = *map_a.get(&cat).unwrap_or(&0.0);
            let r = a / b;
            cats.push((cat, a, b, r));
        }

        if cats.len() < 2 {
            return None;
        }

        // 1) Sort by r_c ascending
        cats.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        // 2) Prefix sums
        let a_pref = cumsum(&cats.iter().map(|c| c.1).collect::<Vec<_>>());
        let b_pref = cumsum(&cats.iter().map(|c| c.2).collect::<Vec<_>>());
        let a_total = *a_pref.last().unwrap();
        let b_total = *b_pref.last().unwrap();

        // Parent stats
        let vol_parent = cell.target_volume();
        let parent_stats = CellStats::new(node.w_xy, node.w_x, node.w_y, vol_parent);

        let mut best: Option<SplitPoint> = None;
        let mut best_gain = f64::NEG_INFINITY;

        // Try both none_to_left options
        for &none_to_left in &[true, false] {
            // 3) Scan prefix splits
            for t in 0..cats.len() - 1 {
                let a_left = a_pref[t];
                let b_left = b_pref[t];
                let a_right = a_total - a_left;
                let b_right = b_total - b_left;

                let subset_left: HashSet<usize> = cats[0..=t].iter().map(|c| c.0).collect();

                let cat_op = CategoricalSplitOp {
                    subset_left: subset_left.clone(),
                };

                let (
                    w_xy_left,
                    w_xy_right,
                    w_x_left,
                    w_x_right,
                    w_y_left,
                    w_y_right,
                    vol_left,
                    vol_right,
                ) = match split_kind {
                    SplitKind::XSplit => {
                        let (vl, vr) = cell.child_target_volumes(col_name, &cat_op, none_to_left);
                        (a_left, a_right, b_left, b_right, node.w_y, node.w_y, vl, vr)
                    }
                    SplitKind::YSplit => {
                        // For Y-split on categorical target: w_x unchanged,
                        // volumes come from the rule split (category counts).
                        let (vl, vr) = cell.child_target_volumes(col_name, &cat_op, none_to_left);
                        (a_left, a_right, node.w_x, node.w_x, b_left, b_right, vl, vr)
                    }
                };

                let left_stats = CellStats::new(w_xy_left, w_x_left, w_y_left, vol_left);
                let right_stats = CellStats::new(w_xy_right, w_x_right, w_y_right, vol_right);

                if !restrictions.is_valid_children(&left_stats, &right_stats, node.depth) {
                    continue;
                }

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
                        op: Box::new(cat_op),
                    });
                }
            }
        }

        best
    }
}

// ---------------------------------------------------------------------------
// IntegerColumnSplitSearcher
// ---------------------------------------------------------------------------

/// Presorted-scan split search for integer columns.
///
/// ## Algorithm
///
/// Analogous to [`ContinuousColumnSplitSearcher`] but operates on `i64` values.
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
    ) -> Option<SplitPoint> {
        let col_name = col.name();
        let weights_xy = dataset.weights_xy();
        let weights_x = dataset.weights_x();
        let weights_y = dataset.weights_y();

        let sx = node.sorted.sorted_x.get(col_name)?;
        let sy = node.sorted.sorted_y.get(col_name)?;
        let sxy = node.sorted.sorted_xy.get(col_name)?;

        // Separate nulls and compute their weights
        let (sx_some, wx_null) = split_nulls_integer(sx, col, weights_x);
        let (sy_some, wy_null) = split_nulls_integer(sy, col, weights_y);
        let (sxy_some, wxy_null) = split_nulls_integer(sxy, col, weights_xy);

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
                        op: Box::new(int_op),
                    });
                }
            }
        }

        best
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Separate non-null indices from null indices for integer columns.
fn split_nulls_integer(sorted: &[u32], col: &dyn ColumnView, weights: &[f64]) -> (Vec<u32>, f64) {
    let mut some = Vec::with_capacity(sorted.len());
    let mut null_weight = 0.0;
    for &idx in sorted {
        if col.is_null(idx as usize) {
            null_weight += weights[idx as usize];
        } else {
            some.push(idx);
        }
    }
    (some, null_weight)
}

/// Separate non-null indices from null indices, returning `(non_null_indices, total_null_weight)`.
fn split_nulls_continuous(
    sorted: &[u32],
    col: &dyn ColumnView,
    weights: &[f64],
) -> (Vec<u32>, f64) {
    let mut some = Vec::with_capacity(sorted.len());
    let mut null_weight = 0.0;
    for &idx in sorted {
        if col.is_null(idx as usize) {
            null_weight += weights[idx as usize];
        } else {
            some.push(idx);
        }
    }
    (some, null_weight)
}

/// Cumulative (prefix) sum of a slice. Returns a `Vec` of the same length.
fn cumsum(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut acc = 0.0;
    for &v in values {
        acc += v;
        result.push(acc);
    }
    result
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

    #[test]
    fn cumsum_works() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let cs = cumsum(&v);
        assert_eq!(cs, vec![1.0, 3.0, 6.0, 10.0]);
    }
}
