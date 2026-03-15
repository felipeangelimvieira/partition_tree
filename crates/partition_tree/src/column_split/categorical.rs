//! Categorical column split searcher.
//!
//! Handles `Enum` and `Categorical` Polars columns using a sort-by-score
//! prefix scan algorithm.
use std::collections::HashSet;

use super::{ColumnSplitSearcher, cumsum};
use crate::cell::Cell;
use crate::dataset_view::{ColumnView, DatasetView};
use crate::loss::{CellStats, LossFunc};
use crate::node::Node;
use crate::split_result::{CategoricalSplitOp, SplitKind, SplitPoint, SplitRestrictions};

// ---------------------------------------------------------------------------
// CategoricalColumnSplitSearcher
// ---------------------------------------------------------------------------

/// Sort-and-scan split search for categorical columns.
///
/// ## Algorithm
///
/// 1. Accumulate per-category statistics $(a_c, b_c)$ from the node's
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
/// $O(n + m \log m)$ where $n$ is the node's sample count and $m$ is the
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
        dataset_size: f64,
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
                if loss.uses_empirical_y_measure() {
                    // BalancedLogLoss: b_c = empirical weight sum Σ w_y per category
                    for &idx in sy {
                        if let Some(cat) = col.get_cat(idx as usize) {
                            *map_b.entry(cat).or_insert(0.0) += weights_y[idx as usize];
                        }
                    }
                } else {
                    // ConditionalLogLoss: b_c = 1.0 per category (Lebesgue volume)
                    for &idx in sy {
                        if let Some(cat) = col.get_cat(idx as usize) {
                            map_b.entry(cat).or_insert(1.0);
                        }
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

        // 1) Sort by r_c ascending, breaking ties by category index
        cats.sort_by(|a, b| {
            a.3.partial_cmp(&b.3)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

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

                if !restrictions.is_valid_children(
                    &left_stats,
                    &right_stats,
                    node.depth,
                    cell.target_domain_volume(),
                ) {
                    continue;
                }

                let gain = loss.gain(&parent_stats, &left_stats, &right_stats, dataset_size);
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
