//! Per-column split search algorithms.
//!
//! The [`ColumnSplitSearcher`] trait defines the contract for finding the best
//! split point on a single column. Three built-in implementations are provided:
//!
//! | Searcher                           | Algorithm                              |
//! |------------------------------------|----------------------------------------|
//! | [`ContinuousColumnSplitSearcher`]  | Presorted scan with moving XY pointer  |
//! | [`CategoricalColumnSplitSearcher`] | Sort-by-score prefix scan              |
//! | [`IntegerColumnSplitSearcher`]     | Presorted scan (integer thresholds)    |
//!
//! ## Extension
//!
//! To support a new dtype, implement [`ColumnSplitSearcher`] and register it
//! via a custom [`DTypePlugin`](super::dtype_plugin::DTypePlugin).
pub mod categorical;
pub mod continuous;
pub mod integer;

pub use categorical::CategoricalColumnSplitSearcher;
pub use continuous::ContinuousColumnSplitSearcher;
pub use integer::IntegerColumnSplitSearcher;

use crate::cell::Cell;
use crate::dataset_view::{ColumnView, DatasetView};
use crate::loss::{CellStats, LossFunc};
use crate::node::Node;
use crate::split_result::{SplitKind, SplitPoint, SplitRestrictions};

// ---------------------------------------------------------------------------
// ColumnSplitSearcher trait
// ---------------------------------------------------------------------------

/// Contract for finding the best split point on a single column.
///
/// Implementations receive:
///
/// - The current [`Node`] (with presorted indices and weights).
/// - The node's [`Cell`] (for volume computation).
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
// Shared helpers
// ---------------------------------------------------------------------------

/// Separate non-null indices from null indices, returning
/// `(non_null_indices, total_null_weight)`.
pub(crate) fn split_nulls(
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
pub(crate) fn cumsum(values: &[f64]) -> Vec<f64> {
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

    #[test]
    fn cumsum_works() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let cs = cumsum(&v);
        assert_eq!(cs, vec![1.0, 3.0, 6.0, 10.0]);
    }
}
