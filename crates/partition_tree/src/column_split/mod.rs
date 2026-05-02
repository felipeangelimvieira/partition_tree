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
pub mod quantized_continuous;

pub use categorical::CategoricalColumnSplitSearcher;
pub use continuous::ContinuousColumnSplitSearcher;
pub use integer::IntegerColumnSplitSearcher;
pub use quantized_continuous::QuantizedContinuousColumnSplitSearcher;

use crate::cell::Cell;
use crate::dataset_view::{ColumnView, DatasetView};
use crate::loss::LossFunc;
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
    /// `dataset_size` is the normalizing constant $D$ forwarded to
    /// [`LossFunc::gain`].
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
        max_candidate_split_points: Option<usize>,
        dataset_size: f64,
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

/// Deterministically clip valid candidate split positions.
///
/// The input is assumed to already contain only valid split boundaries. When a
/// cap is provided, we keep a deterministic subset that spans the full range of
/// positions instead of taking the first `max` entries.
pub(crate) fn clip_candidate_positions(
    candidate_positions: &[usize],
    max_candidate_split_points: Option<usize>,
) -> Vec<usize> {
    let total = candidate_positions.len();

    match max_candidate_split_points {
        _ if total == 0 => Vec::new(),
        None => candidate_positions.to_vec(),
        Some(0) => Vec::new(),
        Some(max) if max >= total => candidate_positions.to_vec(),
        // With a single retained candidate, choose the middle valid position.
        Some(1) => vec![candidate_positions[total / 2]],
        Some(max) => (0..max)
            .map(|slot| {
                // Map each output slot onto the closed interval [0, total - 1]
                // so the retained candidates are spread across the full search
                // range and remain stable across runs.
                let idx = slot * (total - 1) / (max - 1);
                candidate_positions[idx]
            })
            .collect(),
    }
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

    #[test]
    fn clip_candidate_positions_preserves_all_when_unbounded() {
        let positions = vec![1, 3, 4, 8, 10];

        assert_eq!(clip_candidate_positions(&positions, None), positions);
    }

    #[test]
    fn clip_candidate_positions_samples_evenly() {
        let positions = vec![1, 3, 4, 8, 10];

        assert_eq!(
            clip_candidate_positions(&positions, Some(3)),
            vec![1, 4, 10]
        );
        assert_eq!(clip_candidate_positions(&positions, Some(1)), vec![4]);
        assert!(clip_candidate_positions(&positions, Some(0)).is_empty());
    }
}
