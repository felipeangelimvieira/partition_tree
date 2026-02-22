//! Multi-column split orchestrator.
//!
//! The [`SplitSearcher`] is the bridge between the [`TreeBuilder`](super::tree_builder::TreeBuilder)
//! and the per-column [`ColumnSplitSearcher`](super::column_split::ColumnSplitSearcher)
//! implementations. Given a node, it:
//!
//! 1. Classifies every column as `XSplit` (feature) or `YSplit` (target).
//! 2. Looks up the appropriate [`DTypePlugin`](super::dtype_plugin::DTypePlugin) in
//!    the [`DTypeRegistry`](super::dtype_plugin::DTypeRegistry).
//! 3. Dispatches column-level searches **in parallel** via `rayon`.
//! 4. Returns the split with the highest gain across all columns.
use std::sync::Arc;

use rayon::prelude::*;

use super::dataset_view::DatasetView;
use super::dtype_plugin::DTypeRegistry;
use super::loss::LossFunc;
use super::node::Node;
use super::split_result::{SplitKind, SplitPoint, SplitRestrictions};

/// Orchestrates split search across all columns of a dataset.
///
/// Feature columns are searched as [`SplitKind::XSplit`], target columns as
/// [`SplitKind::YSplit`]. Individual column searches are parallelized via
/// `rayon::par_iter`.
pub struct SplitSearcher {
    /// Registry mapping dtype → plugin (with split searcher).
    pub registry: Arc<DTypeRegistry>,
}

impl SplitSearcher {
    pub fn new(registry: Arc<DTypeRegistry>) -> Self {
        Self { registry }
    }

    /// Find the best split across **all** columns for the given node.
    ///
    /// Returns `None` when no column produces a valid split that satisfies
    /// the [`SplitRestrictions`].
    pub fn find_best_split(
        &self,
        node: &Node,
        dataset: &dyn DatasetView,
        loss: &dyn LossFunc,
        restrictions: &SplitRestrictions,
    ) -> Option<SplitPoint> {
        // Collect all (column, split_kind) pairs to search
        let columns = dataset.columns();

        let search_tasks: Vec<(&dyn super::dataset_view::ColumnView, SplitKind)> = columns
            .into_iter()
            .map(|col| {
                let kind = if col.is_target() {
                    SplitKind::YSplit
                } else {
                    SplitKind::XSplit
                };
                (col, kind)
            })
            .collect();

        // Search all columns in parallel
        let results: Vec<Option<SplitPoint>> = search_tasks
            .par_iter()
            .map(|(col, split_kind)| {
                let dtype = col.logical_dtype();
                let plugin = self.registry.get(dtype)?;
                let searcher = plugin.split_searcher();

                searcher.search(node, &node.cell, *col, *split_kind, dataset, loss, restrictions)
            })
            .collect();

        // Return the split with the highest gain
        results
            .into_iter()
            .flatten()
            .max_by(|a, b| {
                a.gain
                    .partial_cmp(&b.gain)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::ContinuousInterval;
    use crate::v2::cell::Cell;
    use crate::v2::dataset_view::PolarsDatasetView;
    use crate::v2::loss::ConditionalLogLoss;
    use crate::v2::rule::RuleType;
    use polars::prelude::*;

    fn make_test_setup() -> (PolarsDatasetView, Node, SplitSearcher) {
        let df = DataFrame::new(vec![
            Column::new("x1".into(), &[1.0_f64, 3.0, 2.0, 5.0, 4.0]),
            Column::new("target__y1".into(), &[10.0_f64, 30.0, 20.0, 50.0, 40.0]),
        ])
        .unwrap();
        let dataset = PolarsDatasetView::new(&df);

        let cell = Cell::new()
            .with_rule(
                "x1",
                RuleType::Continuous(ContinuousInterval::new(
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    true,
                    true,
                    Some((f64::NEG_INFINITY, f64::INFINITY)),
                    true,
                )),
            )
            .with_rule(
                "target__y1",
                RuleType::Continuous(ContinuousInterval::new(
                    0.0,
                    60.0,
                    true,
                    true,
                    Some((0.0, 60.0)),
                    true,
                )),
            );

        let node = Node::root(&dataset, cell);
        let registry = Arc::new(DTypeRegistry::default());
        let searcher = SplitSearcher::new(registry);

        (dataset, node, searcher)
    }

    #[test]
    fn finds_best_split_across_columns() {
        let (dataset, node, searcher) = make_test_setup();
        let loss = ConditionalLogLoss::new(5.0);
        let restrictions = SplitRestrictions::default();

        let result = searcher.find_best_split(&node, &dataset, &loss, &restrictions);

        assert!(result.is_some(), "should find a valid split");
        let split = result.unwrap();
        assert!(split.gain > 0.0);
    }
}
