//! Multi-column split orchestrator.
//!
//! The [`SplitSearcher`] is the bridge between the [`TreeBuilder`](super::tree_builder::TreeBuilder)
//! and the per-column [`ColumnSplitSearcher`](super::column_split::ColumnSplitSearcher)
//! implementations. Given a node, it:
//!
//! 1. Classifies every column as `XSplit` (feature) or `YSplit` (target).
//! 2. Looks up the appropriate [`DTypePlugin`](super::dtype_plugin::DTypePlugin) in
//!    the [`DTypeRegistry`].
//! 3. Dispatches column-level searches **in parallel** via `rayon`.
//! 4. Returns the split with the highest gain across all columns.
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

use crate::dataset_view::DatasetView;
use crate::dtype_plugin::DTypeRegistry;
use crate::loss::LossFunc;
use crate::node::Node;
use crate::split_result::{SplitKind, SplitPoint, SplitRestrictions};

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
    /// When `max_features` is `Some(k)`, only `k` randomly chosen feature
    /// columns are considered (target columns are always included).
    ///
    /// `dataset_size` is the normalizing constant $D$ forwarded to
    /// [`LossFunc::gain`].
    ///
    /// Returns `None` when no column produces a valid split that satisfies
    /// the [`SplitRestrictions`].
    pub fn find_best_split(
        &self,
        node: &Node,
        dataset: &dyn DatasetView,
        loss: &dyn LossFunc,
        restrictions: &SplitRestrictions,
        max_features: Option<usize>,
        rng: &mut StdRng,
        dataset_size: f64,
    ) -> Option<SplitPoint> {
        // Collect all (column, split_kind) pairs to search
        let columns = dataset.columns();

        let search_tasks: Vec<(&dyn crate::dataset_view::ColumnView, SplitKind)> =
            if let Some(max_f) = max_features {
                // Separate features and targets
                let mut features: Vec<(
                    &dyn crate::dataset_view::ColumnView,
                    SplitKind,
                )> = Vec::new();
                let mut targets: Vec<(
                    &dyn crate::dataset_view::ColumnView,
                    SplitKind,
                )> = Vec::new();

                for col in &columns {
                    if col.is_target() {
                        targets.push((*col, SplitKind::YSplit));
                    } else {
                        features.push((*col, SplitKind::XSplit));
                    }
                }

                // Shuffle and subsample feature columns
                features.shuffle(rng);
                features.truncate(max_f);
                features.extend(targets);
                features
            } else {
                columns
                    .into_iter()
                    .map(|col| {
                        let kind = if col.is_target() {
                            SplitKind::YSplit
                        } else {
                            SplitKind::XSplit
                        };
                        (col, kind)
                    })
                    .collect()
            };

        // Search all columns in parallel
        let results: Vec<Option<SplitPoint>> = search_tasks
            .par_iter()
            .map(|(col, split_kind)| {
                let dtype = col.logical_dtype();
                let plugin = self.registry.get(dtype)?;
                let searcher = plugin.split_searcher();

                searcher.search(
                    node,
                    &node.cell,
                    *col,
                    *split_kind,
                    dataset,
                    loss,
                    restrictions,
                    dataset_size,
                )
            })
            .collect();

        // Return the split with the highest gain
        results.into_iter().flatten().max_by(|a, b| {
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
    use crate::cell::Cell;
    use crate::dataset_view::PolarsDatasetView;
    use crate::loss::ConditionalLogLoss;
    use crate::rule::DynRule;
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

        let node = Node::root(&dataset, cell);
        let registry = Arc::new(DTypeRegistry::default());
        let searcher = SplitSearcher::new(registry);

        (dataset, node, searcher)
    }

    #[test]
    fn finds_best_split_across_columns() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let (dataset, node, searcher) = make_test_setup();
        let loss = ConditionalLogLoss;
        let restrictions = SplitRestrictions::default();
        let mut rng = StdRng::seed_from_u64(42);

        let result = searcher.find_best_split(&node, &dataset, &loss, &restrictions, None, &mut rng, 5.0);

        assert!(result.is_some(), "should find a valid split");
        let split = result.unwrap();
        assert!(split.gain > 0.0);
    }

    #[test]
    fn find_best_split_with_max_features_finds_split() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        // Setup has 1 feature (x1) and 1 target (target__y1).
        // max_features=Some(1) keeps the only feature → should still find a split.
        let (dataset, node, searcher) = make_test_setup();
        let loss = ConditionalLogLoss;
        let restrictions = SplitRestrictions::default();
        let mut rng = StdRng::seed_from_u64(42);

        let result = searcher.find_best_split(
            &node, &dataset, &loss, &restrictions, Some(1), &mut rng, 5.0,
        );

        assert!(result.is_some(), "should find a split even with max_features=1");
        assert!(result.unwrap().gain > 0.0);
    }

    #[test]
    fn max_features_always_includes_targets() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        // With a multi-feature dataset, max_features=1 selects only 1 feature
        // but target columns should still be searchable (YSplit).
        let df = DataFrame::new(vec![
            Column::new("x1".into(), &[1.0_f64, 3.0, 2.0, 5.0, 4.0]),
            Column::new("x2".into(), &[10.0_f64, 20.0, 30.0, 40.0, 50.0]),
            Column::new("target__y1".into(), &[10.0_f64, 30.0, 20.0, 50.0, 40.0]),
        ])
        .unwrap();
        let dataset = PolarsDatasetView::new(&df);

        let cell = Cell::new()
            .with_rule(
                "x1",
                Box::new(ContinuousInterval::new(
                    f64::NEG_INFINITY, f64::INFINITY, true, true,
                    Some((f64::NEG_INFINITY, f64::INFINITY)), true,
                )) as Box<dyn DynRule>,
            )
            .with_rule(
                "x2",
                Box::new(ContinuousInterval::new(
                    f64::NEG_INFINITY, f64::INFINITY, true, true,
                    Some((f64::NEG_INFINITY, f64::INFINITY)), true,
                )) as Box<dyn DynRule>,
            )
            .with_rule(
                "target__y1",
                Box::new(ContinuousInterval::new(
                    0.0, 60.0, true, true, Some((0.0, 60.0)), true,
                )) as Box<dyn DynRule>,
            );

        let node = Node::root(&dataset, cell);
        let registry = Arc::new(DTypeRegistry::default());
        let searcher = SplitSearcher::new(registry);
        let loss = ConditionalLogLoss;
        let restrictions = SplitRestrictions::default();

        // Run many iterations — with max_features=1, only 1 of 2 features
        // is chosen per call, but the target is always included.
        // Over many runs, if targets are excluded, we'd see only XSplits.
        let mut saw_ysplit = false;
        for seed in 0..50 {
            let mut rng = StdRng::seed_from_u64(seed);
            if let Some(split) = searcher.find_best_split(
                &node, &dataset, &loss, &restrictions, Some(1), &mut rng, 5.0,
            ) {
                if split.split_kind == SplitKind::YSplit {
                    saw_ysplit = true;
                    break;
                }
            }
        }

        assert!(
            saw_ysplit,
            "target columns should always be included: expected at least one YSplit"
        );
    }
}
