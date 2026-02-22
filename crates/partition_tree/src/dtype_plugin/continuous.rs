//! Continuous dtype plugin.
//!
//! Handles `Float64`, `Float32`, `Int32`, `Int64` and similar numeric columns.
//!
//! - **Feature columns**: creates an unbounded interval $(-\infty, +\infty)$.
//! - **Target columns**: scans for `[min, max]` and applies the boundary
//!   expansion factor to produce a finite interval.
use crate::conf::TARGET_PREFIX;
use crate::rules::ContinuousInterval;

use crate::column_split::{ColumnSplitSearcher, ContinuousColumnSplitSearcher};
use crate::dataset_view::{ColumnView, LogicalDType};
use crate::rule::DynRule;
use super::DTypePlugin;

// ---------------------------------------------------------------------------
// ContinuousPlugin
// ---------------------------------------------------------------------------

/// Plugin for continuous (`Float64`, `Float32`, `Int32`, `Int64`) columns.
///
/// - **Feature columns**: creates an unbounded interval $(-\infty, +\infty)$.
/// - **Target columns**: scans for `[min, max]` and applies the boundary
///   expansion factor to produce a finite interval.
pub struct ContinuousPlugin {
    searcher: ContinuousColumnSplitSearcher,
}

impl ContinuousPlugin {
    /// Create a new `ContinuousPlugin`.
    pub fn new() -> Self {
        Self {
            searcher: ContinuousColumnSplitSearcher,
        }
    }
}

impl Default for ContinuousPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DTypePlugin for ContinuousPlugin {
    fn logical_dtype(&self) -> LogicalDType {
        LogicalDType::Continuous
    }

    fn default_rule(&self, col: &dyn ColumnView, boundaries_expansion_factor: f64) -> Box<dyn DynRule> {
        let is_target = col.name().starts_with(TARGET_PREFIX);

        if is_target {
            // For target columns: scan for min/max and expand
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..col.len() {
                if let Some(v) = col.get_f64(i) {
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }
            }

            if min_val > max_val {
                // No non-null values
                min_val = 0.0;
                max_val = 1.0;
            }

            let range = max_val - min_val;
            let mean = (max_val + min_val) / 2.0;
            let expansion = range * (1.0 + boundaries_expansion_factor);
            let low = mean - expansion / 2.0;
            let high = mean + expansion / 2.0;

            Box::new(ContinuousInterval::new(
                low,
                high,
                true,
                true,
                Some((low, high)),
                true,
            ))
        } else {
            // For feature columns: unbounded interval
            Box::new(ContinuousInterval::new(
                f64::NEG_INFINITY,
                f64::INFINITY,
                true,
                true,
                Some((f64::NEG_INFINITY, f64::INFINITY)),
                true,
            ))
        }
    }

    fn split_searcher(&self) -> &dyn ColumnSplitSearcher {
        &self.searcher
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset_view::{DatasetView, PolarsDatasetView};
    use crate::rules::ContinuousInterval;
    use polars::prelude::*;

    #[test]
    fn continuous_plugin_creates_unbounded_feature_rule() {
        let df = DataFrame::new(vec![Column::new("x1".into(), &[1.0_f64, 2.0, 3.0])]).unwrap();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("x1").unwrap();

        let plugin = ContinuousPlugin::new();
        let rule = plugin.default_rule(col, 0.1);

        let ci = rule.as_any().downcast_ref::<ContinuousInterval>()
            .expect("expected ContinuousInterval");
        assert!(ci.low.is_infinite());
        assert!(ci.high.is_infinite());
    }

    #[test]
    fn continuous_plugin_creates_bounded_target_rule() {
        let df = DataFrame::new(vec![Column::new(
            "target__y1".into(),
            &[10.0_f64, 20.0, 30.0],
        )])
        .unwrap();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("target__y1").unwrap();

        let plugin = ContinuousPlugin::new();
        let rule = plugin.default_rule(col, 0.1);

        let ci = rule.as_any().downcast_ref::<ContinuousInterval>()
            .expect("expected ContinuousInterval");
        assert!(ci.low < 10.0, "low should be below min value");
        assert!(ci.high > 30.0, "high should be above max value");
        assert!(ci.low.is_finite());
        assert!(ci.high.is_finite());
    }
}
