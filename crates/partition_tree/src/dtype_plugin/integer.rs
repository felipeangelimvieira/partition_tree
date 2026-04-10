//! Integer dtype plugin.
//!
//! Handles `Int32` and `Int64` Polars columns as discrete integer values.
//!
//! - **Feature columns**: creates an `IntegerInterval` spanning
//!   `[i64::MIN, i64::MAX]` (effectively unbounded).
//! - **Target columns**: scans for `[min, max]` of observed values and
//!   applies the boundary expansion factor to produce a wider interval.
use crate::conf::TARGET_PREFIX;
use crate::rules::IntegerInterval;

use super::DTypePlugin;
use crate::column_split::{ColumnSplitSearcher, IntegerColumnSplitSearcher};
use crate::dataset_view::{ColumnView, LogicalDType};
use crate::rule::DynRule;

// ---------------------------------------------------------------------------
// IntegerPlugin
// ---------------------------------------------------------------------------

/// Plugin for integer (`Int32`, `Int64`) columns.
///
/// - **Feature columns**: creates an `IntegerInterval` spanning
///   `[i64::MIN, i64::MAX]` (unbounded integer domain).
/// - **Target columns**: scans for `[min, max]` and applies the boundary
///   expansion factor to produce a wider interval.
pub struct IntegerPlugin {
    searcher: IntegerColumnSplitSearcher,
}

impl IntegerPlugin {
    /// Create a new `IntegerPlugin`.
    pub fn new() -> Self {
        Self {
            searcher: IntegerColumnSplitSearcher,
        }
    }
}

impl Default for IntegerPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DTypePlugin for IntegerPlugin {
    fn logical_dtype(&self) -> LogicalDType {
        LogicalDType::Integer
    }

    fn default_rule(
        &self,
        col: &dyn ColumnView,
        boundaries_expansion_factor: f64,
    ) -> Box<dyn DynRule> {
        let is_target = col.name().starts_with(TARGET_PREFIX);

        if is_target {
            // For target columns: scan for min/max and expand
            let mut min_val = i64::MAX;
            let mut max_val = i64::MIN;
            for i in 0..col.len() {
                if let Some(v) = col.get_i64(i) {
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
                min_val = 0;
                max_val = 1;
            }

            // Expand: add extra integers on each side proportional to the range
            let range = (max_val as f64 - min_val as f64).abs();
            let expansion = (range * boundaries_expansion_factor / 2.0).ceil() as i64;
            let low = min_val.saturating_sub(expansion);
            let high = max_val.saturating_add(expansion);

            Box::new(IntegerInterval::new(low, high, Some((low, high)), true))
        } else {
            // For feature columns: unbounded integer interval
            Box::new(IntegerInterval::new(
                i64::MIN,
                i64::MAX,
                Some((i64::MIN, i64::MAX)),
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
    use crate::rules::IntegerInterval;
    use polars::prelude::*;

    #[test]
    fn integer_plugin_creates_unbounded_feature_rule() {
        let df = DataFrame::new(vec![Column::new("x1".into(), &[1_i64, 2, 3])]).unwrap();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("x1").unwrap();

        let plugin = IntegerPlugin::new();
        let rule = plugin.default_rule(col, 0.1);

        let ii = rule
            .as_any()
            .downcast_ref::<IntegerInterval>()
            .expect("expected IntegerInterval");
        assert_eq!(ii.low, i64::MIN);
        assert_eq!(ii.high, i64::MAX);
    }

    #[test]
    fn integer_plugin_creates_bounded_target_rule() {
        let df = DataFrame::new(vec![Column::new("target__y1".into(), &[10_i64, 20, 30])]).unwrap();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("target__y1").unwrap();

        let plugin = IntegerPlugin::new();
        let rule = plugin.default_rule(col, 0.1);

        let ii = rule
            .as_any()
            .downcast_ref::<IntegerInterval>()
            .expect("expected IntegerInterval");
        assert!(ii.low <= 10, "low should be at or below min value");
        assert!(ii.high >= 30, "high should be at or above max value");
    }

    #[test]
    fn integer_plugin_logical_dtype() {
        let plugin = IntegerPlugin::new();
        assert_eq!(plugin.logical_dtype(), LogicalDType::Integer);
    }
}
