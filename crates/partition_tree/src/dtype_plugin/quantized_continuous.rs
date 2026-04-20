//! Quantized-continuous dtype plugin.
//!
//! Handles `f64` columns constrained to a fixed lattice `resolution * i`.
//! Rule and split semantics match the integer dtype on that lattice while
//! preserving `f64` values in the public API.
use crate::conf::TARGET_PREFIX;
use crate::rules::QuantizedContinuousInterval;

use super::DTypePlugin;
use crate::column_split::{ColumnSplitSearcher, QuantizedContinuousColumnSplitSearcher};
use crate::dataset_view::{ColumnView, LogicalDType};
use crate::rule::DynRule;

/// Plugin for quantized-continuous columns.
pub struct QuantizedContinuousPlugin {
    resolution: f64,
    searcher: QuantizedContinuousColumnSplitSearcher,
}

impl QuantizedContinuousPlugin {
    /// Create a new quantized-continuous plugin.
    pub fn new(resolution: f64) -> Self {
        assert!(
            resolution.is_finite() && resolution > 0.0,
            "QuantizedContinuousPlugin requires a finite positive resolution"
        );

        Self {
            resolution,
            searcher: QuantizedContinuousColumnSplitSearcher::new(resolution),
        }
    }

    /// The lattice resolution used by this plugin.
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    fn scan_lattice_bounds(&self, col: &dyn ColumnView) -> Option<(i64, i64)> {
        let mut min_idx = i64::MAX;
        let mut max_idx = i64::MIN;

        for i in 0..col.len() {
            let Some(value) = col.get_f64(i) else {
                continue;
            };

            let idx = QuantizedContinuousInterval::quantize_value(value, self.resolution)
                .unwrap_or_else(|| {
                    panic!(
                        "Column '{}' uses QuantizedContinuous(resolution={}) but value {} is not aligned to the lattice",
                        col.name(),
                        self.resolution,
                        value
                    )
                });

            if idx < min_idx {
                min_idx = idx;
            }
            if idx > max_idx {
                max_idx = idx;
            }
        }

        if min_idx > max_idx {
            None
        } else {
            Some((min_idx, max_idx))
        }
    }
}

impl Default for QuantizedContinuousPlugin {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl DTypePlugin for QuantizedContinuousPlugin {
    fn logical_dtype(&self) -> LogicalDType {
        LogicalDType::QuantizedContinuous
    }

    fn default_rule(
        &self,
        col: &dyn ColumnView,
        boundaries_expansion_factor: f64,
    ) -> Box<dyn DynRule> {
        let is_target = col.name().starts_with(TARGET_PREFIX);
        let bounds = self.scan_lattice_bounds(col);

        if is_target {
            let (min_idx, max_idx) = bounds.unwrap_or((0, 1));
            let range_steps = (max_idx - min_idx).abs() as f64;
            let expansion = (range_steps * boundaries_expansion_factor / 2.0).ceil() as i64;
            let low_idx = min_idx.saturating_sub(expansion);
            let high_idx = max_idx.saturating_add(expansion);

            Box::new(QuantizedContinuousInterval::new(
                low_idx,
                high_idx,
                self.resolution,
                Some((low_idx, high_idx)),
                true,
            ))
        } else {
            Box::new(QuantizedContinuousInterval::new(
                i64::MIN,
                i64::MAX,
                self.resolution,
                Some((i64::MIN, i64::MAX)),
                true,
            ))
        }
    }

    fn split_searcher(&self) -> &dyn ColumnSplitSearcher {
        &self.searcher
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset_view::{DatasetView, PolarsDatasetView};
    use polars::prelude::*;

    #[test]
    fn quantized_plugin_creates_unbounded_feature_rule() {
        let df = DataFrame::new(vec![Column::new("x1".into(), &[0.0_f64, 0.5, 1.0])]).unwrap();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("x1").unwrap();

        let plugin = QuantizedContinuousPlugin::new(0.5);
        let rule = plugin.default_rule(col, 0.1);

        let qi = rule
            .as_any()
            .downcast_ref::<QuantizedContinuousInterval>()
            .expect("expected QuantizedContinuousInterval");
        assert_eq!(qi.resolution, 0.5);
        assert_eq!(qi.low_idx, i64::MIN);
        assert_eq!(qi.high_idx, i64::MAX);
    }

    #[test]
    fn quantized_plugin_creates_bounded_target_rule() {
        let df = DataFrame::new(vec![Column::new(
            "target__y1".into(),
            &[10.0_f64, 20.0, 30.0],
        )])
        .unwrap();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("target__y1").unwrap();

        let plugin = QuantizedContinuousPlugin::new(1.0);
        let rule = plugin.default_rule(col, 0.1);

        let qi = rule
            .as_any()
            .downcast_ref::<QuantizedContinuousInterval>()
            .expect("expected QuantizedContinuousInterval");
        assert!(qi.low() <= 10.0, "low should be at or below min value");
        assert!(qi.high() >= 30.0, "high should be at or above max value");
    }

    #[test]
    fn quantized_plugin_logical_dtype() {
        let plugin = QuantizedContinuousPlugin::new(0.25);
        assert_eq!(plugin.logical_dtype(), LogicalDType::QuantizedContinuous);
    }
}
