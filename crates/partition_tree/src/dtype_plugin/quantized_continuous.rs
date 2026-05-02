//! Quantized-continuous dtype plugin.
//!
//! Handles `f64` columns by snapping values to bins centered on `resolution * i`.
//! Rule and split semantics match the integer dtype on that lattice while
//! preserving `f64` values in the public API.
use crate::conf::TARGET_PREFIX;
use crate::rules::QuantizedContinuousInterval;

use super::DTypePlugin;
use crate::column_split::{ColumnSplitSearcher, QuantizedContinuousColumnSplitSearcher};
use crate::dataset_view::{ColumnView, LogicalDType, LogicalDTypeKind};
use crate::rule::DynRule;

/// Plugin for quantized-continuous columns.
pub struct QuantizedContinuousPlugin {
    searcher: QuantizedContinuousColumnSplitSearcher,
}

impl QuantizedContinuousPlugin {
    /// Create a new quantized-continuous plugin.
    pub fn new() -> Self {
        Self {
            searcher: QuantizedContinuousColumnSplitSearcher,
        }
    }

    fn resolution_for_column(col: &dyn ColumnView) -> f64 {
        match col.logical_dtype() {
            LogicalDType::QuantizedContinuous(spec) => spec.resolution(),
            dtype => panic!("QuantizedContinuousPlugin requires a quantized column, got {dtype:?}"),
        }
    }

    fn scan_lattice_bounds(col: &dyn ColumnView, resolution: f64) -> Option<(i64, i64)> {
        let mut min_idx = i64::MAX;
        let mut max_idx = i64::MIN;

        for i in 0..col.len() {
            let Some(value) = col.get_f64(i) else {
                continue;
            };

            let idx = QuantizedContinuousInterval::quantize_value(value, resolution)
                .unwrap_or_else(|_| {
                    panic!(
                        "Column '{}' uses QuantizedContinuous(resolution={}) but value {} could not be quantized",
                        col.name(),
                        resolution,
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

    fn floor_to_i64(value: f64) -> i64 {
        value.floor().clamp(i64::MIN as f64, i64::MAX as f64) as i64
    }

    fn ceil_to_i64(value: f64) -> i64 {
        value.ceil().clamp(i64::MIN as f64, i64::MAX as f64) as i64
    }

    fn expand_target_bounds_by_magnitude(
        min_idx: i64,
        max_idx: i64,
        resolution: f64,
        boundaries_expansion_factor: f64,
    ) -> (i64, i64) {
        let min_val = min_idx as f64 * resolution;
        let max_val = max_idx as f64 * resolution;
        let range = max_val - min_val;
        let mean = (max_val + min_val) / 2.0;
        let expanded_width = range * (1.0 + boundaries_expansion_factor);
        let low = mean - expanded_width / 2.0;
        let high = mean + expanded_width / 2.0;

        // Keep the quantized rule lattice-aligned, but derive the enclosing
        // bounds from value-space magnitude rather than whole extra steps.
        let low_idx = Self::floor_to_i64(low / resolution + 0.5);
        let high_idx = Self::ceil_to_i64(high / resolution - 0.5);
        (low_idx, high_idx)
    }
}

impl Default for QuantizedContinuousPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DTypePlugin for QuantizedContinuousPlugin {
    fn logical_dtype_kind(&self) -> LogicalDTypeKind {
        LogicalDTypeKind::QuantizedContinuous
    }

    fn default_rule(
        &self,
        col: &dyn ColumnView,
        boundaries_expansion_factor: f64,
    ) -> Box<dyn DynRule> {
        let resolution = Self::resolution_for_column(col);
        let is_target = col.name().starts_with(TARGET_PREFIX);
        let bounds = Self::scan_lattice_bounds(col, resolution);

        if is_target {
            let (min_idx, max_idx) = bounds.unwrap_or((0, 1));
            let (low_idx, high_idx) = Self::expand_target_bounds_by_magnitude(
                min_idx,
                max_idx,
                resolution,
                boundaries_expansion_factor,
            );

            Box::new(QuantizedContinuousInterval::new(
                low_idx,
                high_idx,
                resolution,
                Some((low_idx, high_idx)),
                true,
            ))
        } else {
            Box::new(QuantizedContinuousInterval::new(
                i64::MIN,
                i64::MAX,
                resolution,
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

    fn quantized_view(df: &DataFrame) -> PolarsDatasetView {
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "x1".to_string(),
            LogicalDType::quantized_continuous(0.5).unwrap(),
        );
        PolarsDatasetView::try_with_dtype_overrides(df, &overrides).unwrap()
    }

    #[test]
    fn quantized_plugin_creates_unbounded_feature_rule() {
        let df = DataFrame::new(vec![Column::new("x1".into(), &[0.0_f64, 0.5, 1.0])]).unwrap();
        let view = quantized_view(&df);
        let col = view.column("x1").unwrap();

        let plugin = QuantizedContinuousPlugin::new();
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
            &[10.0_f64, 10.5, 11.0],
        )])
        .unwrap();

        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "target__y1".to_string(),
            LogicalDType::quantized_continuous(0.5).unwrap(),
        );
        let view = PolarsDatasetView::try_with_dtype_overrides(&df, &overrides).unwrap();
        let col = view.column("target__y1").unwrap();

        let plugin = QuantizedContinuousPlugin::new();
        let rule = plugin.default_rule(col, 0.1);

        let qi = rule
            .as_any()
            .downcast_ref::<QuantizedContinuousInterval>()
            .expect("expected QuantizedContinuousInterval");
        assert!(qi.low() <= 10.0, "low should be at or below min value");
        assert!(qi.high() >= 11.0, "high should be at or above max value");
    }

    #[test]
    fn quantized_plugin_expands_targets_by_value_magnitude() {
        let df = DataFrame::new(vec![Column::new(
            "target__y1".into(),
            &[10.0_f64, 10.5, 11.0],
        )])
        .unwrap();

        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "target__y1".to_string(),
            LogicalDType::quantized_continuous(0.5).unwrap(),
        );
        let view = PolarsDatasetView::try_with_dtype_overrides(&df, &overrides).unwrap();
        let col = view.column("target__y1").unwrap();

        let plugin = QuantizedContinuousPlugin::new();
        let rule = plugin.default_rule(col, 0.1);

        let qi = rule
            .as_any()
            .downcast_ref::<QuantizedContinuousInterval>()
            .expect("expected QuantizedContinuousInterval");
        assert_eq!(
            qi.low_idx, 20,
            "small value-space expansion should not add a full extra bin on the left"
        );
        assert_eq!(
            qi.high_idx, 22,
            "small value-space expansion should not add a full extra bin on the right"
        );
        assert!((qi.low() - 9.75).abs() < 1e-10);
        assert!((qi.high() - 11.25).abs() < 1e-10);
    }

    #[test]
    fn quantized_plugin_logical_dtype_kind() {
        let plugin = QuantizedContinuousPlugin::new();
        assert_eq!(
            plugin.logical_dtype_kind(),
            LogicalDTypeKind::QuantizedContinuous
        );
    }
}
