//! Dtype plugin registry for extensible data-type support.
//!
//! A [`DTypePlugin`] encapsulates all dtype-specific logic:
//!
//! - Creating default (root) rules for a column of that type.
//! - Providing the appropriate [`ColumnSplitSearcher`](super::column_split::ColumnSplitSearcher).
//!
//! The [`DTypeRegistry`] maps [`LogicalDType`] → [`DTypePlugin`], replacing
//! the monolithic `match` blocks in v1’s `DtypeAdapter`.
//!
//! ## Built-in Plugins
//!
//! | Plugin               | Handles             | Default rule                  |
//! |----------------------|---------------------|-------------------------------|
//! | [`ContinuousPlugin`] | `Float64`, etc.     | Unbounded interval (features) or observed-range with expansion (targets) |
//! | [`CategoricalPlugin`]| `Enum`, `Categorical`| `BelongsTo` covering all observed codes |
//!
//! ## Extension
//!
//! To support a new dtype:
//!
//! 1. Implement [`DTypePlugin`] for your type.
//! 2. Register it via [`DTypeRegistry::register`].
//! 3. Provide a matching [`ColumnSplitSearcher`](super::column_split::ColumnSplitSearcher).
use std::collections::HashMap;
use std::sync::Arc;

use crate::conf::TARGET_PREFIX;
use crate::rules::{BelongsTo, ContinuousInterval};

use super::column_split::{
    CategoricalColumnSplitSearcher, ColumnSplitSearcher, ContinuousColumnSplitSearcher,
};
use super::dataset_view::{ColumnView, LogicalDType};
use super::rule::RuleType;

// ---------------------------------------------------------------------------
// DTypePlugin trait
// ---------------------------------------------------------------------------

/// Extension point for adding new data types to the partition tree.
///
/// Each plugin knows how to:
///
/// 1. **Create a default (root) rule** for a column of its type
///    (e.g., an interval spanning the observed range for continuous, or a
///    `BelongsTo` covering all observed categories).
/// 2. **Return the [`ColumnSplitSearcher`](super::column_split::ColumnSplitSearcher)**
///    used to find splits on that type.
///
/// # Thread Safety
///
/// Must be `Send + Sync` because the registry is shared via `Arc` across
/// parallel split searches.
pub trait DTypePlugin: Send + Sync {
    /// The logical dtype this plugin handles.
    fn logical_dtype(&self) -> LogicalDType;

    /// Create a default (root) rule for a column.
    ///
    /// - **Continuous features**: unbounded interval $(-\infty, +\infty)$.
    /// - **Continuous targets**: interval spanning `[min, max]` of observed
    ///   values, expanded by `boundaries_expansion_factor`.
    /// - **Categorical**: a `BelongsTo` rule covering all observed category codes.
    fn default_rule(
        &self,
        col: &dyn ColumnView,
        boundaries_expansion_factor: f64,
    ) -> RuleType;

    /// The column-level split searcher for this dtype.
    fn split_searcher(&self) -> &dyn ColumnSplitSearcher;
}

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

    fn default_rule(
        &self,
        col: &dyn ColumnView,
        boundaries_expansion_factor: f64,
    ) -> RuleType {
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

            RuleType::Continuous(ContinuousInterval::new(
                low,
                high,
                true,
                true,
                Some((low, high)),
                true,
            ))
        } else {
            // For feature columns: unbounded interval
            RuleType::Continuous(ContinuousInterval::new(
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
// CategoricalPlugin
// ---------------------------------------------------------------------------

/// Plugin for categorical (`Enum`, `Categorical`) columns.
///
/// Scans for all observed category codes and creates a `BelongsTo` rule
/// with a sorted domain for deterministic behaviour.
pub struct CategoricalPlugin {
    searcher: CategoricalColumnSplitSearcher,
}

impl CategoricalPlugin {
    pub fn new() -> Self {
        Self {
            searcher: CategoricalColumnSplitSearcher,
        }
    }
}

impl Default for CategoricalPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DTypePlugin for CategoricalPlugin {
    fn logical_dtype(&self) -> LogicalDType {
        LogicalDType::Categorical
    }

    fn default_rule(
        &self,
        col: &dyn ColumnView,
        _boundaries_expansion_factor: f64,
    ) -> RuleType {
        // Scan for all observed category codes
        let mut seen = std::collections::HashSet::new();
        let mut domain_order = Vec::new();
        for i in 0..col.len() {
            if let Some(cat) = col.get_cat(i) {
                if seen.insert(cat) {
                    domain_order.push(cat);
                }
            }
        }

        // Sort domain for determinism
        domain_order.sort();

        let domain_names: Vec<String> = domain_order
            .iter()
            .map(|c| format!("cat_{c}"))
            .collect();
        let values: std::collections::HashSet<usize> =
            domain_order.iter().copied().collect();

        RuleType::BelongsTo(BelongsTo::new(
            values,
            Arc::new(domain_order),
            Arc::new(domain_names),
            true,
        ))
    }

    fn split_searcher(&self) -> &dyn ColumnSplitSearcher {
        &self.searcher
    }
}

// ---------------------------------------------------------------------------
// DTypeRegistry
// ---------------------------------------------------------------------------

/// Maps [`LogicalDType`] → [`DTypePlugin`].
///
/// Use [`DTypeRegistry::default()`] to get a registry with built-in plugins
/// for [`Continuous`](ContinuousPlugin) and [`Categorical`](CategoricalPlugin)
/// types. Register custom plugins via [`DTypeRegistry::register`].
pub struct DTypeRegistry {
    plugins: HashMap<LogicalDType, Box<dyn DTypePlugin>>,
}

impl DTypeRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a plugin. Replaces any existing plugin for the same dtype.
    pub fn register(&mut self, plugin: Box<dyn DTypePlugin>) {
        self.plugins.insert(plugin.logical_dtype(), plugin);
    }

    /// Get the plugin for a logical dtype.
    pub fn get(&self, dtype: LogicalDType) -> Option<&dyn DTypePlugin> {
        self.plugins.get(&dtype).map(|p| p.as_ref())
    }

    /// Get the plugin for a logical dtype, panicking if not found.
    pub fn get_or_panic(&self, dtype: LogicalDType) -> &dyn DTypePlugin {
        self.get(dtype)
            .unwrap_or_else(|| panic!("No DTypePlugin registered for {dtype:?}"))
    }
}

impl Default for DTypeRegistry {
    /// Create a registry with built-in plugins for Continuous and Categorical.
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(ContinuousPlugin::new()));
        reg.register(Box::new(CategoricalPlugin::new()));
        reg
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::dataset_view::{DatasetView, PolarsDatasetView};
    use polars::prelude::*;

    #[test]
    fn default_registry_has_both_plugins() {
        let reg = DTypeRegistry::default();
        assert!(reg.get(LogicalDType::Continuous).is_some());
        assert!(reg.get(LogicalDType::Categorical).is_some());
    }

    #[test]
    fn continuous_plugin_creates_unbounded_feature_rule() {
        let df = DataFrame::new(vec![
            Column::new("x1".into(), &[1.0_f64, 2.0, 3.0]),
        ])
        .unwrap();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("x1").unwrap();

        let plugin = ContinuousPlugin::new();
        let rule = plugin.default_rule(col, 0.1);

        match &rule {
            RuleType::Continuous(ci) => {
                assert!(ci.low.is_infinite());
                assert!(ci.high.is_infinite());
            }
            _ => panic!("expected Continuous rule"),
        }
    }

    #[test]
    fn continuous_plugin_creates_bounded_target_rule() {
        let df = DataFrame::new(vec![
            Column::new("target__y1".into(), &[10.0_f64, 20.0, 30.0]),
        ])
        .unwrap();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("target__y1").unwrap();

        let plugin = ContinuousPlugin::new();
        let rule = plugin.default_rule(col, 0.1);

        match &rule {
            RuleType::Continuous(ci) => {
                assert!(ci.low < 10.0, "low should be below min value");
                assert!(ci.high > 30.0, "high should be above max value");
                assert!(ci.low.is_finite());
                assert!(ci.high.is_finite());
            }
            _ => panic!("expected Continuous rule"),
        }
    }
}
