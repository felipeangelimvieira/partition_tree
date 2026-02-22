//! Dtype plugin registry for extensible data-type support.
//!
//! A [`DTypePlugin`] encapsulates all dtype-specific logic:
//!
//! - Creating default (root) rules for a column of that type.
//! - Providing the appropriate [`ColumnSplitSearcher`].
//!
//! The [`DTypeRegistry`] maps [`LogicalDType`] → [`DTypePlugin`], replacing
//! the monolithic `match` blocks in v1's `DtypeAdapter`.
//!
//! ## Built-in Plugins
//!
//! | Plugin               | Handles              | Default rule                                                              |
//! |----------------------|----------------------|---------------------------------------------------------------------------|
//! | [`ContinuousPlugin`] | `Float64`, etc.      | Unbounded interval (features) or observed-range with expansion (targets)  |
//! | [`CategoricalPlugin`]| `Enum`, `Categorical`| `BelongsTo` covering all observed codes                                   |
//!
//! ## Extension
//!
//! To support a new dtype:
//!
//! 1. Implement [`DTypePlugin`] for your type.
//! 2. Register it via [`DTypeRegistry::register`].
//! 3. Provide a matching [`ColumnSplitSearcher`].
pub mod categorical;
pub mod continuous;
pub mod integer;

pub use categorical::CategoricalPlugin;
pub use continuous::ContinuousPlugin;
pub use integer::IntegerPlugin;

use std::collections::HashMap;

use crate::column_split::ColumnSplitSearcher;
use crate::dataset_view::{ColumnView, LogicalDType};
use crate::rule::DynRule;

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
/// 2. **Return the [`ColumnSplitSearcher`]**
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
    fn default_rule(&self, col: &dyn ColumnView, boundaries_expansion_factor: f64) -> Box<dyn DynRule>;

    /// The column-level split searcher for this dtype.
    fn split_searcher(&self) -> &dyn ColumnSplitSearcher;
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
    /// Create a registry with built-in plugins for Continuous, Categorical, and Integer.
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(ContinuousPlugin::new()));
        reg.register(Box::new(CategoricalPlugin::new()));
        reg.register(Box::new(IntegerPlugin::new()));
        reg
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_all_plugins() {
        let reg = DTypeRegistry::default();
        assert!(reg.get(LogicalDType::Continuous).is_some());
        assert!(reg.get(LogicalDType::Categorical).is_some());
        assert!(reg.get(LogicalDType::Integer).is_some());
    }
}
