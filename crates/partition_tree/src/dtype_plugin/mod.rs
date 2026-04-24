//! Dtype plugin registry for extensible data-type support.
//!
//! A [`DTypePlugin`] encapsulates all dtype-specific logic:
//!
//! - Creating default (root) rules for a column of that type.
//! - Providing the appropriate [`ColumnSplitSearcher`].
//!
//! The [`DTypeRegistry`] maps [`LogicalDTypeKind`] → [`DTypePlugin`], replacing
//! the monolithic `match` blocks in v1's `DtypeAdapter`.
//!
//! ## Built-in Plugins
//!
//! | Plugin                        | Handles                       | Default rule                                                              |
//! |-------------------------------|-------------------------------|---------------------------------------------------------------------------|
//! | [`ContinuousPlugin`]          | `Float64`, etc.               | Unbounded interval (features) or observed-range with expansion (targets)  |
//! | [`CategoricalPlugin`]         | `Enum`, `Categorical`         | `BelongsTo` covering all observed codes                                   |
//! | [`IntegerPlugin`]             | `Int32`, `Int64`              | Integer interval over the observed or full integer domain                 |
//! | [`QuantizedContinuousPlugin`] | Quantized real-valued columns | Lattice interval over `resolution * i`                                    |
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
pub mod quantized_continuous;

pub use categorical::CategoricalPlugin;
pub use continuous::ContinuousPlugin;
pub use integer::IntegerPlugin;
pub use quantized_continuous::QuantizedContinuousPlugin;

use std::collections::HashMap;

use crate::column_split::ColumnSplitSearcher;
use crate::dataset_view::{ColumnView, LogicalDType, LogicalDTypeKind};
use crate::rule::DynRule;

/// Extension point for adding new data types to the partition tree.
pub trait DTypePlugin: Send + Sync {
    /// The logical dtype kind this plugin handles.
    fn logical_dtype_kind(&self) -> LogicalDTypeKind;

    /// Create a default (root) rule for a column.
    fn default_rule(
        &self,
        col: &dyn ColumnView,
        boundaries_expansion_factor: f64,
    ) -> Box<dyn DynRule>;

    /// The column-level split searcher for this dtype.
    fn split_searcher(&self) -> &dyn ColumnSplitSearcher;
}

/// Maps [`LogicalDTypeKind`] → [`DTypePlugin`].
pub struct DTypeRegistry {
    plugins: HashMap<LogicalDTypeKind, Box<dyn DTypePlugin>>,
}

impl DTypeRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a plugin. Replaces any existing plugin for the same dtype kind.
    pub fn register(&mut self, plugin: Box<dyn DTypePlugin>) {
        self.plugins.insert(plugin.logical_dtype_kind(), plugin);
    }

    /// Get the plugin for a logical dtype.
    pub fn get(&self, dtype: LogicalDType) -> Option<&dyn DTypePlugin> {
        self.plugins.get(&dtype.kind()).map(|p| p.as_ref())
    }

    /// Get the plugin for a logical dtype, panicking if not found.
    pub fn get_or_panic(&self, dtype: LogicalDType) -> &dyn DTypePlugin {
        self.get(dtype)
            .unwrap_or_else(|| panic!("No DTypePlugin registered for {dtype:?}"))
    }
}

impl Default for DTypeRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(ContinuousPlugin::new()));
        reg.register(Box::new(CategoricalPlugin::new()));
        reg.register(Box::new(IntegerPlugin::new()));
        reg.register(Box::new(QuantizedContinuousPlugin::new()));
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_all_plugins() {
        let reg = DTypeRegistry::default();
        assert!(reg.get(LogicalDType::Continuous).is_some());
        assert!(reg.get(LogicalDType::Categorical).is_some());
        assert!(reg.get(LogicalDType::Integer).is_some());
        assert!(reg
            .get(LogicalDType::quantized_continuous(1.0).unwrap())
            .is_some());
    }
}
