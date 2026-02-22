//! Categorical dtype plugin.
//!
//! Handles `Enum` and `Categorical` Polars columns. Scans all observed
//! category codes and creates a [`BelongsTo`](crate::rules::BelongsTo) rule
//! with a sorted domain for deterministic behaviour.
use std::collections::HashSet;
use std::sync::Arc;

use crate::rules::BelongsTo;

use super::super::column_split::{CategoricalColumnSplitSearcher, ColumnSplitSearcher};
use super::super::dataset_view::{ColumnView, LogicalDType};
use super::super::rule::DynRule;
use super::DTypePlugin;

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
    /// Create a new `CategoricalPlugin`.
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

    fn default_rule(&self, col: &dyn ColumnView, _boundaries_expansion_factor: f64) -> Box<dyn DynRule> {
        // Scan for all observed category codes
        let mut seen = HashSet::new();
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

        let domain_names: Vec<String> = domain_order.iter().map(|c| format!("cat_{c}")).collect();
        let values: HashSet<usize> = domain_order.iter().copied().collect();

        Box::new(BelongsTo::new(
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
