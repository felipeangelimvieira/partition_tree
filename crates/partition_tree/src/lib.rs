//! # Partition Tree
//!
//! A trait-driven, extensible partition tree implementation that models
//! conditional density estimation via recursive space partitioning.
//!
//! ## Overview
//!
//! The library is built around four key extension points:
//!
//! | Trait                    | Purpose                                          |
//! |--------------------------|--------------------------------------------------|
//! | [`LossFunc`]             | Defines split gain (conditional, balanced, custom)|
//! | [`ColumnSplitSearcher`]  | Per-dtype split algorithm                        |
//! | [`DTypePlugin`]          | Dtype-specific rules and searchers               |
//! | [`DatasetView`]          | Backend-agnostic tabular data access             |
//!
//! ## Architecture
//!
//! ```text
//!   DatasetView (Polars impl)
//!        │
//!        ▼
//!   TreeBuilder ──► SplitSearcher ──► DTypeRegistry
//!        │               │                  │
//!        │               ▼                  ▼
//!        │          ColumnSplitSearcher   DTypePlugin
//!        │          (Continuous / Cat.)   (rules, searchers)
//!        │
//!        ▼
//!   Tree (FittedNode arena + leaves + split history)
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use partition_tree::{
//!     TreeBuilder, TreeBuilderConfig, SplitRestrictions,
//!     ConditionalLogLoss, DTypeRegistry, PolarsDatasetView,
//! };
//!
//! // 1. Wrap your Polars DataFrame
//! let dataset = PolarsDatasetView::new(&df);
//!
//! // 2. Configure and build
//! let config = TreeBuilderConfig {
//!     max_leaves: 8,
//!     boundaries_expansion_factor: 0.0,
//!     restrictions: SplitRestrictions { max_depth: 5, ..Default::default() },
//! };
//! let loss = Box::new(ConditionalLogLoss);
//! let registry = Arc::new(DTypeRegistry::default());
//! let tree = TreeBuilder::new(config, loss, registry).build(&dataset);
//!
//! // 3. Inspect results
//! println!("{tree}");
//! ```

// ── Core modules ────────────────────────────────────────────────────────────

pub mod cell;
pub mod column_split;
pub mod conf;
pub mod dataset_view;
pub mod dtype_plugin;
pub mod estimators;
pub mod loss;
pub mod node;
pub mod predict;
pub mod rule;
pub mod rules;
pub mod serde;
pub mod split_result;
pub mod split_searcher;
pub mod tree;
pub mod tree_builder;

// ── Re-exports for convenience ──────────────────────────────────────────────

/// Loss function trait and built-in implementations.
pub use loss::{
    BalancedLogLoss, CellStats, ConditionalLogLoss, LossFunc, MeanIntegratedSquaredError,
};

/// Multi-dimensional partition constraint.
pub use cell::Cell;

/// Rule types and dtype-erased rule trait.
pub use rule::{DynRule, DynValue};

/// Split result types, restrictions, and priority-queue entry.
pub use split_result::{
    CandidateSplit, CategoricalSplitOp, ContinuousSplitOp, IntegerSplitOp, SplitKind, SplitOp,
    SplitPoint, SplitRestrictions,
};

/// Dataset abstraction and Polars-backed implementation.
pub use dataset_view::{
    ColumnView, DatasetView, LogicalDType, PolarsColumnView, PolarsDatasetView,
};

/// Construction-time tree node.
pub use node::Node;

/// Per-column split search trait and built-in implementations.
pub use column_split::{
    CategoricalColumnSplitSearcher, ColumnSplitSearcher, ContinuousColumnSplitSearcher,
    IntegerColumnSplitSearcher,
};

/// Dtype plugin trait and registry.
pub use dtype_plugin::{
    CategoricalPlugin, ContinuousPlugin, DTypePlugin, DTypeRegistry, IntegerPlugin,
};

/// Multi-column split orchestrator.
pub use split_searcher::SplitSearcher;

/// Tree builder configuration and best-first construction.
pub use tree_builder::{TreeBuilder, TreeBuilderConfig};

/// Fitted tree, nodes, split records, and leaf summaries.
pub use tree::{FittedNode, LeafInfo, SplitRecord, Tree};

/// Prediction types: conditioned cells, piecewise distributions, and mean vectors.
pub use predict::{ConditionedCell, MeanVector, PiecewiseConstantDistribution};

/// Estimator wrapper (implements `estimators::api::Estimator`).
pub use estimators::PartitionTree;

/// Forest estimator (implements `estimators::api::Estimator`).
pub use estimators::PartitionForest;
