//! # Partition Tree v2
//!
//! A trait-driven, extensible partition tree implementation that models
//! conditional density estimation via recursive space partitioning.
//!
//! ## Overview
//!
//! The v2 module replaces the monolithic v1 tree with a composable architecture
//! built around four key extension points:
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
//! ## Partition-Tree Model
//!
//! The partition tree operates on a product space $X \times Y$ with three
//! measures:
//!
//! - **XY-measure** ($\mu_{XY}$): the joint empirical distribution.
//! - **X-measure** ($\mu_X$): the feature marginal.
//! - **Y-measure** ($\mu_Y$): the target marginal (geometric volume or counting measure).
//!
//! Each split is classified as:
//!
//! - **XSplit** (feature split): refines the feature constraint $A_X$;
//!   the target constraint $A_Y$ is unchanged. Both children inherit the
//!   parent's full `sorted_y`.
//! - **YSplit** (target split): refines the target constraint $A_Y$;
//!   the feature constraint $A_X$ is unchanged. Both children inherit the
//!   parent's full `sorted_x`.
//!
//! This distinction is critical for correct weight propagation—see
//! [`Node::propagate_children`].
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use partition_tree::v2::{
//!     TreeBuilder, TreeBuilderConfig, SplitRestrictions,
//!     ConditionalLogLoss, DTypeRegistry, PolarsDatasetView,
//! };
//!
//! // 1. Wrap your Polars DataFrame
//! // let dataset = PolarsDatasetView::new(&df);
//!
//! // 2. Configure and build
//! // let config = TreeBuilderConfig {
//! //     max_leaves: 8,
//! //     boundaries_expansion_factor: 0.0,
//! //     restrictions: SplitRestrictions { max_depth: 5, ..Default::default() },
//! // };
//! // let loss = Box::new(ConditionalLogLoss::new(dataset.n_rows() as f64));
//! // let registry = Arc::new(DTypeRegistry::default());
//! // let tree = TreeBuilder::new(config, loss, registry).build(&dataset);
//!
//! // 3. Inspect results
//! // println!("{tree}");
//! ```
//!
//! ## Modules
//!
//! | Module              | Description                                                  |
//! |---------------------|--------------------------------------------------------------|
//! | [`loss`]            | Loss functions and gain computation                          |
//! | [`rule`]            | Rule types with convenience methods on v1 rules              |
//! | [`cell`]            | Multi-dimensional partition constraint (conjunction of rules)|
//! | [`split_result`]    | Split descriptors, restrictions, and candidate heap entry    |
//! | [`dataset_view`]    | Trait abstraction over tabular data (Polars implementation)  |
//! | [`node`]            | Construction-time node with presorted index maps             |
//! | [`column_split`]    | Per-column split search algorithms                           |
//! | [`dtype_plugin`]    | Dtype plugin registry for extensible data-type support       |
//! | [`split_searcher`]  | Orchestrator dispatching to column-level searchers           |
//! | [`tree_builder`]    | Best-first tree construction loop                            |
//! | [`tree`]            | Fitted tree with prediction and display APIs                 |
//! | [`predict`]         | Conditional prediction: distributions, means, ensembling     |
//! | [`estimator`]       | Estimator trait implementation for v2 tree                   |

pub mod cell;
pub mod column_split;
pub mod dataset_view;
pub mod dtype_plugin;
pub mod estimator;
pub mod forest;
pub mod loss;
pub mod node;
pub mod predict;
pub mod rule;
#[path = "serde/mod.rs"]
pub mod serde_support;
pub mod split_result;
pub mod split_searcher;
pub mod tree;
pub mod tree_builder;

// ── Re-exports for convenience ──────────────────────────────────────────────

/// Loss function trait and built-in implementations.
pub use loss::{BalancedLogLoss, CellStats, ConditionalLogLoss, LossFunc};

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

/// Estimator wrapper for v2 tree (implements `estimators::api::Estimator`).
pub use estimator::PartitionTreeV2;

/// Forest estimator for v2 trees (implements `estimators::api::Estimator`).
pub use forest::PartitionForestV2;
