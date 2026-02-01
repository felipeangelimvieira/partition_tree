//! # Partition Tree
//!
//! A Rust library for working with rules and decision trees.
//!
//! ## Key Features
//!
//! - **Generic BelongsTo Rule**: Optimized categorical rule with O(1) membership tests
//! - **Memory Efficient**: Shared ordered domain storage via `Arc<Vec<T>>`
//! - **Type Safe**: Generic rules work with any hashable type
//!
//! ## Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use partition_tree::{rules::BelongsTo, rules::Rule};
//!
//! // Create a shared ordered domain for memory efficiency (using usize indices for categorical data)
//! let domain = Arc::new((0..4).collect::<Vec<usize>>());
//! let domain_names = Arc::new(["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect::<Vec<_>>());
//! // Active categories supplied in insertion order
//! let rule = BelongsTo::new([0usize, 2usize].into_iter(), Arc::clone(&domain), Arc::clone(&domain_names), false);
//!
//! // Fast O(1) membership testing on categorical indices (preserves order)
//! let data = vec![Some(0), Some(1), Some(2), Some(3)];
//! let results = rule.evaluate(&data); // [true, false, true, false]
//! ```

// Module declarations
pub mod cell;
pub mod conf;
pub mod dataframe;
pub mod density;
pub mod dtype_adapter;
pub mod estimator;
pub mod node;
pub mod onedimpartition;
pub mod predict;
pub mod rules;
pub mod split;
pub mod tree;
pub mod estimator_forest;