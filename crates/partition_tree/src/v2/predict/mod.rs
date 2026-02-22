//! Prediction module for v2 partition trees.
//!
//! This module provides the types and methods for conditional prediction
//! from fitted partition trees:
//!
//! | Type                              | Purpose                                       |
//! |-----------------------------------|-----------------------------------------------|
//! | [`ConditionedCell`]               | Leaf cell projected to target space + mass     |
//! | [`PiecewiseConstantDistribution`] | Piecewise-constant conditional distribution    |
//! | [`MeanVector`]                    | Per-target-column mean (type alias)            |
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Single-tree prediction
//! let dists = tree.predict_distributions(&dataset);
//! let means = tree.predict_mean_vectors(&dataset);
//! let df    = tree.predict_mean(&dataset);
//!
//! // Ensemble prediction
//! let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);
//! let ens_mean = ens.mean_vector();
//! ```
pub mod conditioned_cell;
pub mod piecewise_distribution;

pub use conditioned_cell::ConditionedCell;
pub use piecewise_distribution::{MeanVector, PiecewiseConstantDistribution};
