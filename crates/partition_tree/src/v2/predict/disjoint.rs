//! Disjoint resolution for one-dimensional piecewise-constant distributions.
//!
//! After ensembling, cells from different trees may overlap in target space.
//! [`DisjointResolvable`] provides a uniform interface to collect boundary
//! points and create sub-rules for each disjoint fragment, enabling a
//! sweep-line algorithm that converts overlapping cells into a proper
//! partition where each target-space point belongs to exactly one fragment.
//!
//! ## Supported rule types
//!
//! | Type                  | Fragment semantics             |
//! |-----------------------|--------------------------------|
//! | [`ContinuousInterval`]| Sub-intervals between boundary endpoints |
//! | [`IntegerInterval`]   | Sub-intervals between integer boundary points |
//! | [`BelongsTo`]         | Individual categories as unit bins |
//!
//! ## Complexity guard
//!
//! Only **one-dimensional** target spaces (a single target column) are
//! supported. Multi-dimensional disjoint resolution would require computing
//! the arrangement of axis-aligned hyper-rectangles, whose complexity is
//! exponential in the number of dimensions. Call
//! [`PiecewiseConstantDistribution::resolve_disjoint`](super::piecewise_distribution::PiecewiseConstantDistribution::resolve_disjoint)
//! for the public entry point, which validates dimensionality before
//! dispatching.

use std::fmt;

use crate::rules::{BelongsTo, ContinuousInterval, IntegerInterval};
use crate::v2::rule::DynRule;

// ---------------------------------------------------------------------------
// DisjointResolvable trait
// ---------------------------------------------------------------------------

/// A one-dimensional rule that can be decomposed into disjoint fragments.
///
/// Each implementation emits **boundary points** (sorted, deduplicated by the
/// caller) and can produce a **sub-rule** for any fragment defined by a pair
/// of consecutive boundary points.
///
/// # Design notes
///
/// This trait is deliberately *not* added to [`DynRule`] to keep the core
/// trait minimal. Instead, concrete types implement it and a free-standing
/// [`as_disjoint_resolvable`] function downcasts from `&dyn DynRule`.
pub trait DisjointResolvable: fmt::Debug {
    /// Emit the boundary points that define this rule's extent.
    ///
    /// - **Continuous**: `[low, high]`
    /// - **Integer**: `[low, high + 1]` (half-open representation)
    /// - **Categorical**: `[0, 1, 2, …, domain_size]` (each category is a
    ///   unit-width bin `[i, i+1)`)
    fn boundaries(&self) -> Vec<f64>;

    /// Create a sub-rule covering the fragment `[frag_low, frag_high)`.
    ///
    /// Returns `None` if this rule does not overlap the fragment (e.g., the
    /// category index falls outside the active set).
    fn sub_rule(&self, frag_low: f64, frag_high: f64) -> Option<Box<dyn DynRule>>;

    /// Whether a scalar probe point falls within this rule's region.
    ///
    /// Used during the sweep to decide which source cells contribute density
    /// to a given fragment. The point is typically the midpoint of a fragment.
    fn contains_point(&self, point: f64) -> bool;

    /// Fraction of a cell's mass that should be assigned to a fragment.
    ///
    /// For **continuous/integer** rules the mass is distributed uniformly
    /// across the rule's volume, so the fraction is
    /// `fragment_width / rule_width`.
    ///
    /// For **categorical** rules the `mean()` vector already encodes which
    /// categories are active (one-hot indicator), so each per-category
    /// fragment receives the **full** cell mass (fraction `1.0`). The
    /// `mean_vector()` aggregation then naturally produces correct
    /// probabilities.
    fn mass_fraction(&self, frag_low: f64, frag_high: f64) -> f64;
}

// ---------------------------------------------------------------------------
// ContinuousInterval
// ---------------------------------------------------------------------------

impl DisjointResolvable for ContinuousInterval {
    fn boundaries(&self) -> Vec<f64> {
        vec![self.low, self.high]
    }

    fn sub_rule(&self, frag_low: f64, frag_high: f64) -> Option<Box<dyn DynRule>> {
        // Fragment must overlap this interval
        let lo = frag_low.max(self.low);
        let hi = frag_high.min(self.high);
        if hi <= lo {
            return None;
        }
        Some(Box::new(ContinuousInterval::new(
            lo,
            hi,
            true,  // lower closed
            false, // upper open — disjoint fragments are half-open [lo, hi)
            Some(self.domain),
            self.accept_none,
        )))
    }

    fn contains_point(&self, point: f64) -> bool {
        point >= self.low && point < self.high
    }

    fn mass_fraction(&self, frag_low: f64, frag_high: f64) -> f64 {
        let cell_vol = self.high - self.low;
        if cell_vol <= 0.0 {
            return 0.0;
        }
        (frag_high - frag_low) / cell_vol
    }
}

// ---------------------------------------------------------------------------
// IntegerInterval
// ---------------------------------------------------------------------------

impl DisjointResolvable for IntegerInterval {
    fn boundaries(&self) -> Vec<f64> {
        // Represent as half-open [low, high+1) so that consecutive integers
        // produce contiguous fragments of width 1.
        vec![self.low as f64, (self.high + 1) as f64]
    }

    fn sub_rule(&self, frag_low: f64, frag_high: f64) -> Option<Box<dyn DynRule>> {
        // Convert back to inclusive integer bounds
        let lo = (frag_low.ceil()) as i64;
        let hi = (frag_high.ceil()) as i64 - 1; // half-open → inclusive
        let lo = lo.max(self.low);
        let hi = hi.min(self.high);
        if hi < lo {
            return None;
        }
        Some(Box::new(IntegerInterval::new(
            lo,
            hi,
            Some(self.domain),
            self.accept_none,
        )))
    }

    fn contains_point(&self, point: f64) -> bool {
        let int_val = point.floor() as i64;
        int_val >= self.low && int_val <= self.high
    }

    fn mass_fraction(&self, frag_low: f64, frag_high: f64) -> f64 {
        let cell_vol = (self.high - self.low + 1) as f64;
        if cell_vol <= 0.0 {
            return 0.0;
        }
        (frag_high - frag_low) / cell_vol
    }
}

// ---------------------------------------------------------------------------
// BelongsTo (categorical)
// ---------------------------------------------------------------------------

impl DisjointResolvable for BelongsTo {
    fn boundaries(&self) -> Vec<f64> {
        // Each category index `i` occupies the unit bin `[i, i+1)`.
        // Emit boundaries for the full domain so all categories are covered.
        (0..=self.domain.len()).map(|i| i as f64).collect()
    }

    fn sub_rule(&self, frag_low: f64, frag_high: f64) -> Option<Box<dyn DynRule>> {
        // Each fragment should span exactly one category: [i, i+1).
        let cat_idx = frag_low.floor() as usize;
        let width = frag_high - frag_low;
        if (width - 1.0).abs() > 0.5 {
            // Not a single-category fragment — skip
            return None;
        }
        if cat_idx >= self.domain.len() {
            return None;
        }
        // Check that this category is in the active set
        if !self.values.contains(&cat_idx) {
            return None;
        }
        let mut values = std::collections::HashSet::new();
        values.insert(cat_idx);
        Some(Box::new(BelongsTo::new(
            values,
            std::sync::Arc::clone(&self.domain),
            std::sync::Arc::clone(&self.domain_names),
            self.accept_none,
        )))
    }

    fn contains_point(&self, point: f64) -> bool {
        let cat_idx = point.floor() as usize;
        self.values.contains(&cat_idx)
    }

    fn mass_fraction(&self, _frag_low: f64, _frag_high: f64) -> f64 {
        // Categorical: each per-category fragment receives the full cell mass.
        // The one-hot `mean()` vector already encodes which categories are
        // active, so no density-based splitting is needed.
        1.0
    }
}

// ---------------------------------------------------------------------------
// Downcast helper
// ---------------------------------------------------------------------------

/// Attempt to downcast a `&dyn DynRule` to `&dyn DisjointResolvable`.
///
/// Returns `None` if the concrete type is not one of the supported rule types
/// (`ContinuousInterval`, `IntegerInterval`, `BelongsTo`).
///
/// # Example
///
/// ```rust,ignore
/// let rule: &dyn DynRule = &some_rule;
/// if let Some(dr) = as_disjoint_resolvable(rule) {
///     let bounds = dr.boundaries();
/// }
/// ```
pub fn as_disjoint_resolvable(rule: &dyn DynRule) -> Option<&dyn DisjointResolvable> {
    let any = rule.as_any();
    if let Some(ci) = any.downcast_ref::<ContinuousInterval>() {
        return Some(ci as &dyn DisjointResolvable);
    }
    if let Some(ii) = any.downcast_ref::<IntegerInterval>() {
        return Some(ii as &dyn DisjointResolvable);
    }
    if let Some(bt) = any.downcast_ref::<BelongsTo>() {
        return Some(bt as &dyn DisjointResolvable);
    }
    None
}

// ---------------------------------------------------------------------------
// DisjointError
// ---------------------------------------------------------------------------

/// Error type for disjoint resolution failures.
#[derive(Debug, Clone)]
pub enum DisjointError {
    /// The distribution has no cells to resolve.
    Empty,
    /// Multi-dimensional target space is not supported.
    MultiDimensional {
        /// Number of target dimensions found.
        n_dims: usize,
    },
    /// Cells have inconsistent target column names.
    InconsistentColumns {
        /// The expected column name (from the first cell).
        expected: String,
        /// The column name that was found in a later cell.
        found: String,
    },
    /// A rule type could not be downcast to `DisjointResolvable`.
    UnsupportedRuleType,
}

impl fmt::Display for DisjointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisjointError::Empty => write!(f, "distribution has no cells"),
            DisjointError::MultiDimensional { n_dims } => write!(
                f,
                "resolve_disjoint only supports 1-dimensional targets, found {n_dims} dimensions \
                 (multi-dimensional resolution has exponential complexity)"
            ),
            DisjointError::InconsistentColumns { expected, found } => write!(
                f,
                "inconsistent target columns: expected '{expected}', found '{found}'"
            ),
            DisjointError::UnsupportedRuleType => {
                write!(f, "rule type does not implement DisjointResolvable")
            }
        }
    }
}

impl std::error::Error for DisjointError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{ContinuousInterval, IntegerInterval};
    use std::collections::HashSet;
    use std::sync::Arc;

    // -- ContinuousInterval ------------------------------------------------

    #[test]
    fn continuous_boundaries() {
        let ci = ContinuousInterval::new(2.0, 8.0, true, false, Some((0.0, 10.0)), false);
        assert_eq!(ci.boundaries(), vec![2.0, 8.0]);
    }

    #[test]
    fn continuous_sub_rule_within() {
        let ci = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), false);
        let sub = ci.sub_rule(3.0, 7.0).unwrap();
        assert!((sub.volume() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn continuous_sub_rule_no_overlap() {
        let ci = ContinuousInterval::new(0.0, 4.0, true, false, Some((0.0, 10.0)), false);
        assert!(ci.sub_rule(5.0, 8.0).is_none());
    }

    #[test]
    fn continuous_contains_point() {
        let ci = ContinuousInterval::new(2.0, 6.0, true, false, Some((0.0, 10.0)), false);
        assert!(ci.contains_point(2.0));
        assert!(ci.contains_point(4.0));
        assert!(!ci.contains_point(6.0)); // upper-exclusive
        assert!(!ci.contains_point(1.0));
    }

    // -- IntegerInterval ---------------------------------------------------

    #[test]
    fn integer_boundaries() {
        let ii = IntegerInterval::new(2, 5, Some((0, 10)), false);
        // half-open: [2, 6)
        assert_eq!(ii.boundaries(), vec![2.0, 6.0]);
    }

    #[test]
    fn integer_sub_rule_single() {
        let ii = IntegerInterval::new(0, 9, Some((0, 9)), false);
        let sub = ii.sub_rule(3.0, 4.0).unwrap();
        // Should cover [3, 3] → volume = 1
        assert!((sub.volume() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn integer_sub_rule_range() {
        let ii = IntegerInterval::new(0, 9, Some((0, 9)), false);
        let sub = ii.sub_rule(2.0, 5.0).unwrap();
        // Covers [2, 4] → volume = 3
        assert!((sub.volume() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn integer_sub_rule_no_overlap() {
        let ii = IntegerInterval::new(0, 3, Some((0, 9)), false);
        assert!(ii.sub_rule(5.0, 8.0).is_none());
    }

    #[test]
    fn integer_contains_point() {
        let ii = IntegerInterval::new(2, 5, Some((0, 9)), false);
        assert!(ii.contains_point(2.0));
        assert!(ii.contains_point(5.0));
        assert!(!ii.contains_point(6.0));
        assert!(!ii.contains_point(1.0));
    }

    // -- BelongsTo ---------------------------------------------------------

    #[test]
    fn categorical_boundaries() {
        let domain: Vec<usize> = (0..4).collect();
        let names: Vec<String> = domain.iter().map(|i| format!("c{i}")).collect();
        let values: HashSet<usize> = [0, 2].into_iter().collect();
        let bt = BelongsTo::new(values, Arc::new(domain), Arc::new(names), false);
        assert_eq!(bt.boundaries(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn categorical_sub_rule_active() {
        let domain: Vec<usize> = (0..3).collect();
        let names: Vec<String> = domain.iter().map(|i| format!("c{i}")).collect();
        let values: HashSet<usize> = [0, 2].into_iter().collect();
        let bt = BelongsTo::new(values, Arc::new(domain), Arc::new(names), false);

        // Category 0 is active
        let sub = bt.sub_rule(0.0, 1.0).unwrap();
        assert!((sub.volume() - 1.0).abs() < 1e-10);

        // Category 1 is NOT active
        assert!(bt.sub_rule(1.0, 2.0).is_none());

        // Category 2 is active
        assert!(bt.sub_rule(2.0, 3.0).is_some());
    }

    #[test]
    fn categorical_contains_point() {
        let domain: Vec<usize> = (0..3).collect();
        let names: Vec<String> = domain.iter().map(|i| format!("c{i}")).collect();
        let values: HashSet<usize> = [0, 2].into_iter().collect();
        let bt = BelongsTo::new(values, Arc::new(domain), Arc::new(names), false);

        assert!(bt.contains_point(0.5)); // bin [0,1) → category 0
        assert!(!bt.contains_point(1.5)); // bin [1,2) → category 1
        assert!(bt.contains_point(2.5)); // bin [2,3) → category 2
    }

    // -- Downcast helper ---------------------------------------------------

    #[test]
    fn downcast_continuous() {
        let rule: Box<dyn DynRule> =
            Box::new(ContinuousInterval::new(0.0, 1.0, true, false, None, false));
        assert!(as_disjoint_resolvable(rule.as_ref()).is_some());
    }

    #[test]
    fn downcast_integer() {
        let rule: Box<dyn DynRule> = Box::new(IntegerInterval::new(0, 5, None, false));
        assert!(as_disjoint_resolvable(rule.as_ref()).is_some());
    }

    #[test]
    fn downcast_categorical() {
        let domain: Vec<usize> = (0..3).collect();
        let names: Vec<String> = domain.iter().map(|i| format!("c{i}")).collect();
        let values: HashSet<usize> = [0].into_iter().collect();
        let rule: Box<dyn DynRule> = Box::new(BelongsTo::new(
            values,
            Arc::new(domain),
            Arc::new(names),
            false,
        ));
        assert!(as_disjoint_resolvable(rule.as_ref()).is_some());
    }
}
