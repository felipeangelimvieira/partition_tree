//! Rule types for v2 partition tree.
//!
//! This module re-exports the core rule types from v1 ([`ContinuousInterval`],
//! [`BelongsToGeneric`]) and adds convenience methods on [`RuleType`] so that
//! the v2 tree builder can work with rules polymorphically without downcasting.
//!
//! ## Rule Kinds
//!
//! | Variant                 | Represents                           | Volume                |
//! |-------------------------|--------------------------------------|-----------------------|
//! | [`RuleType::Continuous`]| Half-open interval $[\text{lo}, \text{hi})$ | $\text{hi} - \text{lo}$ |
//! | [`RuleType::BelongsTo`] | Set of categorical codes             | \|active set\|           |
//!
//! Both variants carry a *domain* (the parent's full range / code set) used to
//! compute relative and phi-transformed volumes.
use std::collections::HashSet;
use std::fmt;

// Re-export the v1 rule types unchanged
pub use crate::rules::{
    BelongsTo, BelongsToGeneric, ContinuousInterval, Rule, RuleType, RuleValue,
};

// ---------------------------------------------------------------------------
// Convenience methods on RuleType for v2 consumers
// ---------------------------------------------------------------------------

impl RuleType {
    /// Whether this rule accepts null / missing values.
    pub fn accept_none(&self) -> bool {
        match self {
            RuleType::Continuous(ci) => ci.accept_none,
            RuleType::BelongsTo(bt) => bt.accept_none,
        }
    }

    /// Raw volume of the rule.
    pub fn volume(&self) -> f64 {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::volume(ci),
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::volume(bt),
        }
    }

    /// Relative volume (normalized by domain).
    pub fn relative_volume(&self) -> f64 {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::relative_volume(ci),
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::relative_volume(bt),
        }
    }

    /// Phi-transformed volume.
    pub fn phi_volume(&self) -> f64 {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::phi_volume(ci),
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::phi_volume(bt),
        }
    }

    /// Vector representation used for downstream decoding.
    pub fn mean(&self) -> Vec<f64> {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::mean(ci),
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::mean(bt),
        }
    }

    /// Split a continuous rule at `threshold`.
    ///
    /// Returns `(left, right)` where left covers $[\text{lo}, \text{threshold})$
    /// and right covers $[\text{threshold}, \text{hi})$.
    ///
    /// # Panics
    ///
    /// Panics if called on a [`RuleType::BelongsTo`] variant.
    pub fn split_continuous(&self, threshold: f64, none_to_left: bool) -> (RuleType, RuleType) {
        match self {
            RuleType::Continuous(ci) => {
                let (left, right) =
                    <ContinuousInterval as Rule<f64>>::split(ci, threshold, Some(none_to_left));
                (RuleType::Continuous(left), RuleType::Continuous(right))
            }
            RuleType::BelongsTo(_) => {
                panic!("split_continuous called on a categorical RuleType")
            }
        }
    }

    /// Split a categorical rule by subset.
    ///
    /// `subset_left` contains the category codes routed to the left child;
    /// remaining active codes go to the right child.
    ///
    /// # Panics
    ///
    /// Panics if called on a [`RuleType::Continuous`] variant.
    pub fn split_categorical(
        &self,
        subset_left: HashSet<usize>,
        none_to_left: bool,
    ) -> (RuleType, RuleType) {
        match self {
            RuleType::BelongsTo(bt) => {
                let (left, right) = bt.split_subset(subset_left, Some(none_to_left));
                (RuleType::BelongsTo(left), RuleType::BelongsTo(right))
            }
            RuleType::Continuous(_) => {
                panic!("split_categorical called on a continuous RuleType")
            }
        }
    }

    /// Evaluate membership for a continuous value.
    ///
    /// Returns `true` if the value falls within the interval, or if the
    /// value is `None` and `accept_none` is set.
    ///
    /// # Panics
    ///
    /// Panics if called on a [`RuleType::BelongsTo`] variant.
    pub fn evaluate_continuous(&self, value: Option<f64>) -> bool {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::_evaluate(ci, &value),
            RuleType::BelongsTo(_) => {
                panic!("evaluate_continuous called on a categorical RuleType")
            }
        }
    }

    /// Evaluate membership for a categorical value (usize code).
    ///
    /// Returns `true` if the code is in the active set, or if the
    /// value is `None` and `accept_none` is set.
    ///
    /// # Panics
    ///
    /// Panics if called on a [`RuleType::Continuous`] variant.
    pub fn evaluate_categorical(&self, value: Option<usize>) -> bool {
        match self {
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::_evaluate(bt, &value),
            RuleType::Continuous(_) => {
                panic!("evaluate_categorical called on a continuous RuleType")
            }
        }
    }

    /// Domain bounds for continuous rules, None for categorical.
    pub fn continuous_domain(&self) -> Option<(f64, f64)> {
        match self {
            RuleType::Continuous(ci) => Some(ci.domain()),
            RuleType::BelongsTo(_) => None,
        }
    }

    /// Number of active categories for categorical rules, None for continuous.
    pub fn categorical_cardinality(&self) -> Option<usize> {
        match self {
            RuleType::BelongsTo(bt) => Some(bt.values.len()),
            RuleType::Continuous(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_continuous(low: f64, high: f64) -> RuleType {
        RuleType::Continuous(ContinuousInterval::new(
            low,
            high,
            true,
            true,
            Some((low, high)),
            true,
        ))
    }

    fn make_categorical(values: Vec<usize>, domain_size: usize) -> RuleType {
        let domain: Vec<usize> = (0..domain_size).collect();
        let names: Vec<String> = domain.iter().map(|i| format!("cat_{i}")).collect();
        let bt = BelongsTo::new(
            values.into_iter().collect(),
            Arc::new(domain),
            Arc::new(names),
            true,
        );
        RuleType::BelongsTo(bt)
    }

    #[test]
    fn continuous_volume() {
        let r = make_continuous(0.0, 10.0);
        assert!((r.volume() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn continuous_split_volumes_sum() {
        let r = make_continuous(0.0, 10.0);
        let (left, right) = r.split_continuous(3.0, true);
        let total = left.volume() + right.volume();
        assert!((total - 10.0).abs() < 1e-10);
    }

    #[test]
    fn categorical_volume() {
        let r = make_categorical(vec![0, 1, 2], 5);
        assert!((r.volume() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn categorical_split_preserves_domain() {
        let r = make_categorical(vec![0, 1, 2, 3], 5);
        let subset: HashSet<usize> = [0, 1].into_iter().collect();
        let (left, right) = r.split_categorical(subset, true);
        assert!((left.volume() - 2.0).abs() < 1e-10);
        assert!((right.volume() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn evaluate_continuous_membership() {
        let r = make_continuous(0.0, 10.0);
        assert!(r.evaluate_continuous(Some(5.0)));
        assert!(!r.evaluate_continuous(Some(15.0)));
        assert!(r.evaluate_continuous(None)); // accept_none = true
    }

    #[test]
    fn evaluate_categorical_membership() {
        let r = make_categorical(vec![0, 2], 4);
        assert!(r.evaluate_categorical(Some(0)));
        assert!(!r.evaluate_categorical(Some(1)));
        assert!(r.evaluate_categorical(Some(2)));
        assert!(r.evaluate_categorical(None)); // accept_none = true
    }
}
