//! Rule types for v2 partition tree.
//!
//! ## `DynRule` — dtype-erased rule trait
//!
//! [`DynRule`] is the trait-object interface that replaces the two-variant
//! `RuleType` enum. Every rule stored in a [`Cell`](super::cell::Cell) is a
//! `Box<dyn DynRule>`, so adding a new dtype never requires touching
//! downstream match arms.
//!
//! Concrete implementations:
//!
//! | Type                    | Represents                                   | Volume                     |
//! |-------------------------|----------------------------------------------|----------------------------|
//! | [`ContinuousInterval`]  | Half-open interval $[\text{lo}, \text{hi})$  | $\text{hi} - \text{lo}$    |
//! | [`BelongsTo`]           | Set of categorical codes                     | \|active set\|             |
//!
//! ## `DynValue` — dtype-erased sample value
//!
//! [`DynValue`] carries a single sample value for rule evaluation and
//! map-based prediction without requiring separate per-dtype maps.
use std::collections::HashSet;
use std::fmt;

// Re-export the v1 rule types unchanged
pub use crate::rules::{
    BelongsTo, BelongsToGeneric, ContinuousInterval, IntegerInterval, Rule, RuleType, RuleValue,
};

// ---------------------------------------------------------------------------
// DynValue — dtype-erased sample value
// ---------------------------------------------------------------------------

/// A single sample value, dtype-erased.
///
/// Used in [`DynRule::contains`] and in map-based prediction
/// ([`Tree::predict_leaf_from_map`](super::tree::Tree::predict_leaf_from_map)).
#[derive(Debug, Clone)]
pub enum DynValue {
    /// Continuous (f64) value.
    Continuous(f64),
    /// Categorical code (usize).
    Categorical(usize),
    /// Integer (i64) value.
    Integer(i64),
}

// ---------------------------------------------------------------------------
// DynRule trait
// ---------------------------------------------------------------------------

/// Dtype-erased rule trait stored in [`Cell`](super::cell::Cell).
///
/// Implementors must provide membership testing, volume computation,
/// and cloneability so that cells can be cheaply duplicated during
/// tree construction.
///
/// # Thread Safety
///
/// Must be `Send + Sync` for parallel split search.
pub trait DynRule: Send + Sync + fmt::Debug + fmt::Display {
    /// Raw volume of the rule (interval length, set cardinality, etc.).
    fn volume(&self) -> f64;

    /// Relative volume normalized by domain.
    fn relative_volume(&self) -> f64;

    /// Phi-transformed volume.
    fn phi_volume(&self) -> f64;

    /// Mean vector representation (used for downstream decoding).
    fn mean(&self) -> Vec<f64>;

    /// Whether this rule accepts null / missing values.
    fn accept_none(&self) -> bool;

    /// Evaluate membership for a (possibly null) sample value.
    ///
    /// Returns `true` if the value belongs to this rule's region,
    /// or if the value is `None` and [`accept_none`](DynRule::accept_none) is set.
    fn contains(&self, value: Option<&DynValue>) -> bool;

    /// Clone this rule into a new `Box`.
    fn clone_box(&self) -> Box<dyn DynRule>;

    /// Downcast helper — returns `self` as `&dyn Any` for rare cases
    /// where concrete access is needed (e.g., display of split thresholds).
    fn as_any(&self) -> &dyn std::any::Any;
}

impl Clone for Box<dyn DynRule> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// DynRule impl for ContinuousInterval
// ---------------------------------------------------------------------------

impl DynRule for ContinuousInterval {
    fn volume(&self) -> f64 {
        <ContinuousInterval as Rule<f64>>::volume(self)
    }

    fn relative_volume(&self) -> f64 {
        <ContinuousInterval as Rule<f64>>::relative_volume(self)
    }

    fn phi_volume(&self) -> f64 {
        <ContinuousInterval as Rule<f64>>::phi_volume(self)
    }

    fn mean(&self) -> Vec<f64> {
        <ContinuousInterval as Rule<f64>>::mean(self)
    }

    fn accept_none(&self) -> bool {
        self.accept_none
    }

    fn contains(&self, value: Option<&DynValue>) -> bool {
        match value {
            None => self.accept_none,
            Some(DynValue::Continuous(v)) => {
                <ContinuousInterval as Rule<f64>>::_evaluate(self, &Some(*v))
            }
            Some(_) => false,
        }
    }

    fn clone_box(&self) -> Box<dyn DynRule> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// DynRule impl for BelongsTo
// ---------------------------------------------------------------------------

impl DynRule for BelongsTo {
    fn volume(&self) -> f64 {
        <BelongsTo as Rule<usize>>::volume(self)
    }

    fn relative_volume(&self) -> f64 {
        <BelongsTo as Rule<usize>>::relative_volume(self)
    }

    fn phi_volume(&self) -> f64 {
        <BelongsTo as Rule<usize>>::phi_volume(self)
    }

    fn mean(&self) -> Vec<f64> {
        <BelongsTo as Rule<usize>>::mean(self)
    }

    fn accept_none(&self) -> bool {
        self.accept_none
    }

    fn contains(&self, value: Option<&DynValue>) -> bool {
        match value {
            None => self.accept_none,
            Some(DynValue::Categorical(v)) => {
                <BelongsTo as Rule<usize>>::_evaluate(self, &Some(*v))
            }
            Some(_) => false,
        }
    }

    fn clone_box(&self) -> Box<dyn DynRule> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// DynRule impl for IntegerInterval
// ---------------------------------------------------------------------------

impl DynRule for IntegerInterval {
    fn volume(&self) -> f64 {
        <IntegerInterval as Rule<i64>>::volume(self)
    }

    fn relative_volume(&self) -> f64 {
        <IntegerInterval as Rule<i64>>::relative_volume(self)
    }

    fn phi_volume(&self) -> f64 {
        <IntegerInterval as Rule<i64>>::phi_volume(self)
    }

    fn mean(&self) -> Vec<f64> {
        <IntegerInterval as Rule<i64>>::mean(self)
    }

    fn accept_none(&self) -> bool {
        self.accept_none
    }

    fn contains(&self, value: Option<&DynValue>) -> bool {
        match value {
            None => self.accept_none,
            Some(DynValue::Integer(v)) => {
                <IntegerInterval as Rule<i64>>::_evaluate(self, &Some(*v))
            }
            Some(_) => false,
        }
    }

    fn clone_box(&self) -> Box<dyn DynRule> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers: RuleType ↔ Box<dyn DynRule>
// ---------------------------------------------------------------------------

impl From<RuleType> for Box<dyn DynRule> {
    fn from(rt: RuleType) -> Self {
        match rt {
            RuleType::Continuous(ci) => Box::new(ci),
            RuleType::BelongsTo(bt) => Box::new(bt),
            RuleType::Integer(ii) => Box::new(ii),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience methods on RuleType for v2 consumers
// ---------------------------------------------------------------------------

impl RuleType {
    /// Whether this rule accepts null / missing values.
    pub fn accept_none(&self) -> bool {
        match self {
            RuleType::Continuous(ci) => ci.accept_none,
            RuleType::BelongsTo(bt) => bt.accept_none,
            RuleType::Integer(ii) => ii.accept_none,
        }
    }

    /// Raw volume of the rule.
    pub fn volume(&self) -> f64 {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::volume(ci),
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::volume(bt),
            RuleType::Integer(ii) => <IntegerInterval as Rule<i64>>::volume(ii),
        }
    }

    /// Relative volume (normalized by domain).
    pub fn relative_volume(&self) -> f64 {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::relative_volume(ci),
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::relative_volume(bt),
            RuleType::Integer(ii) => <IntegerInterval as Rule<i64>>::relative_volume(ii),
        }
    }

    /// Phi-transformed volume.
    pub fn phi_volume(&self) -> f64 {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::phi_volume(ci),
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::phi_volume(bt),
            RuleType::Integer(ii) => <IntegerInterval as Rule<i64>>::phi_volume(ii),
        }
    }

    /// Vector representation used for downstream decoding.
    pub fn mean(&self) -> Vec<f64> {
        match self {
            RuleType::Continuous(ci) => <ContinuousInterval as Rule<f64>>::mean(ci),
            RuleType::BelongsTo(bt) => <BelongsTo as Rule<usize>>::mean(bt),
            RuleType::Integer(ii) => <IntegerInterval as Rule<i64>>::mean(ii),
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
            RuleType::Integer(_) => {
                panic!("split_continuous called on an integer RuleType")
            }
        }
    }

    /// Split an integer rule at `threshold`.
    ///
    /// Returns `(left, right)` where left covers $[\text{lo}, \text{threshold}-1]$
    /// and right covers $[\text{threshold}, \text{hi}]$.
    ///
    /// # Panics
    ///
    /// Panics if called on a non-integer variant.
    pub fn split_integer(&self, threshold: i64, none_to_left: bool) -> (RuleType, RuleType) {
        match self {
            RuleType::Integer(ii) => {
                let (left, right) =
                    <IntegerInterval as Rule<i64>>::split(ii, threshold, Some(none_to_left));
                (RuleType::Integer(left), RuleType::Integer(right))
            }
            _ => panic!("split_integer called on a non-integer RuleType"),
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
            RuleType::Continuous(_) | RuleType::Integer(_) => {
                panic!("split_categorical called on a non-categorical RuleType")
            }
        }
    }

    /// Evaluate membership for an integer value.
    ///
    /// # Panics
    ///
    /// Panics if called on a non-integer variant.
    pub fn evaluate_integer(&self, value: Option<i64>) -> bool {
        match self {
            RuleType::Integer(ii) => <IntegerInterval as Rule<i64>>::_evaluate(ii, &value),
            _ => panic!("evaluate_integer called on a non-integer RuleType"),
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
            RuleType::BelongsTo(_) | RuleType::Integer(_) => {
                panic!("evaluate_continuous called on a non-continuous RuleType")
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
            RuleType::Continuous(_) | RuleType::Integer(_) => {
                panic!("evaluate_categorical called on a non-categorical RuleType")
            }
        }
    }

    /// Domain bounds for continuous rules, None for categorical.
    pub fn continuous_domain(&self) -> Option<(f64, f64)> {
        match self {
            RuleType::Continuous(ci) => Some(ci.domain()),
            RuleType::BelongsTo(_) => None,
            RuleType::Integer(_) => None,
        }
    }

    /// Number of active categories for categorical rules, None for continuous.
    pub fn categorical_cardinality(&self) -> Option<usize> {
        match self {
            RuleType::BelongsTo(bt) => Some(bt.values.len()),
            RuleType::Continuous(_) => None,
            RuleType::Integer(_) => None,
        }
    }

    /// Domain bounds for integer rules, None for other types.
    pub fn integer_domain(&self) -> Option<(i64, i64)> {
        match self {
            RuleType::Integer(ii) => Some(ii.domain()),
            _ => None,
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
