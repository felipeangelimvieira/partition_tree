use core::fmt;
use std::any::Any;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::BitAnd;
use std::sync::Arc;

/// Trait alias for types that can be used as rule values.
/// This simplifies the generic bounds throughout the codebase.
pub trait RuleValue: Eq + Hash + Clone + 'static {}

// Blanket implementation for any type that satisfies the bounds
impl<T: Eq + Hash + Clone + 'static> RuleValue for T {}

pub trait Rule<T>: Any {
    fn as_any(&self) -> &dyn Any;

    fn _evaluate_none(&self) -> bool {
        false // Default implementation for None handling
    }

    fn _evaluate_some(&self, value: &T) -> bool;
    // Private element-wise evaluation method
    fn _evaluate(&self, value: &Option<T>) -> bool {
        match value {
            None => self._evaluate_none(),
            Some(x) => self._evaluate_some(x),
        }
    }

    // Public vector evaluation method with default implementation
    fn evaluate(&self, data: &[Option<T>]) -> Vec<bool> {
        data.iter().map(|x| self._evaluate(x)).collect()
    }
    // Phi-transformed length metric for geometric exploration
    fn phi_length(u: f64) -> f64 {
        u
    }

    fn phi_volume(&self) -> f64 {
        let raw = self.volume();
        Self::phi_length(raw)
    }
    // Numeric "volume" of the rule
    fn volume(&self) -> f64;

    fn relative_volume(&self) -> f64;

    fn mean(&self) -> Vec<f64>;

    fn split(&self, point: T, none_to_left: Option<bool>) -> (Self, Self)
    where
        Self: Sized;

    fn inverse_one_hot(&self, vec: &Vec<f64>) -> T;
}

/// -------------------
/// ContinuousInterval
/// -------------------
/// Represents a continuous interval with bounds and closure properties

#[derive(Clone, Debug)]
pub struct ContinuousInterval {
    // Represents a continuous interval with bounds and closure properties
    pub low: f64,
    pub high: f64,
    pub lower_closed: bool,
    pub upper_closed: bool,
    pub domain: (f64, f64), // (min, max) domain bounds
    pub accept_none: bool,  // Whether to accept None values
}

impl ContinuousInterval {
    pub fn new(
        low: f64,
        high: f64,
        lower_closed: bool,
        upper_closed: bool,
        domain: Option<(f64, f64)>,
        accept_none: bool,
    ) -> Self {
        let domain = domain.unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        Self {
            low,
            high,
            lower_closed,
            upper_closed,
            domain,
            accept_none,
        }
    }

    pub fn domain(&self) -> (f64, f64) {
        self.domain
    }
}

impl Rule<f64> for ContinuousInterval {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn _evaluate_some(&self, value: &f64) -> bool {
        let x = *value;
        let lower_check = if self.lower_closed {
            x >= self.low
        } else {
            x > self.low
        };
        let upper_check = if self.upper_closed {
            x <= self.high
        } else {
            x < self.high
        };
        lower_check && upper_check
    }

    fn _evaluate_none(&self) -> bool {
        self.accept_none
    }

    fn volume(&self) -> f64 {
        (self.high - self.low).abs()
    }

    /// Get the relative volume of this interval within its domain
    fn relative_volume(&self) -> f64 {
        let interval_size = (self.high - self.low).abs();
        let domain_size = (self.domain.1 - self.domain.0).abs();
        if domain_size == 0.0 || domain_size.is_infinite() {
            0.0
        } else {
            interval_size / domain_size
        }
    }

    fn mean(&self) -> Vec<f64> {
        vec![(self.low + self.high) / 2.0]
    }

    fn split(&self, point: f64, none_to_left: Option<bool>) -> (Self, Self) {
        let accept_none_left = self.accept_none && none_to_left.unwrap_or(true);
        let accept_none_right = self.accept_none && !none_to_left.unwrap_or(false);

        let left = ContinuousInterval {
            low: self.low,
            high: point,
            lower_closed: self.lower_closed,
            upper_closed: false,
            domain: self.domain,
            accept_none: accept_none_left, // If none is accepted, it goes to the left
        };
        let right = ContinuousInterval {
            low: point,
            high: self.high,
            lower_closed: true,
            upper_closed: self.upper_closed,
            domain: self.domain,
            accept_none: accept_none_right, // If none is accepted, it goes to the right
        };
        (left, right)
    }

    fn inverse_one_hot(&self, vec: &Vec<f64>) -> f64 {
        assert!(
            vec.len() == 1,
            "ContinuousInterval expects a single-value vector for inverse_one_hot"
        );
        vec[0]
    }

    fn phi_length(u: f64) -> f64 {
        if !u.is_finite() {
            1.0
        } else if u <= 0.0 {
            0.0
        } else {
            u / (1.0 + u)
        }
    }
}

impl BitAnd for ContinuousInterval {
    type Output = ContinuousInterval;

    fn bitand(self, other: Self) -> Self::Output {
        let low = self.low.max(other.low);
        let high = self.high.min(other.high);
        let lower_closed = self.lower_closed && other.lower_closed;
        let upper_closed = self.upper_closed && other.upper_closed;
        let accept_none = self.accept_none && other.accept_none;

        ContinuousInterval {
            low,
            high,
            lower_closed,
            upper_closed,
            domain: self.domain, // Preserve domain from the first interval
            accept_none,
        }
    }
}

impl Into<Vec<f64>> for ContinuousInterval {
    fn into(self) -> Vec<f64> {
        vec![(self.low + self.high) / 2.0]
    }
}

impl fmt::Display for ContinuousInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ContinuousInterval({}, {}, {}, {}, ({}, {}))",
            self.low, self.high, self.lower_closed, self.upper_closed, self.domain.0, self.domain.1
        )
    }
}

// ---------------------------
// Generic BelongsTo over any physical type (u32, i64, bool, String, ...)
// ---------------------------

#[derive(Clone, Debug)]
pub struct BelongsToGeneric<T: RuleValue> {
    // Set for O(1) membership checks during evaluation
    pub values: HashSet<T>,
    // Ordered full domain of values
    pub domain: Arc<Vec<T>>,
    // Ordered names aligned with domain indices/positions
    pub domain_names: Arc<Vec<String>>,
    pub accept_none: bool, // Whether to accept None values
}

impl<T: RuleValue> BelongsToGeneric<T> {
    pub fn new(
        values: HashSet<T>,
        domain: Arc<Vec<T>>,
        domain_names: Arc<Vec<String>>,
        accept_none: bool,
    ) -> Self {
        // Deduplicate provided values, preserving insertion order

        Self {
            values,
            domain,
            domain_names,
            accept_none,
        }
    }

    pub fn values_as_vec(&self) -> Vec<Option<T>> {
        self.values.iter().cloned().map(Some).collect()
    }

    pub fn domain_as_vec(&self) -> Vec<T> {
        self.domain.as_ref().iter().cloned().collect()
    }

    pub fn contains(&self, v: &T) -> bool {
        if self.domain.contains(v) {
            self.values.contains(v)
        } else {
            true // Out-of-domain values are considered as "belongs to"
        }
    }

    /// Split by a subset of categories.
    /// Left child gets all categories in the subset (regardless of whether they're in current values).
    /// Right child gets all current values NOT in the subset.
    pub fn split_subset(&self, subset: HashSet<T>, none_to_left: Option<bool>) -> (Self, Self) {
        let accept_none_left = self.accept_none && none_to_left.unwrap_or(true);
        let accept_none_right = self.accept_none && !none_to_left.unwrap_or(false);

        // Left gets the subset (all categories in the subset, matching original behavior)
        let left = BelongsToGeneric::new(
            subset.clone(),
            Arc::clone(&self.domain),
            Arc::clone(&self.domain_names),
            accept_none_left,
        );

        // Right keeps values that are in self.values but NOT in subset
        let right_vals: HashSet<T> = self
            .values
            .iter()
            .filter(|v| !subset.contains(*v))
            .cloned()
            .collect();

        let right = BelongsToGeneric::new(
            right_vals,
            Arc::clone(&self.domain),
            Arc::clone(&self.domain_names),
            accept_none_right,
        );
        (left, right)
    }
}

impl<T: RuleValue> Rule<T> for BelongsToGeneric<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn _evaluate_some(&self, value: &T) -> bool {
        // Use contains so out-of-domain values follow the configured containment policy
        self.contains(value)
    }

    fn _evaluate_none(&self) -> bool {
        self.accept_none
    }

    fn volume(&self) -> f64 {
        self.values.len() as f64
    }

    fn relative_volume(&self) -> f64 {
        if self.domain.is_empty() {
            0.0
        } else {
            self.values.len() as f64 / self.domain.len() as f64
        }
    }

    fn mean(&self) -> Vec<f64> {
        self.domain
            .iter()
            .map(|value| {
                if self.values.contains(value) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn split(&self, point: T, none_to_left: Option<bool>) -> (Self, Self)
    where
        Self: Sized,
    {
        // Single-point split: left gets the point, right gets everything else
        self.split_subset(HashSet::from([point]), none_to_left)
    }

    fn inverse_one_hot(&self, vec: &Vec<f64>) -> T {
        //! Get the index with the highest value in the one-hot vector
        //! and the corresponding class value from the domain.
        let argmax = vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        self.domain[argmax].clone()
    }

    fn phi_length(u: f64) -> f64 {
        u - 1.0
    }
}

impl<T: RuleValue> BitAnd for BelongsToGeneric<T> {
    type Output = BelongsToGeneric<T>;

    fn bitand(self, other: Self) -> Self::Output {
        assert!(
            Arc::ptr_eq(&self.domain, &other.domain),
            "Cannot perform AND operation on BelongsTo rules with different category domains",
        );
        // Ordered intersection by iterating domain order
        let mut inter: HashSet<T> =
            HashSet::with_capacity(self.values.len().min(other.values.len()));
        for code in self.domain.iter() {
            if self.values.contains(code) && other.values.contains(code) {
                inter.insert(code.clone());
            }
        }
        BelongsToGeneric::new(
            inter,
            Arc::clone(&self.domain),
            Arc::clone(&self.domain_names),
            self.accept_none && other.accept_none,
        )
    }
}

impl<T: RuleValue> Into<Vec<f64>> for BelongsToGeneric<T> {
    fn into(self) -> Vec<f64> {
        self.domain
            .iter()
            .map(|v| if self.values.contains(v) { 1.0 } else { 0.0 })
            .collect()
    }
}

// Provide a stable, meaningful Display specifically for usize-coded categories
// where the value directly indexes into the domain_names vector.
impl fmt::Display for BelongsToGeneric<usize> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values_str: Vec<String> = self
            .values
            .iter()
            .map(|v| {
                self.domain_names
                    .get(*v)
                    .cloned()
                    .unwrap_or_else(|| format!("{}", v))
            })
            .collect();
        write!(f, "BelongsTo({})", values_str.join(", "))
    }
}

// Backward-compatible alias (usize codes)
pub type BelongsTo = BelongsToGeneric<usize>;

// Convenience aliases for common Polars physical types
pub type BelongsToU8 = BelongsToGeneric<u8>;
pub type BelongsToU16 = BelongsToGeneric<u16>;
pub type BelongsToU32 = BelongsToGeneric<u32>;
pub type BelongsToU64 = BelongsToGeneric<u64>;
pub type BelongsToI8 = BelongsToGeneric<i8>;
pub type BelongsToI16 = BelongsToGeneric<i16>;
pub type BelongsToI32 = BelongsToGeneric<i32>;
pub type BelongsToI64 = BelongsToGeneric<i64>;
pub type BelongsToBool = BelongsToGeneric<bool>;
pub type BelongsToString = BelongsToGeneric<String>;

// Specialized helpers for usize variant to keep existing API in tests
impl BelongsToGeneric<usize> {
    pub fn values_as_usize(&self) -> Vec<Option<usize>> {
        self.values.iter().copied().map(Some).collect()
    }
    pub fn domain_as_usize(&self) -> Vec<usize> {
        self.domain.as_ref().into_iter().copied().collect()
    }
}

// Rule types
#[derive(Clone, Debug)]
pub enum RuleType {
    Continuous(ContinuousInterval),
    BelongsTo(BelongsTo),
}

impl RuleType {
    pub fn is_categorical(&self) -> bool {
        matches!(self, RuleType::BelongsTo(_))
    }
}

impl fmt::Display for RuleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuleType::Continuous(interval) => write!(f, "{}", interval),
            RuleType::BelongsTo(belongs_to) => write!(f, "{}", belongs_to),
        }
    }
}
