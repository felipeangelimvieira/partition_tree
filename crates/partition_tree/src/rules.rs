use core::fmt;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::BitAnd;
use std::sync::Arc;

use crate::dataset_view::QuantizedContinuousSpec;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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
        let vol = self.values.len() as f64;
        if vol <= 0.0 {
            return vec![0.0; self.domain.len()];
        }
        self.domain
            .iter()
            .map(|value| {
                if self.values.contains(value) {
                    1.0 / vol
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

// ---------------------------------------------------------------------------
// IntegerInterval
// ---------------------------------------------------------------------------

/// Represents a discrete integer interval $[\text{low}, \text{high}]$.
///
/// Volume is the count of integers in the range: $\text{high} - \text{low} + 1$.
/// Splits produce sub-intervals at an integer boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerInterval {
    /// Inclusive lower bound.
    pub low: i64,
    /// Inclusive upper bound.
    pub high: i64,
    /// Domain bounds $(d_{\min}, d_{\max})$ used for relative volume.
    pub domain: (i64, i64),
    /// Whether to accept `None` (null / missing) values.
    pub accept_none: bool,
}

impl IntegerInterval {
    /// Create a new integer interval.
    ///
    /// `domain` defaults to `(low, high)` if `None`.
    pub fn new(low: i64, high: i64, domain: Option<(i64, i64)>, accept_none: bool) -> Self {
        let domain = domain.unwrap_or((low, high));
        Self {
            low,
            high,
            domain,
            accept_none,
        }
    }

    /// Domain bounds.
    pub fn domain(&self) -> (i64, i64) {
        self.domain
    }

    /// Discrete volume: count of integers in `[low, high]`.
    fn discrete_volume(lo: i64, hi: i64) -> f64 {
        if hi < lo {
            0.0
        } else {
            // Use f64 arithmetic to avoid i64 overflow for extreme ranges.
            (hi as f64) - (lo as f64) + 1.0
        }
    }
}

impl Rule<i64> for IntegerInterval {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn _evaluate_some(&self, value: &i64) -> bool {
        *value >= self.low && *value <= self.high
    }

    fn _evaluate_none(&self) -> bool {
        self.accept_none
    }

    fn volume(&self) -> f64 {
        Self::discrete_volume(self.low, self.high)
    }

    fn relative_volume(&self) -> f64 {
        let interval = Self::discrete_volume(self.low, self.high);
        let domain = Self::discrete_volume(self.domain.0, self.domain.1);
        if domain <= 0.0 {
            0.0
        } else {
            interval / domain
        }
    }

    fn mean(&self) -> Vec<f64> {
        vec![(self.low as f64 + self.high as f64) / 2.0]
    }

    fn split(&self, threshold: i64, none_to_left: Option<bool>) -> (Self, Self) {
        let accept_none_left = self.accept_none && none_to_left.unwrap_or(true);
        let accept_none_right = self.accept_none && !none_to_left.unwrap_or(false);

        // Left: [low, threshold - 1],  Right: [threshold, high]
        let left = IntegerInterval {
            low: self.low,
            high: threshold - 1,
            domain: self.domain,
            accept_none: accept_none_left,
        };
        let right = IntegerInterval {
            low: threshold,
            high: self.high,
            domain: self.domain,
            accept_none: accept_none_right,
        };
        (left, right)
    }

    fn inverse_one_hot(&self, vec: &Vec<f64>) -> i64 {
        assert!(
            vec.len() == 1,
            "IntegerInterval expects a single-value vector for inverse_one_hot"
        );
        vec[0] as i64
    }

    fn phi_length(u: f64) -> f64 {
        if u <= 0.0 { 0.0 } else { u / (1.0 + u) }
    }
}

impl BitAnd for IntegerInterval {
    type Output = IntegerInterval;

    fn bitand(self, other: Self) -> Self::Output {
        let low = self.low.max(other.low);
        let high = self.high.min(other.high);
        IntegerInterval {
            low,
            high,
            domain: self.domain,
            accept_none: self.accept_none && other.accept_none,
        }
    }
}

impl fmt::Display for IntegerInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IntegerInterval([{}, {}], domain=[{}, {}])",
            self.low, self.high, self.domain.0, self.domain.1
        )
    }
}

// ---------------------------------------------------------------------------
// QuantizedContinuousInterval
// ---------------------------------------------------------------------------

/// Represents a discrete lattice interval over real values.
///
/// Values must lie on `resolution * i` for some integer `i`. The stored bounds
/// are lattice indices, but the rule evaluates and reports values in the
/// original `f64` space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedContinuousInterval {
    /// Lower lattice boundary index.
    pub low_idx: i64,
    /// Upper lattice boundary index.
    pub high_idx: i64,
    /// Whether the lower boundary is closed.
    pub lower_closed: bool,
    /// Whether the upper boundary is closed.
    pub upper_closed: bool,
    /// Positive lattice resolution.
    pub resolution: f64,
    /// Domain bounds in lattice indices.
    pub domain: (i64, i64),
    /// Whether to accept `None` (null / missing) values.
    pub accept_none: bool,
}

impl QuantizedContinuousInterval {
    /// Create a new quantized-continuous interval.
    pub fn new(
        low_idx: i64,
        high_idx: i64,
        resolution: f64,
        domain: Option<(i64, i64)>,
        accept_none: bool,
    ) -> Self {
        QuantizedContinuousSpec::new(resolution)
            .expect("QuantizedContinuousInterval requires a finite positive resolution");

        let domain = domain.unwrap_or((low_idx, high_idx));
        Self {
            low_idx,
            high_idx,
            lower_closed: true,
            upper_closed: true,
            resolution,
            domain,
            accept_none,
        }
    }

    pub fn domain(&self) -> (i64, i64) {
        self.domain
    }

    pub fn low(&self) -> f64 {
        (self.low_idx as f64 + if self.lower_closed { -0.5 } else { 0.5 }) * self.resolution
    }

    pub fn high(&self) -> f64 {
        (self.high_idx as f64 + if self.upper_closed { 0.5 } else { -0.5 }) * self.resolution
    }

    fn boundary_tolerance(boundary: f64, resolution: f64) -> f64 {
        f64::EPSILON * boundary.abs().max(resolution.abs()).max(1.0) * 16.0
    }

    pub fn quantize_value(value: f64, resolution: f64) -> Result<i64, String> {
        QuantizedContinuousSpec::new(resolution)?.quantize_value(value)
    }

    pub fn split_boundary_from_index(threshold_idx: i64, resolution: f64) -> f64 {
        (threshold_idx as f64 - 0.5) * resolution
    }

    fn split_index_from_boundary(point: f64, resolution: f64) -> Result<i64, String> {
        QuantizedContinuousSpec::new(resolution)
            .expect("QuantizedContinuousInterval requires a finite positive resolution");

        if !point.is_finite() {
            return Err(format!(
                "split point {point} is not a valid half-step boundary for resolution {resolution}"
            ));
        }

        let shifted = point / resolution + 0.5;
        let rounded = shifted.round();
        let tolerance = shifted.abs().max(1.0) * 1e-9;

        if (shifted - rounded).abs() > tolerance {
            return Err(format!(
                "split point {point} is not aligned to half-step boundaries for resolution {resolution}"
            ));
        }

        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            return Err(format!(
                "split point {point} is outside the quantized i64 range"
            ));
        }

        Ok(rounded as i64)
    }

    fn discrete_volume(lo: i64, hi: i64, lower_closed: bool, upper_closed: bool) -> f64 {
        if hi < lo {
            0.0
        } else {
            let mut n_cells = (hi as f64) - (lo as f64) + 1.0;
            if !lower_closed {
                n_cells -= 1.0;
            }
            if !upper_closed {
                n_cells -= 1.0;
            }
            n_cells.max(0.0)
        }
    }

    pub(crate) fn split_at_index(
        &self,
        threshold_idx: i64,
        none_to_left: Option<bool>,
    ) -> (Self, Self) {
        let accept_none_left = self.accept_none && none_to_left.unwrap_or(true);
        let accept_none_right = self.accept_none && !none_to_left.unwrap_or(false);
        let min_threshold_idx = if self.lower_closed {
            self.low_idx
        } else {
            self.low_idx.saturating_add(1)
        };
        let max_threshold_idx = if self.upper_closed {
            self.high_idx.saturating_add(1)
        } else {
            self.high_idx
        };
        let threshold_idx = threshold_idx.clamp(min_threshold_idx, max_threshold_idx);

        let left = QuantizedContinuousInterval {
            low_idx: self.low_idx,
            high_idx: threshold_idx,
            lower_closed: self.lower_closed,
            upper_closed: false,
            resolution: self.resolution,
            domain: self.domain,
            accept_none: accept_none_left,
        };
        let right = QuantizedContinuousInterval {
            low_idx: threshold_idx,
            high_idx: self.high_idx,
            lower_closed: true,
            upper_closed: self.upper_closed,
            resolution: self.resolution,
            domain: self.domain,
            accept_none: accept_none_right,
        };
        (left, right)
    }
}

impl Rule<f64> for QuantizedContinuousInterval {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn _evaluate_some(&self, value: &f64) -> bool {
        if !value.is_finite() {
            return false;
        }

        let lower_ok = if self.low_idx == i64::MIN {
            true
        } else {
            let low = self.low();
            let tolerance = Self::boundary_tolerance(low, self.resolution);

            if self.lower_closed {
                *value >= low - tolerance
            } else {
                *value > low + tolerance
            }
        };
        let upper_ok = if self.high_idx == i64::MAX {
            true
        } else {
            let high = self.high();
            let tolerance = Self::boundary_tolerance(high, self.resolution);

            if self.upper_closed {
                *value <= high + tolerance
            } else {
                *value < high - tolerance
            }
        };

        lower_ok && upper_ok
    }

    fn _evaluate_none(&self) -> bool {
        self.accept_none
    }

    fn volume(&self) -> f64 {
        Self::discrete_volume(
            self.low_idx,
            self.high_idx,
            self.lower_closed,
            self.upper_closed,
        ) * self.resolution
    }

    fn relative_volume(&self) -> f64 {
        let interval = Self::discrete_volume(
            self.low_idx,
            self.high_idx,
            self.lower_closed,
            self.upper_closed,
        );
        let domain = Self::discrete_volume(self.domain.0, self.domain.1, true, true);
        if domain <= 0.0 {
            0.0
        } else {
            interval / domain
        }
    }

    fn mean(&self) -> Vec<f64> {
        vec![(self.low() + self.high()) / 2.0]
    }

    fn split(&self, point: f64, none_to_left: Option<bool>) -> (Self, Self) {
        let threshold_idx =
            Self::split_index_from_boundary(point, self.resolution).unwrap_or_else(|_| {
            panic!(
                "QuantizedContinuousInterval split point {point} is not aligned to half-step boundaries for resolution {}",
                self.resolution
            )
        });
        self.split_at_index(threshold_idx, none_to_left)
    }

    fn inverse_one_hot(&self, vec: &Vec<f64>) -> f64 {
        assert!(
            vec.len() == 1,
            "QuantizedContinuousInterval expects a single-value vector for inverse_one_hot"
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

impl BitAnd for QuantizedContinuousInterval {
    type Output = QuantizedContinuousInterval;

    fn bitand(self, other: Self) -> Self::Output {
        QuantizedContinuousInterval {
            low_idx: self.low_idx.max(other.low_idx),
            high_idx: self.high_idx.min(other.high_idx),
            lower_closed: self.lower_closed && other.lower_closed,
            upper_closed: self.upper_closed && other.upper_closed,
            resolution: self.resolution,
            domain: self.domain,
            accept_none: self.accept_none && other.accept_none,
        }
    }
}

impl fmt::Display for QuantizedContinuousInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let open_low = if self.lower_closed { '[' } else { '(' };
        let open_high = if self.upper_closed { ']' } else { ')' };
        write!(
            f,
            "QuantizedContinuousInterval({}{:.6}, {:.6}{}, resolution={}, domain_idx=[{}, {}])",
            open_low,
            self.low(),
            self.high(),
            open_high,
            self.resolution,
            self.domain.0,
            self.domain.1
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_continuous_volume_scales_with_resolution() {
        let interval = QuantizedContinuousInterval::new(2, 6, 0.5, Some((2, 6)), true);
        assert!((interval.low() - 0.75).abs() < 1e-10);
        assert!((interval.high() - 3.25).abs() < 1e-10);
        assert!((interval.volume() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn quantized_continuous_single_point_has_one_cell_of_volume() {
        let interval = QuantizedContinuousInterval::new(4, 4, 0.5, Some((4, 4)), true);
        assert!((interval.low() - 1.75).abs() < 1e-10);
        assert!((interval.high() - 2.25).abs() < 1e-10);
        assert!((interval.volume() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn quantized_continuous_split_preserves_volume() {
        let interval = QuantizedContinuousInterval::new(2, 6, 0.5, Some((2, 6)), true);
        let (left, right) = interval.split(1.75, Some(true));

        assert!((left.volume() - 1.0).abs() < 1e-10);
        assert!((right.volume() - 1.5).abs() < 1e-10);
        assert!((left.volume() + right.volume() - interval.volume()).abs() < 1e-10);
        assert!(left.evaluate(&[Some(1.5)])[0]);
        assert!(!left.evaluate(&[Some(2.0)])[0]);
        assert!(right.evaluate(&[Some(2.0)])[0]);
    }

    #[test]
    fn quantized_continuous_split_boundary_matches_first_right_index() {
        assert!(
            (QuantizedContinuousInterval::split_boundary_from_index(4, 0.5) - 1.75).abs() < 1e-10
        );
    }

    #[test]
    fn quantized_continuous_resolution_one_matches_integer_interval_geometry() {
        let quantized = QuantizedContinuousInterval::new(2, 6, 1.0, Some((2, 6)), true);
        let integer = IntegerInterval::new(2, 6, Some((2, 6)), true);

        assert!((quantized.volume() - integer.volume()).abs() < 1e-10);
        assert!((quantized.relative_volume() - integer.relative_volume()).abs() < 1e-10);
        assert!((quantized.mean()[0] - integer.mean()[0]).abs() < 1e-10);

        let (q_left, q_right) = quantized.split(3.5, Some(true));
        let (i_left, i_right) = integer.split(4, Some(true));

        assert!((q_left.volume() - i_left.volume()).abs() < 1e-10);
        assert!((q_right.volume() - i_right.volume()).abs() < 1e-10);
        assert!(q_left.evaluate(&[Some(3.0)])[0]);
        assert!(!q_left.evaluate(&[Some(3.5)])[0]);
        assert!(q_right.evaluate(&[Some(3.5)])[0]);
        assert!(!q_left.evaluate(&[Some(4.0)])[0]);
        assert!(q_right.evaluate(&[Some(4.0)])[0]);
    }

    #[test]
    fn quantized_continuous_bin_includes_values_within_half_resolution_of_center() {
        let zero_bin = QuantizedContinuousInterval::new(0, 0, 1.0, Some((0, 0)), true);

        assert_eq!(QuantizedContinuousInterval::quantize_value(0.0, 1.0), Ok(0));
        assert_eq!(
            QuantizedContinuousInterval::quantize_value(0.002, 1.0),
            Ok(0)
        );
        assert!(zero_bin.evaluate(&[Some(0.0)])[0]);
        assert!(zero_bin.evaluate(&[Some(0.002)])[0]);
    }

    #[test]
    fn quantized_continuous_quantize_value_maps_points_to_expected_bins() {
        let cases = [
            (-1.49, -1),
            (-0.51, -1),
            (-0.49, 0),
            (0.002, 0),
            (0.49, 0),
            (0.51, 1),
            (1.49, 1),
            (1.51, 2),
        ];

        for (value, expected_idx) in cases {
            assert_eq!(
                QuantizedContinuousInterval::quantize_value(value, 1.0),
                Ok(expected_idx),
                "value {value} should quantize to bin {expected_idx}"
            );
        }
    }

    #[test]
    fn quantized_continuous_single_bin_includes_closed_endpoints() {
        let zero_bin = QuantizedContinuousInterval::new(0, 0, 1.0, Some((0, 0)), true);

        assert!(zero_bin.evaluate(&[Some(-0.5)])[0]);
        assert!(zero_bin.evaluate(&[Some(0.5)])[0]);
        assert!(!zero_bin.evaluate(&[Some(-0.500_000_1)])[0]);
        assert!(!zero_bin.evaluate(&[Some(0.500_000_1)])[0]);
    }

    #[test]
    fn quantized_continuous_unbounded_interval_tolerance_does_not_swallow_values() {
        let left = QuantizedContinuousInterval {
            low_idx: i64::MIN,
            high_idx: 3,
            lower_closed: true,
            upper_closed: false,
            resolution: 1.0,
            domain: (i64::MIN, i64::MAX),
            accept_none: false,
        };

        assert!(left.evaluate(&[Some(1.0)])[0]);
        assert!(left.evaluate(&[Some(2.49)])[0]);
        assert!(!left.evaluate(&[Some(2.5)])[0]);
    }
}

// Rule types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RuleType {
    Continuous(ContinuousInterval),
    BelongsTo(BelongsTo),
    Integer(IntegerInterval),
    QuantizedContinuous(QuantizedContinuousInterval),
}

impl RuleType {
    pub fn is_categorical(&self) -> bool {
        matches!(self, RuleType::BelongsTo(_))
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, RuleType::Integer(_))
    }

    pub fn is_quantized_continuous(&self) -> bool {
        matches!(self, RuleType::QuantizedContinuous(_))
    }
}

impl fmt::Display for RuleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuleType::Continuous(interval) => write!(f, "{}", interval),
            RuleType::BelongsTo(belongs_to) => write!(f, "{}", belongs_to),
            RuleType::Integer(interval) => write!(f, "{}", interval),
            RuleType::QuantizedContinuous(interval) => write!(f, "{}", interval),
        }
    }
}
