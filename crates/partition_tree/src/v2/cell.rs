//! Multi-dimensional partition constraint.
//!
//! A [`Cell`] is a conjunction of per-column rules. A sample belongs to a cell
//! if and only if it satisfies every rule in the map. Columns without a rule
//! are unconstrained.
//!
//! ## Volume
//!
//! The cell volume is the product of per-rule volumes (multiplicative
//! independence assumption). The **target volume** considers only columns
//! prefixed with `target__`, which is the quantity used in the density
//! denominator $f(y|x) = w_{xy} / (w_x \cdot V_{\text{target}})$.
//!
//! ## Splitting
//!
//! [`Cell::split_continuous`] and [`Cell::split_categorical`] produce
//! `(left_cell, right_cell)` pairs by replacing a single column's rule.
//! The companion `split_*_target_volumes` methods compute child volumes
//! *without* constructing the full child cells, which is used during the
//! split search inner loop.
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;

use crate::conf::TARGET_PREFIX;

use super::rule::RuleType;

/// Multi-dimensional partition constraint: one [`RuleType`] per column.
///
/// The cell is the fundamental building block of the partition tree. Each
/// tree node owns a `Cell` that describes the region of space it represents.
///
/// # Examples
///
/// ```rust,ignore
/// let cell = Cell::new()
///     .with_rule("x1", continuous_rule(0.0, 10.0))
///     .with_rule("target__y", continuous_rule(0.0, 5.0));
///
/// assert!((cell.volume() - 50.0).abs() < 1e-10);
/// assert!((cell.target_volume() - 5.0).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct Cell {
    pub rules: HashMap<String, RuleType>,
}

impl Cell {
    /// Create an empty cell (unconstrained in all dimensions).
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Create a cell from a pre-built map of rules.
    pub fn from_rules(rules: HashMap<String, RuleType>) -> Self {
        Self { rules }
    }

    /// Builder: add a rule for `col` and return self.
    pub fn with_rule(mut self, col: impl Into<String>, rule: RuleType) -> Self {
        self.rules.insert(col.into(), rule);
        self
    }

    /// Insert or replace a rule for `col`.
    pub fn set_rule(&mut self, col: impl Into<String>, rule: RuleType) {
        self.rules.insert(col.into(), rule);
    }

    /// Get the rule for a column, if present.
    pub fn get_rule(&self, col: &str) -> Option<&RuleType> {
        self.rules.get(col)
    }

    /// Product of per-rule volumes (multiplicative independence assumption).
    pub fn volume(&self) -> f64 {
        self.rules.values().map(|r| r.volume()).product()
    }

    /// Product of per-rule relative volumes.
    pub fn relative_volume(&self) -> f64 {
        self.rules.values().map(|r| r.relative_volume()).product()
    }

    /// Product of per-rule phi-volumes.
    pub fn phi_volume(&self) -> f64 {
        self.rules.values().map(|r| r.phi_volume()).product()
    }

    /// Target-only volume: product of volumes of `target__`-prefixed rules.
    ///
    /// Returns at least 1.0 when no target rules exist (unconstrained target
    /// space has unit measure by convention).
    pub fn target_volume(&self) -> f64 {
        self.target_rules()
            .map(|(_, r)| r.volume())
            .product::<f64>()
            .max(1.0) // at least 1.0 when no target rules exist
    }

    /// Target-only phi-volume.
    pub fn target_phi_volume(&self) -> f64 {
        self.target_rules()
            .map(|(_, r)| r.phi_volume())
            .product::<f64>()
            .max(1.0)
    }

    /// Iterate over rules on target columns (prefixed with `target__`).
    pub fn target_rules(&self) -> impl Iterator<Item = (&String, &RuleType)> {
        self.rules
            .iter()
            .filter(|(k, _)| k.starts_with(TARGET_PREFIX))
    }

    /// Iterate over rules on feature columns (not prefixed with `target__`).
    pub fn feature_rules(&self) -> impl Iterator<Item = (&String, &RuleType)> {
        self.rules
            .iter()
            .filter(|(k, _)| !k.starts_with(TARGET_PREFIX))
    }

    /// Number of constrained dimensions.
    pub fn n_rules(&self) -> usize {
        self.rules.len()
    }

    /// Split this cell on a continuous column at `threshold`.
    ///
    /// Returns `(left_cell, right_cell)` where the left child's rule for
    /// `col` covers $[\text{lo}, \text{threshold})$ and the right covers
    /// $[\text{threshold}, \text{hi})$. All other rules are cloned unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `col` is not present in the cell's rules.
    pub fn split_continuous(&self, col: &str, threshold: f64, none_to_left: bool) -> (Cell, Cell) {
        let rule = self
            .rules
            .get(col)
            .unwrap_or_else(|| panic!("Cell::split_continuous: column '{col}' not found"));

        let (left_rule, right_rule) = rule.split_continuous(threshold, none_to_left);

        let mut left_rules = self.rules.clone();
        left_rules.insert(col.to_string(), left_rule);

        let mut right_rules = self.rules.clone();
        right_rules.insert(col.to_string(), right_rule);

        (Cell::from_rules(left_rules), Cell::from_rules(right_rules))
    }

    /// Split this cell on a categorical column by subset.
    ///
    /// `subset_left` contains the category codes routed to the left child;
    /// remaining active codes go to the right.
    ///
    /// # Panics
    ///
    /// Panics if `col` is not present in the cell's rules.
    pub fn split_categorical(
        &self,
        col: &str,
        subset_left: HashSet<usize>,
        none_to_left: bool,
    ) -> (Cell, Cell) {
        let rule = self
            .rules
            .get(col)
            .unwrap_or_else(|| panic!("Cell::split_categorical: column '{col}' not found"));

        let (left_rule, right_rule) = rule.split_categorical(subset_left, none_to_left);

        let mut left_rules = self.rules.clone();
        left_rules.insert(col.to_string(), left_rule);

        let mut right_rules = self.rules.clone();
        right_rules.insert(col.to_string(), right_rule);

        (Cell::from_rules(left_rules), Cell::from_rules(right_rules))
    }

    /// Compute child **target volumes** for a hypothetical continuous split
    /// without constructing the child cells.
    ///
    /// If `col` is a feature column (not `target__`-prefixed), both children
    /// have the same target volume as the parent. Otherwise, the target
    /// volume is recomputed with the split column's rule replaced.
    ///
    /// This is used in the split search inner loop to avoid allocating
    /// child cells at every candidate threshold.
    pub fn split_continuous_target_volumes(
        &self,
        col: &str,
        threshold: f64,
        none_to_left: bool,
    ) -> (f64, f64) {
        // Only target columns affect volume; feature splits leave volume unchanged.
        if !col.starts_with(TARGET_PREFIX) {
            let v = self.target_volume();
            return (v, v);
        }

        let rule = self.rules.get(col).expect("column not found");
        let (left_rule, right_rule) = rule.split_continuous(threshold, none_to_left);

        // Recompute target volume replacing only the split column's rule
        let compute_vol = |replacement: &RuleType| -> f64 {
            self.target_rules()
                .map(|(k, r)| {
                    if k.as_str() == col {
                        replacement.volume()
                    } else {
                        r.volume()
                    }
                })
                .product::<f64>()
                .max(f64::MIN_POSITIVE)
        };

        (compute_vol(&left_rule), compute_vol(&right_rule))
    }

    /// Compute child **target volumes** for a hypothetical categorical split
    /// without constructing the child cells.
    ///
    /// Analogous to [`Cell::split_continuous_target_volumes`] for categorical
    /// columns.
    pub fn split_categorical_target_volumes(
        &self,
        col: &str,
        subset_left: &HashSet<usize>,
        none_to_left: bool,
    ) -> (f64, f64) {
        if !col.starts_with(TARGET_PREFIX) {
            let v = self.target_volume();
            return (v, v);
        }

        let rule = self.rules.get(col).expect("column not found");
        let (left_rule, right_rule) = rule.split_categorical(subset_left.clone(), none_to_left);

        let compute_vol = |replacement: &RuleType| -> f64 {
            self.target_rules()
                .map(|(k, r)| {
                    if k.as_str() == col {
                        replacement.volume()
                    } else {
                        r.volume()
                    }
                })
                .product::<f64>()
                .max(f64::MIN_POSITIVE)
        };

        (compute_vol(&left_rule), compute_vol(&right_rule))
    }

    /// Per-column mean vector, concatenated. Used for prediction decoding.
    pub fn mean_map(&self) -> HashMap<String, Vec<f64>> {
        self.rules
            .iter()
            .map(|(k, r)| (k.clone(), r.mean()))
            .collect()
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries: Vec<_> = self.rules.iter().collect();
        entries.sort_by_key(|(k, _)| k.clone());
        for (col, rule) in &entries {
            writeln!(f, "  {col}: {rule}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{BelongsTo, ContinuousInterval};
    use std::sync::Arc;

    fn continuous_rule(low: f64, high: f64) -> RuleType {
        RuleType::Continuous(ContinuousInterval::new(
            low,
            high,
            true,
            true,
            Some((low, high)),
            true,
        ))
    }

    fn categorical_rule(values: Vec<usize>, domain_size: usize) -> RuleType {
        let domain: Vec<usize> = (0..domain_size).collect();
        let names: Vec<String> = domain.iter().map(|i| format!("cat_{i}")).collect();
        RuleType::BelongsTo(BelongsTo::new(
            values.into_iter().collect(),
            Arc::new(domain),
            Arc::new(names),
            true,
        ))
    }

    #[test]
    fn empty_cell_volume_is_one() {
        // Product of an empty iterator is 1.0
        let cell = Cell::new();
        assert!((cell.volume() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cell_volume_is_product_of_rule_volumes() {
        let cell = Cell::new()
            .with_rule("x1", continuous_rule(0.0, 10.0))
            .with_rule("x2", continuous_rule(0.0, 5.0));
        assert!((cell.volume() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn split_continuous_produces_valid_children() {
        let cell = Cell::new()
            .with_rule("x1", continuous_rule(0.0, 10.0))
            .with_rule("x2", continuous_rule(0.0, 5.0));

        let (left, right) = cell.split_continuous("x1", 3.0, true);

        // x1 split at 3.0: left [0,3), right [3,10]
        assert!((left.get_rule("x1").unwrap().volume() - 3.0).abs() < 1e-10);
        assert!((right.get_rule("x1").unwrap().volume() - 7.0).abs() < 1e-10);

        // x2 unchanged
        assert!((left.get_rule("x2").unwrap().volume() - 5.0).abs() < 1e-10);
        assert!((right.get_rule("x2").unwrap().volume() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn split_categorical_produces_valid_children() {
        let cell = Cell::new().with_rule("color", categorical_rule(vec![0, 1, 2, 3], 5));

        let subset: HashSet<usize> = [0, 1].into_iter().collect();
        let (left, right) = cell.split_categorical("color", subset, true);

        assert!((left.get_rule("color").unwrap().volume() - 2.0).abs() < 1e-10);
        assert!((right.get_rule("color").unwrap().volume() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn target_volume_only_considers_target_columns() {
        let cell = Cell::new()
            .with_rule("x1", continuous_rule(0.0, 10.0))
            .with_rule("target__y1", continuous_rule(0.0, 4.0));

        assert!((cell.target_volume() - 4.0).abs() < 1e-10);
        assert!((cell.volume() - 40.0).abs() < 1e-10);
    }

    #[test]
    fn feature_split_preserves_target_volume() {
        let cell = Cell::new()
            .with_rule("x1", continuous_rule(0.0, 10.0))
            .with_rule("target__y1", continuous_rule(0.0, 4.0));

        let (vol_l, vol_r) = cell.split_continuous_target_volumes("x1", 3.0, true);
        assert!((vol_l - 4.0).abs() < 1e-10);
        assert!((vol_r - 4.0).abs() < 1e-10);
    }

    #[test]
    fn target_split_changes_target_volume() {
        let cell = Cell::new()
            .with_rule("x1", continuous_rule(0.0, 10.0))
            .with_rule("target__y1", continuous_rule(0.0, 4.0));

        let (vol_l, vol_r) = cell.split_continuous_target_volumes("target__y1", 1.0, true);
        assert!((vol_l - 1.0).abs() < 1e-10);
        assert!((vol_r - 3.0).abs() < 1e-10);
    }
}
