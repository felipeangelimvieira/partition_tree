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
//! [`Cell::apply_split`] produces `(left_cell, right_cell)` pairs by
//! delegating to a [`SplitOp`](super::split_result::SplitOp) — the dtype-
//! specific logic is fully encapsulated in the split operation, so adding a
//! new dtype requires no changes here.
//!
//! The companion [`Cell::child_target_volumes`] computes child volumes
//! *without* constructing the full child cells, which is used during the
//! split search inner loop.
use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::conf::TARGET_PREFIX;

use crate::rule::DynRule;
use crate::serde::rules as rules_serde;
use crate::split_result::SplitOp;

/// Multi-dimensional partition constraint: one [`DynRule`] per column.
///
/// The cell is the fundamental building block of the partition tree. Each
/// tree node owns a `Cell` that describes the region of space it represents.
///
/// # Examples
///
/// ```rust,ignore
/// let cell = Cell::new()
///     .with_rule("x1", Box::new(continuous_rule(0.0, 10.0)))
///     .with_rule("target__y", Box::new(continuous_rule(0.0, 5.0)));
///
/// assert!((cell.volume() - 50.0).abs() < 1e-10);
/// assert!((cell.target_volume() - 5.0).abs() < 1e-10);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    #[serde(with = "rules_serde")]
    pub rules: HashMap<String, Box<dyn DynRule>>,
}

impl Cell {
    /// Create an empty cell (unconstrained in all dimensions).
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Create a cell from a pre-built map of rules.
    pub fn from_rules(rules: HashMap<String, Box<dyn DynRule>>) -> Self {
        Self { rules }
    }

    /// Builder: add a rule for `col` and return self.
    pub fn with_rule(mut self, col: impl Into<String>, rule: Box<dyn DynRule>) -> Self {
        self.rules.insert(col.into(), rule);
        self
    }

    /// Insert or replace a rule for `col`.
    pub fn set_rule(&mut self, col: impl Into<String>, rule: Box<dyn DynRule>) {
        self.rules.insert(col.into(), rule);
    }

    /// Get the rule for a column, if present.
    pub fn get_rule(&self, col: &str) -> Option<&dyn DynRule> {
        self.rules.get(col).map(|r| r.as_ref())
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

    /// Domain volume of the target space: product of `domain_volume()` for
    /// each `target__`-prefixed rule.
    ///
    /// This reflects the full target domain (as encoded in each rule's domain
    /// bounds), regardless of how much of it the cell currently covers.
    /// Returns 1.0 when no target rules exist.
    pub fn target_domain_volume(&self) -> f64 {
        self.target_rules()
            .map(|(_, r)| r.domain_volume())
            .product::<f64>()
            .max(1.0)
    }

    /// Whether the cell has any continuous target rules.
    pub fn has_continuous_target(&self) -> bool {
        self.target_rules().any(|(_, r)| r.is_continuous())
    }

    /// Target volume considering only continuous coordinates.
    ///
    /// Returns `None` when no continuous target rules exist.
    pub fn target_continuous_volume(&self) -> Option<f64> {
        if !self.has_continuous_target() {
            return None;
        }
        Some(
            self.target_rules()
                .filter(|(_, r)| r.is_continuous())
                .map(|(_, r)| r.volume())
                .product::<f64>(),
        )
    }

    /// Domain volume of the target space considering only continuous coordinates.
    ///
    /// Returns `None` when no continuous target rules exist.
    pub fn target_continuous_domain_volume(&self) -> Option<f64> {
        if !self.has_continuous_target() {
            return None;
        }
        Some(
            self.target_rules()
                .filter(|(_, r)| r.is_continuous())
                .map(|(_, r)| r.domain_volume())
                .product::<f64>(),
        )
    }

    /// Compute child **continuous target volumes** for a hypothetical split.
    ///
    /// Like [`child_target_volumes`](Self::child_target_volumes) but only
    /// considers continuous target coordinates for the volume fraction
    /// criterion. Returns `None` when no continuous target rules exist.
    pub fn child_target_continuous_volumes(
        &self,
        col: &str,
        op: &dyn SplitOp,
        none_to_left: bool,
    ) -> Option<(f64, f64)> {
        if !self.has_continuous_target() {
            return None;
        }

        if !col.starts_with(TARGET_PREFIX) {
            let v = self.target_continuous_volume().unwrap();
            return Some((v, v));
        }

        let parent_rule = self.rules.get(col).expect("column not found");

        // If the split column is not continuous, the continuous volume is unchanged.
        if !parent_rule.is_continuous() {
            let v = self.target_continuous_volume().unwrap();
            return Some((v, v));
        }

        let (left_vol, right_vol) = op.child_volumes(parent_rule.as_ref(), none_to_left);

        let compute_vol = |replacement_vol: f64| -> f64 {
            self.target_rules()
                .filter(|(_, r)| r.is_continuous())
                .map(|(k, r)| {
                    if k.as_str() == col {
                        replacement_vol
                    } else {
                        r.volume()
                    }
                })
                .product::<f64>()
                .max(f64::MIN_POSITIVE)
        };

        Some((compute_vol(left_vol), compute_vol(right_vol)))
    }

    /// Target-only phi-volume.
    pub fn target_phi_volume(&self) -> f64 {
        self.target_rules()
            .map(|(_, r)| r.phi_volume())
            .product::<f64>()
            .max(1.0)
    }

    /// Iterate over rules on target columns (prefixed with `target__`).
    pub fn target_rules(&self) -> impl Iterator<Item = (&String, &dyn DynRule)> {
        self.rules
            .iter()
            .filter(|(k, _)| k.starts_with(TARGET_PREFIX))
            .map(|(k, r)| (k, r.as_ref()))
    }

    /// Iterate over rules on feature columns (not prefixed with `target__`).
    pub fn feature_rules(&self) -> impl Iterator<Item = (&String, &dyn DynRule)> {
        self.rules
            .iter()
            .filter(|(k, _)| !k.starts_with(TARGET_PREFIX))
            .map(|(k, r)| (k, r.as_ref()))
    }

    /// Number of constrained dimensions.
    pub fn n_rules(&self) -> usize {
        self.rules.len()
    }

    /// Split this cell on `col` using the given [`SplitOp`].
    ///
    /// Returns `(left_cell, right_cell)` where the split column's rule
    /// is replaced by the left/right rules produced by the `SplitOp`,
    /// and all other rules are cloned unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `col` is not present in the cell's rules.
    pub fn apply_split(&self, col: &str, op: &dyn SplitOp, none_to_left: bool) -> (Cell, Cell) {
        let parent_rule = self
            .rules
            .get(col)
            .unwrap_or_else(|| panic!("Cell::apply_split: column '{col}' not found"));

        let (left_rule, right_rule) = op.split_rule(parent_rule.as_ref(), none_to_left);

        let mut left_rules = self.rules.clone();
        left_rules.insert(col.to_string(), left_rule);

        let mut right_rules = self.rules.clone();
        right_rules.insert(col.to_string(), right_rule);

        (Cell::from_rules(left_rules), Cell::from_rules(right_rules))
    }

    /// Compute child **target volumes** for a hypothetical split
    /// without constructing the child cells.
    ///
    /// If `col` is a feature column (not `target__`-prefixed), both children
    /// have the same target volume as the parent. Otherwise, the target
    /// volume is recomputed with the split column's rule replaced.
    ///
    /// This is used in the split search inner loop to avoid allocating
    /// child cells at every candidate threshold.
    pub fn child_target_volumes(
        &self,
        col: &str,
        op: &dyn SplitOp,
        none_to_left: bool,
    ) -> (f64, f64) {
        // Only target columns affect volume; feature splits leave volume unchanged.
        if !col.starts_with(TARGET_PREFIX) {
            let v = self.target_volume();
            return (v, v);
        }

        let parent_rule = self.rules.get(col).expect("column not found");
        let (left_vol, right_vol) = op.child_volumes(parent_rule.as_ref(), none_to_left);

        // Recompute target volume replacing only the split column's volume
        let compute_vol = |replacement_vol: f64| -> f64 {
            self.target_rules()
                .map(|(k, r)| {
                    if k.as_str() == col {
                        replacement_vol
                    } else {
                        r.volume()
                    }
                })
                .product::<f64>()
                .max(f64::MIN_POSITIVE)
        };

        (compute_vol(left_vol), compute_vol(right_vol))
    }

    /// Per-column mean vector, concatenated. Used for prediction decoding.
    pub fn mean_map(&self) -> HashMap<String, Vec<f64>> {
        let mut keys: Vec<&String> = self.rules.keys().collect();
        keys.sort();
        keys.into_iter()
            .map(|k| (k.clone(), self.rules[k].mean()))
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
    use crate::split_result::{CategoricalSplitOp, ContinuousSplitOp};
    use std::collections::HashSet;
    use std::sync::Arc;

    fn continuous_rule(low: f64, high: f64) -> Box<dyn DynRule> {
        Box::new(ContinuousInterval::new(
            low,
            high,
            true,
            true,
            Some((low, high)),
            true,
        ))
    }

    fn categorical_rule(values: Vec<usize>, domain_size: usize) -> Box<dyn DynRule> {
        let domain: Vec<usize> = (0..domain_size).collect();
        let names: Vec<String> = domain.iter().map(|i| format!("cat_{i}")).collect();
        Box::new(BelongsTo::new(
            values.into_iter().collect(),
            Arc::new(domain),
            Arc::new(names),
            true,
        ))
    }

    #[test]
    fn empty_cell_volume_is_one() {
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
    fn apply_split_continuous_produces_valid_children() {
        let cell = Cell::new()
            .with_rule("x1", continuous_rule(0.0, 10.0))
            .with_rule("x2", continuous_rule(0.0, 5.0));

        let op = ContinuousSplitOp {
            threshold: 3.0,
            k_candidate: 0,
            p_xy: 0,
        };
        let (left, right) = cell.apply_split("x1", &op, true);

        // x1 split at 3.0: left [0,3), right [3,10]
        assert!((left.get_rule("x1").unwrap().volume() - 3.0).abs() < 1e-10);
        assert!((right.get_rule("x1").unwrap().volume() - 7.0).abs() < 1e-10);

        // x2 unchanged
        assert!((left.get_rule("x2").unwrap().volume() - 5.0).abs() < 1e-10);
        assert!((right.get_rule("x2").unwrap().volume() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn apply_split_categorical_produces_valid_children() {
        let cell = Cell::new().with_rule("color", categorical_rule(vec![0, 1, 2, 3], 5));

        let subset: HashSet<usize> = [0, 1].into_iter().collect();
        let op = CategoricalSplitOp {
            subset_left: subset,
        };
        let (left, right) = cell.apply_split("color", &op, true);

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

        let op = ContinuousSplitOp {
            threshold: 3.0,
            k_candidate: 0,
            p_xy: 0,
        };
        let (vol_l, vol_r) = cell.child_target_volumes("x1", &op, true);
        assert!((vol_l - 4.0).abs() < 1e-10);
        assert!((vol_r - 4.0).abs() < 1e-10);
    }

    #[test]
    fn target_split_changes_target_volume() {
        let cell = Cell::new()
            .with_rule("x1", continuous_rule(0.0, 10.0))
            .with_rule("target__y1", continuous_rule(0.0, 4.0));

        let op = ContinuousSplitOp {
            threshold: 1.0,
            k_candidate: 0,
            p_xy: 0,
        };
        let (vol_l, vol_r) = cell.child_target_volumes("target__y1", &op, true);
        assert!((vol_l - 1.0).abs() < 1e-10);
        assert!((vol_r - 3.0).abs() < 1e-10);
    }
}
