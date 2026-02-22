//! Conditioned cell — a leaf cell projected to the target space.
//!
//! After conditioning on $X = x$, a leaf's feature rules are irrelevant.
//! A [`ConditionedCell`] retains only the target rules and the conditional
//! mass $m = w_{xy} / w_x$, providing a lightweight building block for
//! [`PiecewiseConstantDistribution`](super::piecewise_distribution::PiecewiseConstantDistribution).
use std::collections::HashMap;

use crate::conf::TARGET_PREFIX;
use crate::dataset_view::ColumnView;
use crate::rule::DynRule;
use crate::tree::FittedNode;

// ---------------------------------------------------------------------------
// ConditionedCell
// ---------------------------------------------------------------------------

/// A leaf cell projected to the target space with its conditional mass.
///
/// Constructed from a [`FittedNode`] by extracting only the `target__`-prefixed
/// rules and computing $m = w_{xy} / w_x$.
///
/// # Examples
///
/// ```rust,ignore
/// let cell = ConditionedCell::from_fitted_node(&node);
/// let vol = cell.target_volume();
/// let density = cell.conditional_density();
/// let means = cell.mean_map();
/// ```
#[derive(Debug, Clone)]
pub struct ConditionedCell {
    /// Target rules only (column name → DynRule).
    pub target_rules: HashMap<String, Box<dyn DynRule>>,
    /// Conditional mass: $w_{xy} / w_x$ for this cell.
    pub mass: f64,
}

impl ConditionedCell {
    /// Create a conditioned cell from a fitted tree node.
    ///
    /// Extracts only `target__`-prefixed rules and computes the
    /// conditional mass. If `w_x` is zero or negative, mass is set to `0.0`.
    pub fn from_fitted_node(node: &FittedNode) -> Self {
        let target_rules: HashMap<String, Box<dyn DynRule>> = node
            .cell
            .rules
            .iter()
            .filter(|(k, _)| k.starts_with(TARGET_PREFIX))
            .map(|(k, r)| (k.clone(), r.clone()))
            .collect();
        let mass = if node.w_x > 0.0 {
            node.w_xy / node.w_x
        } else {
            0.0
        };
        Self { target_rules, mass }
    }

    /// Create a conditioned cell directly from target rules and mass.
    pub fn new(target_rules: HashMap<String, Box<dyn DynRule>>, mass: f64) -> Self {
        Self { target_rules, mass }
    }

    /// Product of target rule volumes.
    ///
    /// Returns at least `1.0` when no target rules exist (unconstrained
    /// target space has unit measure by convention).
    pub fn target_volume(&self) -> f64 {
        self.target_rules
            .values()
            .map(|r| r.volume())
            .product::<f64>()
            .max(1.0)
    }

    /// Conditional density: $m / V_{\text{target}}$.
    ///
    /// Returns `0.0` when target volume is non-positive.
    pub fn conditional_density(&self) -> f64 {
        let vol = self.target_volume();
        if vol <= 0.0 { 0.0 } else { self.mass / vol }
    }

    /// Per-target-column mean vector from `DynRule::mean()`.
    ///
    /// For continuous targets this is `[(low + high) / 2]` (length 1).
    /// For categorical targets this is the one-hot indicator over the
    /// sorted domain (length `|domain|`).
    pub fn mean_map(&self) -> HashMap<String, Vec<f64>> {
        self.target_rules
            .iter()
            .map(|(k, r)| (k.clone(), r.mean()))
            .collect()
    }

    /// Check whether a data row (given by `row_idx` into `columns`)
    /// falls within this cell's target region.
    ///
    /// Only target columns present in `target_rules` are evaluated.
    /// Columns not found in the slice are skipped (treated as unconstrained).
    pub fn contains_target_row(&self, row_idx: usize, columns: &[&dyn ColumnView]) -> bool {
        for (col_name, rule) in &self.target_rules {
            let col = match columns.iter().find(|c| c.name() == col_name) {
                Some(c) => c,
                None => continue,
            };
            let value = col.get_dyn_value(row_idx);
            if !rule.contains(value.as_ref()) {
                return false;
            }
        }
        true
    }

    /// Number of target dimensions.
    pub fn n_target_dims(&self) -> usize {
        self.target_rules.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::ContinuousInterval;
    use crate::cell::Cell;
    use crate::rule::DynRule;
    use crate::tree::FittedNode;

    fn make_fitted_node_with_target() -> FittedNode {
        let cell = Cell::new()
            .with_rule(
                "x1",
                Box::new(ContinuousInterval::new(
                    0.0,
                    10.0,
                    true,
                    true,
                    Some((0.0, 10.0)),
                    true,
                )) as Box<dyn DynRule>,
            )
            .with_rule(
                "target__y1",
                Box::new(ContinuousInterval::new(
                    2.0,
                    6.0,
                    true,
                    true,
                    Some((0.0, 10.0)),
                    true,
                )) as Box<dyn DynRule>,
            );

        FittedNode {
            cell,
            w_xy: 80.0,
            w_x: 100.0,
            w_y: 50.0,
            depth: 1,
            parent: Some(0),
            left_child: None,
            right_child: None,
            is_leaf: true,
        }
    }

    #[test]
    fn from_fitted_node_extracts_target_rules() {
        let node = make_fitted_node_with_target();
        let cc = ConditionedCell::from_fitted_node(&node);

        // Only target rules retained
        assert_eq!(cc.n_target_dims(), 1);
        assert!(cc.target_rules.contains_key("target__y1"));
        assert!(!cc.target_rules.contains_key("x1"));
    }

    #[test]
    fn mass_is_conditional() {
        let node = make_fitted_node_with_target();
        let cc = ConditionedCell::from_fitted_node(&node);
        assert!((cc.mass - 0.8).abs() < 1e-10); // 80/100
    }

    #[test]
    fn target_volume_from_rules() {
        let node = make_fitted_node_with_target();
        let cc = ConditionedCell::from_fitted_node(&node);
        assert!((cc.target_volume() - 4.0).abs() < 1e-10); // 6.0 - 2.0
    }

    #[test]
    fn conditional_density_is_mass_over_volume() {
        let node = make_fitted_node_with_target();
        let cc = ConditionedCell::from_fitted_node(&node);
        // mass=0.8, vol=4.0 → density=0.2
        assert!((cc.conditional_density() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn mean_map_continuous_midpoint() {
        let node = make_fitted_node_with_target();
        let cc = ConditionedCell::from_fitted_node(&node);
        let means = cc.mean_map();
        let y1_mean = &means["target__y1"];
        assert_eq!(y1_mean.len(), 1);
        assert!((y1_mean[0] - 4.0).abs() < 1e-10); // (2+6)/2
    }

    #[test]
    fn zero_w_x_gives_zero_mass() {
        let mut node = make_fitted_node_with_target();
        node.w_x = 0.0;
        let cc = ConditionedCell::from_fitted_node(&node);
        assert!((cc.mass - 0.0).abs() < 1e-10);
    }
}
