//! Loss functions for partition tree split evaluation.
//!
//! A [`LossFunc`] computes the loss contribution of a single cell and the
//! information gain of a proposed split. Two built-in implementations are
//! provided:
//!
//! | Loss function            | Y-measure interpretation           |
//! |--------------------------|---------------------------------   |
//! | [`ConditionalLogLoss`]   | Geometric (Lebesgue volume)        |
//! | [`BalancedLogLoss`]      | Counting / empirical (`w_y`)       |
//!
//! ## Gain formula
//!
//! The gain is computed using a formula compatible with the v1 tree:
//!
//! $$\text{gain} = \frac{N_{xy}}{D}\left[
//!   \sum_{c \in \{L,R\}} \frac{w_{xy}^c}{N_{xy}}
//!   \ln\frac{w_{xy}^c}{w_x^c \cdot V_c}
//!   - \ln\frac{N_{xy}}{N_x \cdot V}\right]$$
//!
//! where $N_{xy} = w_{xy}^L + w_{xy}^R$, $N_x = w_x^L + w_x^R$, and
//! $D$ is the dataset size. For target splits both children carry the full
//! X-measure so $N_x = 2 \times w_x^{\text{parent}}$, which is why the
//! naïve `cell_loss(parent) − Σ cell_loss(child)` does not apply.
//!
//! # Implementing a custom loss
//!
//! ```rust,ignore
//! use partition_tree::v2::loss::{LossFunc, CellStats};
//!
//! struct MyLoss;
//!
//! impl LossFunc for MyLoss {
//!     fn cell_loss(&self, stats: &CellStats) -> f64 { /* … */ }
//!     // Optionally override `gain` for non-standard formulas.
//! }
//! ```

/// Aggregated statistics for a single cell in the partition tree.
///
/// Passed to [`LossFunc::cell_loss`] and [`LossFunc::gain`] to evaluate
/// split quality without holding references to the full node.
#[derive(Debug, Clone, Copy)]
pub struct CellStats {
    /// Weight of samples in the joint (X × Y) measure
    pub w_xy: f64,
    /// Weight of samples in the X measure
    pub w_x: f64,
    /// Weight of samples in the Y measure
    pub w_y: f64,
    /// Volume of the cell
    pub volume: f64,
}

impl CellStats {
    /// Create a new `CellStats` from raw weights and cell volume.
    pub fn new(w_xy: f64, w_x: f64, w_y: f64, volume: f64) -> Self {
        Self {
            w_xy,
            w_x,
            w_y,
            volume,
        }
    }
}

/// Contract for computing the loss of a cell and the information gain of a split.
///
/// # Convention
///
/// Loss is the quantity to **minimize**. The default gain is
/// `parent_loss − (left_loss + right_loss)`, but implementations may
/// override [`LossFunc::gain`] when the parent reconstruction from
/// children is non-trivial (e.g., target splits).
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` because column-level split
/// search is parallelized with `rayon`.
pub trait LossFunc: Send + Sync {
    /// Loss contribution of a single cell.
    ///
    /// Returns 0.0 when any weight or volume is non-positive.
    fn cell_loss(&self, stats: &CellStats) -> f64;

    /// Information gain from splitting `parent` into `left` and `right`.
    ///
    /// The default implementation subtracts child losses from the parent
    /// loss. Override this when the parent's effective statistics differ
    /// from the stored `CellStats` (see [`ConditionalLogLoss::gain`]).
    fn gain(&self, parent: &CellStats, left: &CellStats, right: &CellStats) -> f64 {
        self.cell_loss(parent) - (self.cell_loss(left) + self.cell_loss(right))
    }

    /// Whether the Y-measure is empirical (`w_y`) rather than geometric (Lebesgue volume).
    ///
    /// When `true`, categorical Y-splits compute per-category `b_c` as the
    /// sum of `weights_y` for samples in that category ([`BalancedLogLoss`]).
    /// When `false` (default), `b_c = 1.0` per category, corresponding to
    /// the Lebesgue/geometric volume ([`ConditionalLogLoss`]).
    fn uses_empirical_y_measure(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Conditional Log Loss
// ---------------------------------------------------------------------------

/// Conditional log-loss:
/// $$\ell(\text{cell}) = -\frac{w_{xy}}{D}\,\ln\!\Bigl(\frac{w_{xy}}{w_x \cdot V}\Bigr)$$
///
/// where $D$ is the dataset size and $V$ is the cell's target volume.
///
/// This loss is appropriate when the Y-measure is the Lebesgue measure
/// (continuous targets). The density being estimated is
/// $f(y|x) = w_{xy} / (w_x \cdot V)$.
#[derive(Debug, Clone)]
pub struct ConditionalLogLoss {
    /// Normalizing constant $D$ (typically `dataset.n_rows() as f64`).
    pub dataset_size: f64,
}

impl ConditionalLogLoss {
    /// Create a new `ConditionalLogLoss` with the given dataset size.
    pub fn new(dataset_size: f64) -> Self {
        Self { dataset_size }
    }
}

impl LossFunc for ConditionalLogLoss {
    fn cell_loss(&self, stats: &CellStats) -> f64 {
        let CellStats {
            w_xy, w_x, volume, ..
        } = *stats;

        if w_xy <= 0.0 || w_x <= 0.0 || volume <= 0.0 {
            return 0.0;
        }

        let density = w_xy / (w_x * volume);
        -w_xy * density.ln() / self.dataset_size
    }

    /// Gain computed using the v1-compatible formula:
    ///
    /// $$\text{gain} = \frac{N_{xy}}{D}\left[
    ///   \frac{w_{xy}^L}{N_{xy}} \ln\frac{w_{xy}^L}{w_x^L \cdot V_L}
    /// + \frac{w_{xy}^R}{N_{xy}} \ln\frac{w_{xy}^R}{w_x^R \cdot V_R}
    /// - \ln\frac{N_{xy}}{N_x \cdot V}\right]$$
    ///
    /// where $N_{xy} = w_{xy}^L + w_{xy}^R$, $N_x = w_x^L + w_x^R$,
    /// $V$ is the parent volume.
    ///
    /// This differs from the naïve `parent_loss − sum(child_losses)` for
    /// target splits, where both children carry the full X-measure and
    /// $N_x = 2 \times w_x^{\text{parent}}$.
    fn gain(&self, parent: &CellStats, left: &CellStats, right: &CellStats) -> f64 {
        let total_xy = left.w_xy + right.w_xy;
        let total_x = left.w_x + right.w_x;

        if total_xy <= 0.0 || total_x <= 0.0 || parent.volume <= 0.0 {
            return 0.0;
        }

        // Children log: weighted average of child log-densities
        let mut children_log = 0.0;
        for child in [left, right] {
            if child.w_xy > 0.0 && child.w_x > 0.0 && child.volume > 0.0 {
                let density = child.w_xy / (child.w_x * child.volume);
                children_log += (child.w_xy / total_xy) * density.ln();
            }
        }

        // Parent log: using total_x = w_x_left + w_x_right
        let parent_density = total_xy / (total_x * parent.volume);
        let parent_log = parent_density.ln();

        (total_xy / self.dataset_size) * (children_log - parent_log)
    }
}

// ---------------------------------------------------------------------------
// Balanced Log Loss
// ---------------------------------------------------------------------------

/// Balanced log-loss:
/// $$\ell(\text{cell}) = -\frac{w_{xy}}{D}\,\ln\!\Bigl(\frac{w_{xy}}{w_x \cdot w_y}\Bigr)$$
///
/// Replaces the geometric target volume with the empirical Y-weight
/// $w_y$. This is suitable for categorical targets or situations where
/// a counting measure on Y is preferred over the Lebesgue measure.
#[derive(Debug, Clone)]
pub struct BalancedLogLoss {
    /// Normalizing constant $D$ (typically `dataset.n_rows() as f64`).
    pub dataset_size: f64,
}

impl BalancedLogLoss {
    /// Create a new `BalancedLogLoss` with the given dataset size.
    pub fn new(dataset_size: f64) -> Self {
        Self { dataset_size }
    }
}

impl LossFunc for BalancedLogLoss {
    fn uses_empirical_y_measure(&self) -> bool {
        true
    }

    fn cell_loss(&self, stats: &CellStats) -> f64 {
        let CellStats { w_xy, w_x, w_y, .. } = *stats;

        if w_xy <= 0.0 || w_x <= 0.0 || w_y <= 0.0 {
            return 0.0;
        }

        let ratio = w_xy / (w_x * w_y);
        -w_xy * ratio.ln() / self.dataset_size
    }

    /// Gain using the v1-compatible formula with $w_y$ replacing volume.
    ///
    /// Analogous to [`ConditionalLogLoss::gain`] but uses
    /// $\text{total}_y = w_y^L + w_y^R$ instead of geometric volumes.
    fn gain(&self, parent: &CellStats, left: &CellStats, right: &CellStats) -> f64 {
        let total_xy = left.w_xy + right.w_xy;
        let total_x = left.w_x + right.w_x;
        let total_y = left.w_y + right.w_y;

        if total_xy <= 0.0 || total_x <= 0.0 || total_y <= 0.0 {
            return 0.0;
        }

        let mut children_log = 0.0;
        for child in [left, right] {
            if child.w_xy > 0.0 && child.w_x > 0.0 && child.w_y > 0.0 {
                let ratio = child.w_xy / (child.w_x * child.w_y);
                children_log += (child.w_xy / total_xy) * ratio.ln();
            }
        }

        let parent_ratio = total_xy / (total_x * total_y);
        let parent_log = parent_ratio.ln();

        (total_xy / self.dataset_size) * (children_log - parent_log)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn conditional_log_loss_zero_weights() {
        let loss = ConditionalLogLoss::new(100.0);
        let stats = CellStats::new(0.0, 10.0, 5.0, 1.0);
        assert_eq!(loss.cell_loss(&stats), 0.0);
    }

    #[test]
    fn conditional_log_loss_basic() {
        let loss = ConditionalLogLoss::new(100.0);
        // density = 50 / (100 * 2.0) = 0.25
        // cell_loss = -50 * ln(0.25) / 100 = -0.5 * ln(0.25)
        let stats = CellStats::new(50.0, 100.0, 50.0, 2.0);
        let result = loss.cell_loss(&stats);
        let expected = -50.0 * (0.25_f64).ln() / 100.0;
        assert!(
            approx_eq(result, expected, 1e-10),
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn balanced_log_loss_basic() {
        let loss = BalancedLogLoss::new(100.0);
        // ratio = 50 / (100 * 50) = 0.01
        // cell_loss = -50 * ln(0.01) / 100
        let stats = CellStats::new(50.0, 100.0, 50.0, 2.0);
        let result = loss.cell_loss(&stats);
        let expected = -50.0 * (0.01_f64).ln() / 100.0;
        assert!(
            approx_eq(result, expected, 1e-10),
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn gain_feature_split_matches_parent_minus_children() {
        // For feature splits, w_x_left + w_x_right == parent.w_x,
        // so the v1-compatible gain formula equals cell_loss(parent) - sum(cell_loss(children)).
        let loss = ConditionalLogLoss::new(100.0);
        let parent = CellStats::new(100.0, 200.0, 100.0, 4.0);
        let left = CellStats::new(60.0, 100.0, 60.0, 2.0);
        let right = CellStats::new(40.0, 100.0, 40.0, 2.0);

        let gain = loss.gain(&parent, &left, &right);
        let expected = loss.cell_loss(&parent) - (loss.cell_loss(&left) + loss.cell_loss(&right));
        assert!(
            approx_eq(gain, expected, 1e-10),
            "got {gain}, expected {expected}"
        );
    }

    #[test]
    fn gain_target_split_matches_v1() {
        // For a target split on 200 samples with 50/50 categorical target:
        // Both children carry the full X-measure.
        // v1 gain = ln(2) ≈ 0.693147
        let loss = ConditionalLogLoss::new(200.0);
        let parent = CellStats::new(200.0, 200.0, 200.0, 2.0);
        let left = CellStats::new(100.0, 200.0, 100.0, 1.0);
        let right = CellStats::new(100.0, 200.0, 100.0, 1.0);

        let gain = loss.gain(&parent, &left, &right);
        let expected = 2.0_f64.ln(); // ln(2) ≈ 0.693147
        assert!(
            approx_eq(gain, expected, 1e-6),
            "got {gain}, expected {expected}"
        );
    }

    #[test]
    fn conditional_loss_uses_geometric_y_measure() {
        let loss = ConditionalLogLoss::new(100.0);
        assert!(
            !loss.uses_empirical_y_measure(),
            "ConditionalLogLoss should use geometric (Lebesgue) Y-measure"
        );
    }

    #[test]
    fn balanced_loss_uses_empirical_y_measure() {
        let loss = BalancedLogLoss::new(100.0);
        assert!(
            loss.uses_empirical_y_measure(),
            "BalancedLogLoss should use empirical Y-measure"
        );
    }
}
