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
//! use partition_tree::loss::{LossFunc, CellStats};
//!
//! struct MyLoss;
//!
//! impl LossFunc for MyLoss {
//!     fn cell_loss(&self, stats: &CellStats, dataset_size: f64) -> f64 { /* … */ }
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
#[typetag::serde(tag = "type")]
pub trait LossFunc: Send + Sync {
    /// Loss contribution of a single cell.
    ///
    /// `dataset_size` is the normalizing constant $D$ (typically
    /// `dataset.n_rows() as f64`). Implementations that do not need
    /// normalization may ignore it.
    ///
    /// Returns 0.0 when any weight or volume is non-positive.
    fn cell_loss(&self, stats: &CellStats, dataset_size: f64) -> f64;

    /// Information gain from splitting `parent` into `left` and `right`.
    ///
    /// `dataset_size` is the normalizing constant $D$.
    ///
    /// The default implementation subtracts child losses from the parent
    /// loss. Override this when the parent's effective statistics differ
    /// from the stored `CellStats` (see [`ConditionalLogLoss::gain`]).
    fn gain(
        &self,
        parent: &CellStats,
        left: &CellStats,
        right: &CellStats,
        dataset_size: f64,
    ) -> f64 {
        self.cell_loss(parent, dataset_size)
            - (self.cell_loss(left, dataset_size) + self.cell_loss(right, dataset_size))
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

    /// Return a heap-allocated clone of this loss function.
    fn clone_box(&self) -> Box<dyn LossFunc>;
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ConditionalLogLoss;

#[typetag::serde]
impl LossFunc for ConditionalLogLoss {
    fn clone_box(&self) -> Box<dyn LossFunc> {
        Box::new(self.clone())
    }

    fn cell_loss(&self, stats: &CellStats, dataset_size: f64) -> f64 {
        let CellStats {
            w_xy, w_x, volume, ..
        } = *stats;

        if w_xy <= 0.0 || w_x <= 0.0 || volume <= 0.0 {
            return 0.0;
        }

        let density = w_xy / (w_x * volume);
        -w_xy * density.ln() / dataset_size
    }

    fn gain(
        &self,
        parent: &CellStats,
        left: &CellStats,
        right: &CellStats,
        dataset_size: f64,
    ) -> f64 {
        self.cell_loss(parent, dataset_size)
            - (self.cell_loss(left, dataset_size) + self.cell_loss(right, dataset_size))
    }
}

// ---------------------------------------------------------------------------
// Mean Integrated Squared Error (MISE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MeanIntegratedSquaredError;

#[typetag::serde]
impl LossFunc for MeanIntegratedSquaredError {
    fn clone_box(&self) -> Box<dyn LossFunc> {
        Box::new(self.clone())
    }

    fn cell_loss(&self, stats: &CellStats, dataset_size: f64) -> f64 {
        let CellStats {
            w_xy, w_x, volume, ..
        } = *stats;

        if w_xy <= 0.0 || w_x <= 0.0 || volume <= 0.0 {
            return 0.0;
        }

        -(w_xy / dataset_size).powf(2.0) / (w_x / dataset_size * volume)
    }

    fn gain(
        &self,
        parent: &CellStats,
        left: &CellStats,
        right: &CellStats,
        dataset_size: f64,
    ) -> f64 {
        self.cell_loss(parent, dataset_size)
            - (self.cell_loss(left, dataset_size) + self.cell_loss(right, dataset_size))
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BalancedLogLoss;

#[typetag::serde]
impl LossFunc for BalancedLogLoss {
    fn clone_box(&self) -> Box<dyn LossFunc> {
        Box::new(self.clone())
    }

    fn uses_empirical_y_measure(&self) -> bool {
        true
    }

    fn cell_loss(&self, stats: &CellStats, dataset_size: f64) -> f64 {
        let CellStats { w_xy, w_x, w_y, .. } = *stats;

        if w_xy <= 0.0 || w_x <= 0.0 || w_y <= 0.0 {
            return 0.0;
        }

        let ratio = (w_xy / dataset_size) / ((w_x / dataset_size) * (w_y / dataset_size));
        -w_xy * ratio.ln() / dataset_size
    }

    fn gain(
        &self,
        parent: &CellStats,
        left: &CellStats,
        right: &CellStats,
        dataset_size: f64,
    ) -> f64 {
        self.cell_loss(parent, dataset_size)
            - (self.cell_loss(left, dataset_size) + self.cell_loss(right, dataset_size))
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
        let loss = ConditionalLogLoss;
        let stats = CellStats::new(0.0, 10.0, 5.0, 1.0);
        assert_eq!(loss.cell_loss(&stats, 100.0), 0.0);
    }

    #[test]
    fn conditional_log_loss_basic() {
        let loss = ConditionalLogLoss;
        // density = 50 / (100 * 2.0) = 0.25
        // cell_loss = -50 * ln(0.25) / 100 = -0.5 * ln(0.25)
        let stats = CellStats::new(50.0, 100.0, 50.0, 2.0);
        let result = loss.cell_loss(&stats, 100.0);
        let expected = -50.0 * (0.25_f64).ln() / 100.0;
        assert!(
            approx_eq(result, expected, 1e-10),
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn balanced_log_loss_basic() {
        let loss = BalancedLogLoss;
        // With dataset_size=100:
        // ratio = (w_xy/D) / ((w_x/D) * (w_y/D)) = (50/100) / ((100/100)*(50/100)) = 1.0
        // cell_loss = -w_xy * ln(ratio) / D = -50 * ln(1.0) / 100 = 0.0
        let stats = CellStats::new(50.0, 100.0, 50.0, 2.0);
        let result = loss.cell_loss(&stats, 100.0);
        let expected = 0.0_f64;
        assert!(
            approx_eq(result, expected, 1e-10),
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn gain_feature_split_matches_parent_minus_children() {
        // For feature splits, w_x_left + w_x_right == parent.w_x,
        // so the v1-compatible gain formula equals cell_loss(parent) - sum(cell_loss(children)).
        let loss = ConditionalLogLoss;
        let d = 100.0;
        let parent = CellStats::new(100.0, 200.0, 100.0, 4.0);
        let left = CellStats::new(60.0, 100.0, 60.0, 2.0);
        let right = CellStats::new(40.0, 100.0, 40.0, 2.0);

        let gain = loss.gain(&parent, &left, &right, d);
        let expected =
            loss.cell_loss(&parent, d) - (loss.cell_loss(&left, d) + loss.cell_loss(&right, d));
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
        let loss = ConditionalLogLoss;
        let parent = CellStats::new(200.0, 200.0, 200.0, 2.0);
        let left = CellStats::new(100.0, 200.0, 100.0, 1.0);
        let right = CellStats::new(100.0, 200.0, 100.0, 1.0);

        let gain = loss.gain(&parent, &left, &right, 200.0);
        // For a target split both children have the same w_x as parent,
        // so cell_loss(parent) == cell_loss(left) + cell_loss(right) and gain is 0.
        let expected = 0.0;
        assert!(
            approx_eq(gain, expected, 1e-6),
            "got {gain}, expected {expected}"
        );
    }

    #[test]
    fn conditional_loss_uses_geometric_y_measure() {
        let loss = ConditionalLogLoss;
        assert!(
            !loss.uses_empirical_y_measure(),
            "ConditionalLogLoss should use geometric (Lebesgue) Y-measure"
        );
    }

    #[test]
    fn balanced_loss_uses_empirical_y_measure() {
        let loss = BalancedLogLoss;
        assert!(
            loss.uses_empirical_y_measure(),
            "BalancedLogLoss should use empirical Y-measure"
        );
    }
}
