//! Piecewise-constant conditional distribution over the target space.
//!
//! A [`PiecewiseConstantDistribution`] represents $\hat{P}(Y | X = x)$ as a
//! collection of [`ConditionedCell`]s, each carrying a conditional mass and
//! target-space rules.
//!
//! ## Key operations
//!
//! - [`mean_vector`](PiecewiseConstantDistribution::mean_vector) — weighted
//!   mean across cells, returning a fixed-shape [`MeanVector`] suitable for
//!   ensemble averaging.
//! - [`ensemble`](PiecewiseConstantDistribution::ensemble) — merge multiple
//!   distributions by concatenating cells with re-weighted masses.
//! - [`pdf_segments`](PiecewiseConstantDistribution::pdf_segments) — extract
//!   `(density, low, high)` segments for continuous targets.
//! - [`category_probabilities`](PiecewiseConstantDistribution::category_probabilities)
//!   — per-category probability for categorical targets.
use std::collections::HashMap;

use crate::rules::{BelongsTo, ContinuousInterval, Rule};
use crate::dataset_view::ColumnView;

use super::conditioned_cell::ConditionedCell;

// ---------------------------------------------------------------------------
// MeanVector type alias
// ---------------------------------------------------------------------------

/// Per-target-column mean vector.
///
/// - Key: target column name (e.g., `"target__y1"`).
/// - Value: `Vec<f64>` whose semantics depend on dtype:
///   - **Continuous**: `[midpoint]` (length 1)
///   - **Integer**: `[midpoint]` (length 1)
///   - **Categorical**: `[p_0, p_1, …, p_{K−1}]` (probability over sorted domain)
pub type MeanVector = HashMap<String, Vec<f64>>;

// ---------------------------------------------------------------------------
// PiecewiseConstantDistribution
// ---------------------------------------------------------------------------

/// Piecewise-constant conditional distribution over the target space.
///
/// One instance per sample row — represents $\hat{P}(Y | X = x)$ as a union
/// of conditioned cells from matched tree leaves.
///
/// In a single tree each row matches exactly one leaf, so `cells.len() == 1`.
/// After ensembling, multiple cells may be present.
///
/// # Examples
///
/// ```rust,ignore
/// let dist = PiecewiseConstantDistribution::from_cells(vec![cell]);
/// let mean = dist.mean_vector();
/// ```
#[derive(Debug, Clone)]
pub struct PiecewiseConstantDistribution {
    /// Conditioned cells from matched leaves.
    pub cells: Vec<ConditionedCell>,
}

impl PiecewiseConstantDistribution {
    /// Create from a vector of conditioned cells.
    pub fn from_cells(cells: Vec<ConditionedCell>) -> Self {
        Self { cells }
    }

    /// Create from a single conditioned cell.
    pub fn from_single(cell: ConditionedCell) -> Self {
        Self { cells: vec![cell] }
    }

    /// Number of cells in this distribution.
    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// Total mass: $\sum_c m_c$.
    pub fn total_mass(&self) -> f64 {
        self.cells.iter().map(|c| c.mass).sum()
    }

    /// Weighted mean vector across all cells.
    ///
    /// For each target column, computes:
    /// $$\text{mean}[\text{col}] = \sum_c \frac{m_c}{\sum_c m_c} \cdot \text{rule}_c[\text{col}].\text{mean}()$$
    ///
    /// Falls back to uniform weighting when total mass is zero.
    pub fn mean_vector(&self) -> MeanVector {
        if self.cells.is_empty() {
            return MeanVector::new();
        }

        let total_mass = self.total_mass();
        let uniform = total_mass <= 0.0;
        let n = self.cells.len() as f64;

        let mut result = MeanVector::new();

        for cell in &self.cells {
            let weight = if uniform {
                1.0 / n
            } else {
                cell.mass / total_mass
            };
            let cell_means = cell.mean_map();

            for (col, mean_vec) in cell_means {
                result
                    .entry(col)
                    .and_modify(|acc: &mut Vec<f64>| {
                        for (j, v) in mean_vec.iter().enumerate() {
                            if j < acc.len() {
                                acc[j] += v * weight;
                            }
                        }
                    })
                    .or_insert_with(|| mean_vec.iter().map(|v| v * weight).collect());
            }
        }

        result
    }

    /// Evaluate the piecewise-constant pdf at target points.
    ///
    /// For each row index, finds the cell whose target rules contain the
    /// point and returns `mass / (total_mass * target_volume)`.
    /// Returns `0.0` for points that match no cell.
    pub fn pdf(&self, target_columns: &[&dyn ColumnView], row_indices: &[usize]) -> Vec<f64> {
        let total_mass = self.total_mass();
        if total_mass <= 0.0 {
            return vec![0.0; row_indices.len()];
        }

        row_indices
            .iter()
            .map(|&row_idx| {
                let mut pdf_val = 0.0;
                for cell in &self.cells {
                    if cell.contains_target_row(row_idx, target_columns) {
                        let vol = cell.target_volume();
                        if vol > 0.0 {
                            pdf_val += cell.mass / (total_mass * vol);
                        }
                    }
                }
                pdf_val
            })
            .collect()
    }

    /// Evaluate mass at target points (pdf without dividing by volume).
    ///
    /// For each row index, finds matching cells and returns
    /// `mass / total_mass`. Returns `0.0` for unmatched points.
    pub fn mass_at(&self, target_columns: &[&dyn ColumnView], row_indices: &[usize]) -> Vec<f64> {
        let total_mass = self.total_mass();
        if total_mass <= 0.0 {
            return vec![0.0; row_indices.len()];
        }

        row_indices
            .iter()
            .map(|&row_idx| {
                let mut mass_val = 0.0;
                for cell in &self.cells {
                    if cell.contains_target_row(row_idx, target_columns) {
                        mass_val += cell.mass / total_mass;
                    }
                }
                mass_val
            })
            .collect()
    }

    /// Create a merged distribution from multiple distributions (for ensembling).
    ///
    /// Each source distribution's masses are divided by the number of sources,
    /// preserving the full piecewise structure for downstream pdf/mass queries.
    pub fn ensemble(distributions: &[&PiecewiseConstantDistribution]) -> Self {
        if distributions.is_empty() {
            return Self::from_cells(Vec::new());
        }

        let k = distributions.len() as f64;
        let mut all_cells = Vec::new();

        for dist in distributions {
            for cell in &dist.cells {
                all_cells.push(ConditionedCell::new(
                    cell.target_rules.clone(),
                    cell.mass / k,
                ));
            }
        }

        Self::from_cells(all_cells)
    }

    /// Iterator over conditioned cells.
    pub fn iter(&self) -> impl Iterator<Item = &ConditionedCell> {
        self.cells.iter()
    }

    /// For continuous targets: extract `(density, low, high)` segments.
    ///
    /// Only considers the first `target__`-prefixed rule that is a
    /// `ContinuousInterval`. Cells without a continuous target are skipped.
    pub fn pdf_segments(&self) -> Vec<(f64, f64, f64)> {
        let total_mass = self.total_mass();
        let uniform = total_mass <= 0.0;
        let n = self.cells.len() as f64;

        let mut segments = Vec::with_capacity(self.cells.len());

        for cell in &self.cells {
            // Find the first continuous target rule
            for (_, rule) in &cell.target_rules {
                if let Some(ci) = rule.as_any().downcast_ref::<ContinuousInterval>() {
                    let vol = ci.volume();
                    if vol <= 0.0 {
                        continue;
                    }
                    let density = if uniform {
                        1.0 / (vol * n)
                    } else {
                        cell.mass / (total_mass * vol)
                    };
                    segments.push((density, ci.low, ci.high));
                    break; // only first continuous target per cell
                }
            }
        }

        segments
    }

    /// For categorical targets: return per-column probability vectors.
    ///
    /// Each entry maps a target column name to a probability vector aligned
    /// with the sorted domain of that column's `BelongsTo` rule.
    ///
    /// This is equivalent to extracting the categorical components of
    /// [`mean_vector`](Self::mean_vector).
    pub fn category_probabilities(&self) -> HashMap<String, Vec<f64>> {
        let mean = self.mean_vector();
        // Filter to only categorical targets (those with len > 1 in mean vector,
        // or backed by a BelongsTo rule).
        mean.into_iter()
            .filter(|(_, v)| {
                // Categorical means have length > 1 (one per domain category).
                // Continuous/integer means have length 1.
                v.len() > 1
            })
            .collect()
    }

    /// Get domain names for a categorical target column.
    ///
    /// Extracts `domain_names` from the `BelongsTo` rule of the first cell
    /// that has this column. Returns `None` if the column is not categorical
    /// or not present.
    pub fn categorical_domain_names(&self, col_name: &str) -> Option<Vec<String>> {
        for cell in &self.cells {
            if let Some(rule) = cell.target_rules.get(col_name) {
                if let Some(bt) = rule.as_any().downcast_ref::<BelongsTo>() {
                    return Some(bt.domain_names.as_ref().clone());
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::ContinuousInterval;
    use crate::rule::DynRule;
    use std::collections::HashSet;
    use std::sync::Arc;

    fn make_continuous_cell(low: f64, high: f64, mass: f64) -> ConditionedCell {
        let mut target_rules: HashMap<String, Box<dyn DynRule>> = HashMap::new();
        target_rules.insert(
            "target__y1".to_string(),
            Box::new(ContinuousInterval::new(
                low,
                high,
                true,
                true,
                Some((0.0, 10.0)),
                true,
            )),
        );
        ConditionedCell::new(target_rules, mass)
    }

    fn make_categorical_cell(active: Vec<usize>, domain_size: usize, mass: f64) -> ConditionedCell {
        let domain: Vec<usize> = (0..domain_size).collect();
        let names: Vec<String> = domain.iter().map(|i| format!("cat_{i}")).collect();
        let values: HashSet<usize> = active.into_iter().collect();
        let bt = BelongsTo::new(values, Arc::new(domain), Arc::new(names), true);

        let mut target_rules: HashMap<String, Box<dyn DynRule>> = HashMap::new();
        target_rules.insert("target__color".to_string(), Box::new(bt));
        ConditionedCell::new(target_rules, mass)
    }

    #[test]
    fn single_cell_mean_vector() {
        let cell = make_continuous_cell(2.0, 6.0, 1.0);
        let dist = PiecewiseConstantDistribution::from_single(cell);
        let mean = dist.mean_vector();

        let y1 = &mean["target__y1"];
        assert_eq!(y1.len(), 1);
        assert!((y1[0] - 4.0).abs() < 1e-10); // (2+6)/2
    }

    #[test]
    fn weighted_mean_two_cells() {
        let c1 = make_continuous_cell(0.0, 4.0, 3.0); // mean=2.0, mass=3
        let c2 = make_continuous_cell(4.0, 8.0, 1.0); // mean=6.0, mass=1
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);
        let mean = dist.mean_vector();

        // weighted mean = (3/4)*2 + (1/4)*6 = 1.5 + 1.5 = 3.0
        let y1 = &mean["target__y1"];
        assert!((y1[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn zero_mass_falls_back_to_uniform() {
        let c1 = make_continuous_cell(0.0, 4.0, 0.0); // mean=2.0
        let c2 = make_continuous_cell(4.0, 8.0, 0.0); // mean=6.0
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);
        let mean = dist.mean_vector();

        // uniform: (1/2)*2 + (1/2)*6 = 4.0
        let y1 = &mean["target__y1"];
        assert!((y1[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn total_mass_sums_cells() {
        let c1 = make_continuous_cell(0.0, 4.0, 3.0);
        let c2 = make_continuous_cell(4.0, 8.0, 1.0);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);
        assert!((dist.total_mass() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_divides_mass() {
        let c1 = make_continuous_cell(0.0, 4.0, 1.0);
        let c2 = make_continuous_cell(4.0, 8.0, 1.0);
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        assert_eq!(ens.n_cells(), 2);
        // Each cell's mass is divided by 2
        assert!((ens.cells[0].mass - 0.5).abs() < 1e-10);
        assert!((ens.cells[1].mass - 0.5).abs() < 1e-10);
    }

    #[test]
    fn ensemble_mean_averages_trees() {
        // Tree 1 predicts mean=2.0, Tree 2 predicts mean=6.0
        let c1 = make_continuous_cell(0.0, 4.0, 1.0);
        let c2 = make_continuous_cell(4.0, 8.0, 1.0);
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        // (0.5/1.0)*2 + (0.5/1.0)*6 = 4.0
        assert!((y1[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn pdf_segments_continuous() {
        let c1 = make_continuous_cell(0.0, 4.0, 3.0);
        let c2 = make_continuous_cell(4.0, 8.0, 1.0);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);
        let segs = dist.pdf_segments();

        assert_eq!(segs.len(), 2);
        // total_mass=4, seg0: density = 3/(4*4) = 0.1875, seg1: density = 1/(4*4) = 0.0625
        assert!((segs[0].0 - 0.1875).abs() < 1e-10);
        assert!((segs[0].1 - 0.0).abs() < 1e-10);
        assert!((segs[0].2 - 4.0).abs() < 1e-10);
        assert!((segs[1].0 - 0.0625).abs() < 1e-10);
    }

    #[test]
    fn categorical_mean_vector_is_probability() {
        // Domain: [0, 1, 2] → ["cat_0", "cat_1", "cat_2"]
        // Cell 1: active={0, 1} mass=3  → mean=[1, 1, 0]
        // Cell 2: active={2}    mass=1  → mean=[0, 0, 1]
        let c1 = make_categorical_cell(vec![0, 1], 3, 3.0);
        let c2 = make_categorical_cell(vec![2], 3, 1.0);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);
        let mean = dist.mean_vector();

        let probs = &mean["target__color"];
        assert_eq!(probs.len(), 3);
        // weighted: (3/4)*[1,1,0] + (1/4)*[0,0,1] = [0.75, 0.75, 0.25]
        assert!((probs[0] - 0.75).abs() < 1e-10);
        assert!((probs[1] - 0.75).abs() < 1e-10);
        assert!((probs[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn category_probabilities_filters_categorical() {
        let c1 = make_categorical_cell(vec![0, 1], 3, 1.0);
        let dist = PiecewiseConstantDistribution::from_single(c1);
        let cat_probs = dist.category_probabilities();

        assert!(cat_probs.contains_key("target__color"));
        assert_eq!(cat_probs.len(), 1);
    }

    #[test]
    fn categorical_domain_names_extraction() {
        let c1 = make_categorical_cell(vec![0, 1], 3, 1.0);
        let dist = PiecewiseConstantDistribution::from_single(c1);
        let names = dist.categorical_domain_names("target__color").unwrap();
        assert_eq!(names, vec!["cat_0", "cat_1", "cat_2"]);
    }

    #[test]
    fn empty_distribution() {
        let dist = PiecewiseConstantDistribution::from_cells(Vec::new());
        assert_eq!(dist.n_cells(), 0);
        assert!((dist.total_mass() - 0.0).abs() < 1e-10);
        assert!(dist.mean_vector().is_empty());
    }
}
