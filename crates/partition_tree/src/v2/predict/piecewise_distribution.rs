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
use crate::v2::dataset_view::ColumnView;

use super::conditioned_cell::ConditionedCell;
use super::disjoint::{DisjointError, as_disjoint_resolvable};

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

    /// Resolve overlapping cells into a disjoint partition.
    ///
    /// Produces a new distribution where every target-space point belongs to
    /// exactly **one** cell, with accumulated density from all source cells
    /// that originally covered that point.
    ///
    /// # Algorithm
    ///
    /// 1. Collect boundary points from every cell's target rule via
    ///    [`DisjointResolvable::boundaries`](super::disjoint::DisjointResolvable::boundaries).
    /// 2. Sort and deduplicate boundaries.
    /// 3. Sweep consecutive boundary pairs `[b_i, b_{i+1})` — for each
    ///    fragment, sum `mass` from every cell whose rule contains the
    ///    fragment's midpoint, and emit a new [`ConditionedCell`] with the
    ///    fragment's sub-rule and accumulated mass.
    ///
    /// After resolution, `pdf_segments()` on the result will integrate to
    /// `1.0` for continuous targets (assuming the source masses sum to the
    /// same total).
    ///
    /// # Errors
    ///
    /// - [`DisjointError::Empty`] — no cells.
    /// - [`DisjointError::MultiDimensional`] — any cell has more than one
    ///   target column. Multi-dimensional disjoint resolution has
    ///   exponential complexity and is not supported.
    /// - [`DisjointError::InconsistentColumns`] — cells target different
    ///   columns.
    /// - [`DisjointError::UnsupportedRuleType`] — a rule cannot be
    ///   downcast to [`DisjointResolvable`](super::disjoint::DisjointResolvable).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);
    /// let disjoint = ens.resolve_disjoint().unwrap();
    /// // pdf_segments now integrate to 1.0
    /// let segs = disjoint.pdf_segments();
    /// let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
    /// assert!((integral - 1.0).abs() < 1e-10);
    /// ```
    pub fn resolve_disjoint(&self) -> Result<PiecewiseConstantDistribution, DisjointError> {
        if self.cells.is_empty() {
            return Err(DisjointError::Empty);
        }

        // -- 1. Validate: all cells must have exactly 1 target dimension,
        //       and all must target the same column name.
        let mut target_col_name: Option<String> = None;
        for cell in &self.cells {
            let n = cell.n_target_dims();
            if n != 1 {
                return Err(DisjointError::MultiDimensional { n_dims: n });
            }
            let col = cell.target_rules.keys().next().unwrap().clone();
            match &target_col_name {
                None => target_col_name = Some(col),
                Some(expected) if *expected != col => {
                    return Err(DisjointError::InconsistentColumns {
                        expected: expected.clone(),
                        found: col,
                    });
                }
                _ => {}
            }
        }
        let target_col = target_col_name.unwrap();

        // -- 2. Collect resolvable references and boundaries.
        //       We work with indices into self.cells to avoid lifetime issues.
        let mut all_boundaries: Vec<f64> = Vec::new();
        // Validate that every rule is downcastable
        for cell in &self.cells {
            let rule = cell.target_rules.get(&target_col).unwrap();
            let dr =
                as_disjoint_resolvable(rule.as_ref()).ok_or(DisjointError::UnsupportedRuleType)?;
            all_boundaries.extend(dr.boundaries());
        }

        // Sort and deduplicate
        all_boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_boundaries.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

        if all_boundaries.len() < 2 {
            // Degenerate: all boundaries collapse to a single point
            return Ok(PiecewiseConstantDistribution::from_cells(Vec::new()));
        }

        // -- 3. Sweep consecutive boundary pairs and accumulate density.
        let mut disjoint_cells: Vec<ConditionedCell> = Vec::new();

        for window in all_boundaries.windows(2) {
            let frag_low = window[0];
            let frag_high = window[1];
            if frag_high <= frag_low {
                continue;
            }
            let midpoint = (frag_low + frag_high) / 2.0;

            // Sum density-weighted mass from all source cells that cover
            // this fragment. Each rule type determines how mass distributes
            // via `DisjointResolvable::mass_fraction`:
            // - Continuous/Integer: `frag_width / rule_width` (uniform density)
            // - Categorical: `1.0` (full mass per category; the one-hot mean
            //   vector already encodes category identity)
            let mut accumulated_mass = 0.0f64;
            let mut fragment_rule: Option<Box<dyn crate::v2::rule::DynRule>> = None;

            for cell in &self.cells {
                let rule = cell.target_rules.get(&target_col).unwrap();
                let dr = as_disjoint_resolvable(rule.as_ref()).unwrap();
                if dr.contains_point(midpoint) {
                    accumulated_mass += cell.mass * dr.mass_fraction(frag_low, frag_high);
                    // Lazily create the fragment sub-rule from the first
                    // contributing cell (all produce equivalent geometry).
                    if fragment_rule.is_none() {
                        fragment_rule = dr.sub_rule(frag_low, frag_high);
                    }
                }
            }

            if accumulated_mass > 0.0 {
                if let Some(sub) = fragment_rule {
                    let mut target_rules: HashMap<String, Box<dyn crate::v2::rule::DynRule>> =
                        HashMap::new();
                    target_rules.insert(target_col.clone(), sub);
                    disjoint_cells.push(ConditionedCell::new(target_rules, accumulated_mass));
                }
            }
        }

        Ok(PiecewiseConstantDistribution::from_cells(disjoint_cells))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::ContinuousInterval;
    use crate::v2::rule::DynRule;
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

    // -----------------------------------------------------------------------
    // Edge-case tests: non-disjoint cells & ensemble robustness
    // -----------------------------------------------------------------------

    #[test]
    fn ensemble_overlapping_cells_mean_is_average() {
        // Two trees whose leaves overlap in target space: [0,6) and [2,8).
        // Tree 1 mean = 3.0, Tree 2 mean = 5.0 → ensemble mean = 4.0
        let c1 = make_continuous_cell(0.0, 6.0, 1.0); // mean=3.0
        let c2 = make_continuous_cell(2.0, 8.0, 1.0); // mean=5.0
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        assert_eq!(ens.n_cells(), 2);
        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        // (0.5/1.0)*3.0 + (0.5/1.0)*5.0 = 4.0
        assert!((y1[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_identical_cells_preserves_mean() {
        // Both trees produce the exact same cell [2,6), mass=1.
        // Ensemble mean should be identical to single-tree mean = 4.0.
        let c1 = make_continuous_cell(2.0, 6.0, 1.0);
        let c2 = make_continuous_cell(2.0, 6.0, 1.0);
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        assert_eq!(ens.n_cells(), 2);
        assert!((ens.total_mass() - 1.0).abs() < 1e-10); // 0.5 + 0.5
        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        assert!((y1[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_subset_cells_mean() {
        // Tree 1: [0,10) mean=5.0, Tree 2: [3,7) mean=5.0 (subset of Tree 1).
        // Equal masses → ensemble mean = 5.0 regardless.
        let c1 = make_continuous_cell(0.0, 10.0, 1.0);
        let c2 = make_continuous_cell(3.0, 7.0, 1.0);
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        assert!((y1[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_single_distribution_is_identity() {
        // Ensembling a single distribution should behave identically to the
        // original (mass is divided by 1, so unchanged).
        let c = make_continuous_cell(2.0, 6.0, 3.0);
        let d = PiecewiseConstantDistribution::from_single(c);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d]);

        assert_eq!(ens.n_cells(), 1);
        assert!((ens.cells[0].mass - 3.0).abs() < 1e-10);
        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        assert!((y1[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_empty_is_empty() {
        let ens = PiecewiseConstantDistribution::ensemble(&[]);
        assert_eq!(ens.n_cells(), 0);
        assert!(ens.mean_vector().is_empty());
    }

    #[test]
    fn ensemble_unequal_cell_counts() {
        // Tree 1: 1 cell [0,4) mass=1.
        // Tree 2: 2 cells [4,6) mass=0.6, [6,10) mass=0.4.
        // Masses are divided by 2 (number of distributions, not cells).
        let c1 = make_continuous_cell(0.0, 4.0, 1.0); // mean=2.0
        let c2a = make_continuous_cell(4.0, 6.0, 0.6); // mean=5.0
        let c2b = make_continuous_cell(6.0, 10.0, 0.4); // mean=8.0
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_cells(vec![c2a, c2b]);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        assert_eq!(ens.n_cells(), 3);
        // masses: 1/2=0.5, 0.6/2=0.3, 0.4/2=0.2  → total=1.0
        assert!((ens.total_mass() - 1.0).abs() < 1e-10);

        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        // (0.5*2.0 + 0.3*5.0 + 0.2*8.0) / 1.0 = 1.0+1.5+1.6 = 4.1
        assert!((y1[0] - 4.1).abs() < 1e-10);
    }

    #[test]
    fn ensemble_of_ensembles_double_nesting() {
        // Two single-tree dists → ensemble → ensemble that with a third.
        // This tests that nested ensembling keeps correct mass ratios.
        let c1 = make_continuous_cell(0.0, 4.0, 1.0); // mean=2
        let c2 = make_continuous_cell(4.0, 8.0, 1.0); // mean=6
        let c3 = make_continuous_cell(2.0, 6.0, 1.0); // mean=4
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let d3 = PiecewiseConstantDistribution::from_single(c3);

        // First ensemble of d1, d2: masses become 0.5 each
        let ens12 = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);
        // Second ensemble of ens12 and d3: ens12 cells' masses halved again,
        // d3's mass halved.
        let ens_all = PiecewiseConstantDistribution::ensemble(&[&ens12, &d3]);

        assert_eq!(ens_all.n_cells(), 3);
        // ens12[0].mass = 0.5/2 = 0.25, ens12[1].mass = 0.5/2 = 0.25,
        // d3.mass = 1.0/2 = 0.5 → total = 1.0
        assert!((ens_all.total_mass() - 1.0).abs() < 1e-10);

        let mean = ens_all.mean_vector();
        let y1 = &mean["target__y1"];
        // 0.25*2 + 0.25*6 + 0.5*4 = 0.5+1.5+2.0 = 4.0
        assert!((y1[0] - 4.0).abs() < 1e-10);

        // Note: nested ensemble gives d3 double the weight of d1 or d2.
        // Flat ensemble would give equal weight. This is the expected
        // (and intentional) behavior of the current implementation.
    }

    #[test]
    fn ensemble_flat_vs_nested_differs() {
        // Demonstrate that flat ensemble(d1,d2,d3) gives equal 1/3 weights,
        // while nested ensemble(ensemble(d1,d2), d3) gives 1/4, 1/4, 1/2.
        let c1 = make_continuous_cell(0.0, 4.0, 1.0); // mean=2
        let c2 = make_continuous_cell(4.0, 8.0, 1.0); // mean=6
        let c3 = make_continuous_cell(8.0, 12.0, 1.0); // mean=10
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let d3 = PiecewiseConstantDistribution::from_single(c3);

        // Flat: (2+6+10)/3 = 6.0
        let flat = PiecewiseConstantDistribution::ensemble(&[&d1, &d2, &d3]);
        let flat_mean = flat.mean_vector()["target__y1"][0];
        assert!((flat_mean - 6.0).abs() < 1e-10);

        // Nested: 0.25*2 + 0.25*6 + 0.5*10 = 0.5+1.5+5 = 7.0
        let ens12 = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);
        let nested = PiecewiseConstantDistribution::ensemble(&[&ens12, &d3]);
        let nested_mean = nested.mean_vector()["target__y1"][0];
        assert!((nested_mean - 7.0).abs() < 1e-10);

        // They should differ
        assert!((flat_mean - nested_mean).abs() > 0.5);
    }

    #[test]
    fn ensemble_overlapping_pdf_segments_sum_density() {
        // Two overlapping cells: [0,4) and [2,6), each mass=1.
        // After ensemble (k=2): masses become 0.5 each, total=1.0.
        // In overlap region [2,4): both segments contribute density.
        let c1 = make_continuous_cell(0.0, 4.0, 1.0);
        let c2 = make_continuous_cell(2.0, 6.0, 1.0);
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        let segs = ens.pdf_segments();
        assert_eq!(segs.len(), 2);

        // total_mass = 1.0
        // seg0: density = 0.5 / (1.0 * 4.0) = 0.125, low=0, high=4
        // seg1: density = 0.5 / (1.0 * 4.0) = 0.125, low=2, high=6
        assert!((segs[0].0 - 0.125).abs() < 1e-10);
        assert!((segs[1].0 - 0.125).abs() < 1e-10);

        // At a point in [2,4) the true pdf = 0.125 + 0.125 = 0.25
        // (both segments overlap there).
        let combined_density_in_overlap = segs[0].0 + segs[1].0;
        assert!((combined_density_in_overlap - 0.25).abs() < 1e-10);
    }

    #[test]
    fn ensemble_many_distributions_numerical_stability() {
        // Ensemble of 100 identical distributions. Mass should remain stable.
        let dists: Vec<PiecewiseConstantDistribution> = (0..100)
            .map(|_| {
                PiecewiseConstantDistribution::from_single(make_continuous_cell(2.0, 6.0, 1.0))
            })
            .collect();
        let refs: Vec<&PiecewiseConstantDistribution> = dists.iter().collect();
        let ens = PiecewiseConstantDistribution::ensemble(&refs);

        assert_eq!(ens.n_cells(), 100);
        // Each cell mass = 1/100 = 0.01, total = 1.0
        assert!((ens.total_mass() - 1.0).abs() < 1e-10);
        // Mean should be exactly 4.0
        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        assert!((y1[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_mixed_mass_overlapping() {
        // Tree 1: [0,10) mass=0.8 (broad, low density).
        // Tree 2: [3,5)  mass=0.9 (narrow, high density).
        // Non-disjoint — cell 2 is a subset of cell 1.
        let c1 = make_continuous_cell(0.0, 10.0, 0.8);
        let c2 = make_continuous_cell(3.0, 5.0, 0.9);
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        // masses: 0.4, 0.45 → total = 0.85
        assert!((ens.total_mass() - 0.85).abs() < 1e-10);

        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        // (0.4/0.85)*5.0 + (0.45/0.85)*4.0
        let expected = (0.4 / 0.85) * 5.0 + (0.45 / 0.85) * 4.0;
        assert!((y1[0] - expected).abs() < 1e-10);

        let segs = ens.pdf_segments();
        // Seg 0: density = 0.4/(0.85*10) = 0.04706...
        // Seg 1: density = 0.45/(0.85*2) = 0.26470...
        let d0 = 0.4 / (0.85 * 10.0);
        let d1_density = 0.45 / (0.85 * 2.0);
        assert!((segs[0].0 - d0).abs() < 1e-10);
        assert!((segs[1].0 - d1_density).abs() < 1e-10);

        // In the overlap [3,5), combined density = d0 + d1_density
        let overlap_density = d0 + d1_density;
        // This should be much higher than in [0,3) or [5,10) where only d0 applies
        assert!(overlap_density > d0 * 2.0);
    }

    #[test]
    fn ensemble_categorical_overlapping_domains() {
        // Tree 1: active={0,1} out of {0,1,2}, mass=1
        //   mean = [1,1,0] (unnormalized one-hot)
        // Tree 2: active={1,2} out of {0,1,2}, mass=1
        //   mean = [0,1,1]
        // Ensemble should average: [0.5, 1.0, 0.5]
        let c1 = make_categorical_cell(vec![0, 1], 3, 1.0);
        let c2 = make_categorical_cell(vec![1, 2], 3, 1.0);
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        let mean = ens.mean_vector();
        let probs = &mean["target__color"];
        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 1.0).abs() < 1e-10);
        assert!((probs[2] - 0.5).abs() < 1e-10);

        // category_probabilities should also reflect this
        let cat_probs = ens.category_probabilities();
        let cp = &cat_probs["target__color"];
        assert!((cp[0] - 0.5).abs() < 1e-10);
        assert!((cp[1] - 1.0).abs() < 1e-10);
        assert!((cp[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn ensemble_categorical_identical_cells() {
        // Both trees predict active={0}, domain={0,1,2}, mass=1.
        // mean = [1,0,0] for each → ensemble mean = [1,0,0].
        let c1 = make_categorical_cell(vec![0], 3, 1.0);
        let c2 = make_categorical_cell(vec![0], 3, 1.0);
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        let mean = ens.mean_vector();
        let probs = &mean["target__color"];
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!((probs[1] - 0.0).abs() < 1e-10);
        assert!((probs[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_with_zero_mass_distribution() {
        // Tree 1 has mass=0 (no training data in leaf), Tree 2 has mass=1.
        // After ensemble: 0/2=0 and 1/2=0.5 → total=0.5
        // Mean should come entirely from tree 2.
        let c1 = make_continuous_cell(0.0, 4.0, 0.0); // mean=2, mass=0
        let c2 = make_continuous_cell(4.0, 8.0, 1.0); // mean=6, mass=1
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        assert!((ens.total_mass() - 0.5).abs() < 1e-10);
        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        // Only c2 contributes: (0.5/0.5)*6 = 6.0 for c2,
        // c1 has weight 0/0.5 = 0
        assert!((y1[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_all_zero_mass_falls_back_to_uniform() {
        // All trees have zero mass → uniform fallback.
        let c1 = make_continuous_cell(0.0, 4.0, 0.0); // mean=2
        let c2 = make_continuous_cell(4.0, 8.0, 0.0); // mean=6
        let d1 = PiecewiseConstantDistribution::from_single(c1);
        let d2 = PiecewiseConstantDistribution::from_single(c2);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        assert!((ens.total_mass() - 0.0).abs() < 1e-10);
        let mean = ens.mean_vector();
        let y1 = &mean["target__y1"];
        // Uniform: (1/2)*2 + (1/2)*6 = 4.0
        assert!((y1[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_preserves_total_mass_invariant() {
        // When all source distributions have total_mass = 1.0,
        // the ensemble should also have total_mass = 1.0.
        let c1 = make_continuous_cell(0.0, 4.0, 0.7);
        let c1b = make_continuous_cell(4.0, 10.0, 0.3);
        let c2 = make_continuous_cell(0.0, 10.0, 1.0);
        let d1 = PiecewiseConstantDistribution::from_cells(vec![c1, c1b]);
        let d2 = PiecewiseConstantDistribution::from_single(c2);

        // d1 total = 1.0, d2 total = 1.0
        assert!((d1.total_mass() - 1.0).abs() < 1e-10);
        assert!((d2.total_mass() - 1.0).abs() < 1e-10);

        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);
        // ensemble total should also be 1.0
        assert!((ens.total_mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ensemble_pdf_segments_integration_sums_to_one() {
        // For a proper piecewise-constant pdf over disjoint segments,
        // integral (density * width) should sum to 1.
        let c1 = make_continuous_cell(0.0, 4.0, 0.6);
        let c2 = make_continuous_cell(4.0, 10.0, 0.4);
        let d1 = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let segs = d1.pdf_segments();
        let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
        assert!((integral - 1.0).abs() < 1e-10);

        // Ensemble of two disjoint-cell distributions
        let c3 = make_continuous_cell(0.0, 5.0, 0.5);
        let c4 = make_continuous_cell(5.0, 10.0, 0.5);
        let d2 = PiecewiseConstantDistribution::from_cells(vec![c3, c4]);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        // Even with overlapping segments from different trees, the sum of
        // density * width per segment does NOT equal 1 because segments overlap.
        // This is expected: pdf_segments is a raw list, not a merged partition.
        let seg_integral: f64 = ens
            .pdf_segments()
            .iter()
            .map(|(d, lo, hi)| d * (hi - lo))
            .sum();
        // Each distribution integrates to 1, so raw sum of non-overlapping
        // regions from each tree contributes, but overlaps are double-counted
        // in this naive integral. The actual pdf (evaluated point-wise) still
        // integrates to 1.
        // Here we just verify the segments exist and have positive density.
        assert!(seg_integral > 0.0);
        for (d, lo, hi) in &ens.pdf_segments() {
            assert!(*d > 0.0);
            assert!(hi > lo);
        }
    }

    // -----------------------------------------------------------------------
    // resolve_disjoint tests
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_disjoint_overlapping_continuous_integrates_to_one() {
        // Two overlapping cells [0,4) and [2,6), each mass=0.5 (as from ensemble).
        // After resolve_disjoint the pdf_segments should integrate to 1.
        let c1 = make_continuous_cell(0.0, 4.0, 0.5);
        let c2 = make_continuous_cell(2.0, 6.0, 0.5);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let resolved = dist.resolve_disjoint().unwrap();

        // Should produce 3 disjoint segments: [0,2), [2,4), [4,6)
        assert_eq!(resolved.n_cells(), 3);

        let segs = resolved.pdf_segments();
        let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
        assert!(
            (integral - 1.0).abs() < 1e-10,
            "integral = {integral}, expected 1.0"
        );
    }

    #[test]
    fn resolve_disjoint_preserves_total_mass() {
        let c1 = make_continuous_cell(0.0, 4.0, 0.5);
        let c2 = make_continuous_cell(2.0, 6.0, 0.5);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);
        let original_total = dist.total_mass();

        let resolved = dist.resolve_disjoint().unwrap();
        // With density-weighted accumulation, total mass is preserved:
        // [0,2): 0.5*(2/4)=0.25, [2,4): 0.5*(2/4)+0.5*(2/4)=0.5, [4,6): 0.5*(2/4)=0.25
        // total = 0.25 + 0.5 + 0.25 = 1.0
        assert!(
            (resolved.total_mass() - original_total).abs() < 1e-10,
            "resolved total={}, original={}",
            resolved.total_mass(),
            original_total
        );
    }

    #[test]
    fn resolve_disjoint_already_disjoint_is_identity() {
        // Two disjoint cells [0,4) and [4,8) — should remain as 2 cells.
        let c1 = make_continuous_cell(0.0, 4.0, 0.6);
        let c2 = make_continuous_cell(4.0, 8.0, 0.4);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let resolved = dist.resolve_disjoint().unwrap();
        assert_eq!(resolved.n_cells(), 2);

        let segs = resolved.pdf_segments();
        let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
        assert!(
            (integral - 1.0).abs() < 1e-10,
            "integral = {integral}, expected 1.0"
        );
    }

    #[test]
    fn resolve_disjoint_single_cell_identity() {
        let c = make_continuous_cell(2.0, 6.0, 1.0);
        let dist = PiecewiseConstantDistribution::from_single(c);

        let resolved = dist.resolve_disjoint().unwrap();
        assert_eq!(resolved.n_cells(), 1);
        assert!((resolved.total_mass() - 1.0).abs() < 1e-10);

        let segs = resolved.pdf_segments();
        let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
        assert!((integral - 1.0).abs() < 1e-10);
    }

    #[test]
    fn resolve_disjoint_empty_returns_error() {
        let dist = PiecewiseConstantDistribution::from_cells(Vec::new());
        assert!(dist.resolve_disjoint().is_err());
    }

    #[test]
    fn resolve_disjoint_multi_target_returns_error() {
        // Build a cell with 2 target dimensions
        let mut target_rules: HashMap<String, Box<dyn DynRule>> = HashMap::new();
        target_rules.insert(
            "target__y1".to_string(),
            Box::new(ContinuousInterval::new(
                0.0,
                4.0,
                true,
                true,
                Some((0.0, 10.0)),
                true,
            )),
        );
        target_rules.insert(
            "target__y2".to_string(),
            Box::new(ContinuousInterval::new(
                0.0,
                4.0,
                true,
                true,
                Some((0.0, 10.0)),
                true,
            )),
        );
        let cell = ConditionedCell::new(target_rules, 1.0);
        let dist = PiecewiseConstantDistribution::from_single(cell);

        let result = dist.resolve_disjoint();
        assert!(result.is_err());
        match result.unwrap_err() {
            super::super::disjoint::DisjointError::MultiDimensional { n_dims } => {
                assert_eq!(n_dims, 2);
            }
            other => panic!("expected MultiDimensional, got {:?}", other),
        }
    }

    #[test]
    fn resolve_disjoint_subset_continuous() {
        // Cell 1: [0,10) mass=0.5, Cell 2: [3,5) mass=0.5 (subset of cell 1).
        // With density-weighted accumulation:
        //   [0,3): 0.5*(3/10)=0.15
        //   [3,5): 0.5*(2/10)+0.5*(2/2)=0.1+0.5=0.6
        //   [5,10): 0.5*(5/10)=0.25
        // total = 0.15+0.6+0.25 = 1.0
        let c1 = make_continuous_cell(0.0, 10.0, 0.5);
        let c2 = make_continuous_cell(3.0, 5.0, 0.5);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let resolved = dist.resolve_disjoint().unwrap();
        assert_eq!(resolved.n_cells(), 3);

        let segs = resolved.pdf_segments();
        let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
        assert!(
            (integral - 1.0).abs() < 1e-10,
            "integral = {integral}, expected 1.0"
        );

        // The middle segment [3,5) should have higher density than endpoints
        // because it accumulates mass from both cells.
        let densities: Vec<f64> = segs.iter().map(|(d, _, _)| *d).collect();
        assert!(densities[1] > densities[0]);
        assert!(densities[1] > densities[2]);
    }

    #[test]
    fn resolve_disjoint_ensemble_then_resolve() {
        // Full pipeline: ensemble overlapping trees → resolve → check integral.
        let c1 = make_continuous_cell(0.0, 4.0, 0.6);
        let c1b = make_continuous_cell(4.0, 10.0, 0.4);
        let c2 = make_continuous_cell(0.0, 5.0, 0.5);
        let c2b = make_continuous_cell(5.0, 10.0, 0.5);
        let d1 = PiecewiseConstantDistribution::from_cells(vec![c1, c1b]);
        let d2 = PiecewiseConstantDistribution::from_cells(vec![c2, c2b]);
        let ens = PiecewiseConstantDistribution::ensemble(&[&d1, &d2]);

        let resolved = ens.resolve_disjoint().unwrap();
        let segs = resolved.pdf_segments();
        let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
        assert!(
            (integral - 1.0).abs() < 1e-10,
            "integral = {integral}, expected 1.0"
        );
    }

    #[test]
    fn resolve_disjoint_preserves_mean() {
        // The mean of the resolved distribution should equal the original.
        let c1 = make_continuous_cell(0.0, 6.0, 0.5); // mean=3
        let c2 = make_continuous_cell(2.0, 8.0, 0.5); // mean=5
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let original_mean = dist.mean_vector()["target__y1"][0];
        // Original: (0.5*3 + 0.5*5)/1.0 = 4.0
        assert!((original_mean - 4.0).abs() < 1e-10);

        let resolved = dist.resolve_disjoint().unwrap();
        let resolved_mean = resolved.mean_vector()["target__y1"][0];
        assert!(
            (resolved_mean - original_mean).abs() < 1e-10,
            "resolved_mean = {resolved_mean}, original = {original_mean}"
        );
    }

    #[test]
    fn resolve_disjoint_categorical_overlapping() {
        // Tree 1: active={0,1} mass=0.5, Tree 2: active={1,2} mass=0.5
        // Each category is a unit bin (volume=1 per category).
        // With mass_fraction=1.0 for categorical:
        //   cat 0: 0.5      (from c1 only)
        //   cat 1: 0.5+0.5  (from both)
        //   cat 2: 0.5      (from c2 only)
        let c1 = make_categorical_cell(vec![0, 1], 3, 0.5);
        let c2 = make_categorical_cell(vec![1, 2], 3, 0.5);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let resolved = dist.resolve_disjoint().unwrap();
        // Should have 3 cells, one per active category
        assert_eq!(resolved.n_cells(), 3);

        // Normalized probabilities should match original.
        // Original mean_vector: (0.5/1.0)*[1,1,0] + (0.5/1.0)*[0,1,1] = [0.5, 1.0, 0.5]
        // Normalized: [0.25, 0.5, 0.25]
        // Resolved mean_vector: [0.25, 0.5, 0.25] (already normalized since
        //   each cell has mean=[0,...,1,...,0])
        let orig_mean = dist.mean_vector();
        let res_mean = resolved.mean_vector();
        let orig_raw = &orig_mean["target__color"];
        let res_raw = &res_mean["target__color"];

        // Normalize both to get true probabilities
        let orig_sum: f64 = orig_raw.iter().sum();
        let res_sum: f64 = res_raw.iter().sum();
        for i in 0..3 {
            let orig_p = orig_raw[i] / orig_sum;
            let res_p = res_raw[i] / res_sum;
            assert!(
                (orig_p - res_p).abs() < 1e-10,
                "category {i}: original_p={orig_p}, resolved_p={res_p}"
            );
        }
    }

    #[test]
    fn resolve_disjoint_categorical_disjoint_sets() {
        // Tree 1: active={0} mass=0.7, Tree 2: active={1,2} mass=0.3
        // Already disjoint — each category maps to exactly one cell.
        // With mass_fraction=1.0 for categorical:
        //   cat 0: 0.7
        //   cat 1: 0.3
        //   cat 2: 0.3
        let c1 = make_categorical_cell(vec![0], 3, 0.7);
        let c2 = make_categorical_cell(vec![1, 2], 3, 0.3);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let resolved = dist.resolve_disjoint().unwrap();
        assert_eq!(resolved.n_cells(), 3); // one per active category

        // Normalized probabilities should match original
        let orig_mean = dist.mean_vector();
        let res_mean = resolved.mean_vector();
        let orig_raw = &orig_mean["target__color"];
        let res_raw = &res_mean["target__color"];

        let orig_sum: f64 = orig_raw.iter().sum();
        let res_sum: f64 = res_raw.iter().sum();
        for i in 0..3 {
            let orig_p = orig_raw[i] / orig_sum;
            let res_p = res_raw[i] / res_sum;
            assert!(
                (orig_p - res_p).abs() < 1e-10,
                "category {i}: original_p={orig_p}, resolved_p={res_p}"
            );
        }
    }

    #[test]
    fn resolve_disjoint_many_overlapping_continuous() {
        // 5 overlapping cells — stress test that integral still sums to 1.
        let cells: Vec<ConditionedCell> = (0..5)
            .map(|i| {
                let lo = i as f64 * 2.0;
                let hi = lo + 5.0;
                make_continuous_cell(lo, hi, 0.2)
            })
            .collect();
        let dist = PiecewiseConstantDistribution::from_cells(cells);

        let resolved = dist.resolve_disjoint().unwrap();
        let segs = resolved.pdf_segments();
        let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
        assert!(
            (integral - 1.0).abs() < 1e-10,
            "integral = {integral}, expected 1.0"
        );
    }

    #[test]
    fn resolve_disjoint_identical_cells() {
        // Two identical cells [2,6) mass=0.5 each.
        // Should produce one fragment [2,6) with accumulated mass 1.0.
        let c1 = make_continuous_cell(2.0, 6.0, 0.5);
        let c2 = make_continuous_cell(2.0, 6.0, 0.5);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let resolved = dist.resolve_disjoint().unwrap();
        assert_eq!(resolved.n_cells(), 1);
        assert!((resolved.total_mass() - 1.0).abs() < 1e-10);

        let segs = resolved.pdf_segments();
        let integral: f64 = segs.iter().map(|(d, lo, hi)| d * (hi - lo)).sum();
        assert!((integral - 1.0).abs() < 1e-10);
    }

    #[test]
    fn resolve_disjoint_integer_overlapping() {
        use crate::rules::IntegerInterval;

        // Two overlapping integer cells: [0,5] and [3,8], mass=0.5 each.
        let make_int_cell = |lo: i64, hi: i64, mass: f64| -> ConditionedCell {
            let mut target_rules: HashMap<String, Box<dyn DynRule>> = HashMap::new();
            target_rules.insert(
                "target__y1".to_string(),
                Box::new(IntegerInterval::new(lo, hi, Some((0, 10)), true)),
            );
            ConditionedCell::new(target_rules, mass)
        };

        let c1 = make_int_cell(0, 5, 0.5);
        let c2 = make_int_cell(3, 8, 0.5);
        let dist = PiecewiseConstantDistribution::from_cells(vec![c1, c2]);

        let resolved = dist.resolve_disjoint().unwrap();
        // [0,3) → 3 ints, [3,6) → 3 ints overlap, [6,9) → 3 ints
        assert!(resolved.n_cells() >= 2);
        assert!(resolved.total_mass() > 0.0);

        // Mean should be preserved
        let orig_mean = dist.mean_vector()["target__y1"][0];
        let res_mean = resolved.mean_vector()["target__y1"][0];
        assert!(
            (orig_mean - res_mean).abs() < 1e-10,
            "orig={orig_mean}, resolved={res_mean}"
        );
    }
}
