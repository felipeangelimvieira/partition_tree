//! Split result types for the v2 partition tree.
//!
//! This module defines the types that describe the outcome of a split search:
//!
//! - [`SplitKind`] — whether the split refines the feature or target space.
//! - [`SplitOp`] — dtype-erased split operation trait (replaces the old enum).
//! - [`SplitPoint`] — full description of a chosen split (column, gain, child stats).
//! - [`SplitRestrictions`] — constraints a candidate split must satisfy.
//! - [`CandidateSplit`] — priority-queue entry pairing a node index with its best split.
//!
//! ## `SplitOp` — dtype-erased split operation
//!
//! [`SplitOp`] encapsulates all dtype-specific logic for a single split:
//!
//! | Method          | Purpose                                                |
//! |-----------------|--------------------------------------------------------|
//! | [`go_left`]     | Decide whether a row goes to the left child            |
//! | [`split_rule`]  | Produce `(left, right)` child rules from a parent rule |
//! | [`child_volumes`] | Compute child target volumes without allocating cells |
//!
//! Concrete implementations: [`ContinuousSplitOp`], [`CategoricalSplitOp`].
//!
//! [`go_left`]: SplitOp::go_left
//! [`split_rule`]: SplitOp::split_rule
//! [`child_volumes`]: SplitOp::child_volumes
use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::dataset_view::ColumnView;
use crate::loss::CellStats;
use crate::rule::DynRule;

/// Whether a split refines the feature space (X) or the target space (Y).
///
/// This classification drives two critical behaviours:
///
/// 1. **Index propagation** — [`Node::propagate_children`](super::node::Node::propagate_children)
///    only partitions `sorted_x` for `XSplit` and `sorted_y` for `YSplit`;
///    the other family is inherited unchanged.
/// 2. **Volume computation** — `XSplit` leaves target volume unchanged;
///    `YSplit` updates it from the split rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitKind {
    /// Feature split: refines A_X, A_Y unchanged.
    XSplit,
    /// Target split: refines A_Y, A_X unchanged.
    YSplit,
}

impl fmt::Display for SplitKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitKind::XSplit => write!(f, "XSplit"),
            SplitKind::YSplit => write!(f, "YSplit"),
        }
    }
}

// ---------------------------------------------------------------------------
// SplitOp trait — dtype-erased split operation
// ---------------------------------------------------------------------------

/// Dtype-erased split operation.
///
/// Encapsulates all logic that depends on the split's dtype: routing rows,
/// splitting rules, and computing child volumes.
///
/// Adding a new dtype only requires implementing this trait — no match arms
/// to update anywhere else.
///
/// # Thread Safety
///
/// Must be `Send + Sync` for parallel split search.
pub trait SplitOp: Send + Sync + fmt::Debug + fmt::Display {
    /// Does sample at `row_idx` go to the left child?
    ///
    /// `col` is the column being split; `none_to_left` controls null routing.
    fn go_left(&self, col: &dyn ColumnView, row_idx: usize, none_to_left: bool) -> bool;

    /// Split a parent rule into `(left_rule, right_rule)`.
    fn split_rule(
        &self,
        parent_rule: &dyn DynRule,
        none_to_left: bool,
    ) -> (Box<dyn DynRule>, Box<dyn DynRule>);

    /// Compute child volumes from a parent rule without allocating child rules.
    ///
    /// Returns `(left_volume, right_volume)`.
    fn child_volumes(&self, parent_rule: &dyn DynRule, none_to_left: bool) -> (f64, f64) {
        let (left, right) = self.split_rule(parent_rule, none_to_left);
        (left.volume(), right.volume())
    }

    /// Clone into a new `Box`.
    fn clone_box(&self) -> Box<dyn SplitOp>;
}

impl Clone for Box<dyn SplitOp> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// ContinuousSplitOp
// ---------------------------------------------------------------------------

/// Split operation for continuous columns: `value < threshold` → left.
#[derive(Debug, Clone)]
pub struct ContinuousSplitOp {
    /// Split threshold.
    pub threshold: f64,
    /// Position in presorted candidate list (debug / optional).
    pub k_candidate: usize,
    /// Position in presorted XY list (debug / optional).
    pub p_xy: usize,
}

impl SplitOp for ContinuousSplitOp {
    fn go_left(&self, col: &dyn ColumnView, row_idx: usize, none_to_left: bool) -> bool {
        match col.get_f64(row_idx) {
            Some(v) => v < self.threshold,
            None => none_to_left,
        }
    }

    fn split_rule(
        &self,
        parent_rule: &dyn DynRule,
        none_to_left: bool,
    ) -> (Box<dyn DynRule>, Box<dyn DynRule>) {
        let ci = parent_rule
            .as_any()
            .downcast_ref::<crate::rules::ContinuousInterval>()
            .expect("ContinuousSplitOp requires a ContinuousInterval rule");

        let (left, right) = <crate::rules::ContinuousInterval as crate::rules::Rule<f64>>::split(
            ci,
            self.threshold,
            Some(none_to_left),
        );
        (Box::new(left), Box::new(right))
    }

    fn clone_box(&self) -> Box<dyn SplitOp> {
        Box::new(self.clone())
    }
}

impl fmt::Display for ContinuousSplitOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Continuous(threshold={:.6})", self.threshold)
    }
}

// ---------------------------------------------------------------------------
// CategoricalSplitOp
// ---------------------------------------------------------------------------

/// Split operation for categorical columns: `code ∈ subset` → left.
#[derive(Debug, Clone)]
pub struct CategoricalSplitOp {
    /// Category codes routed to the left child.
    pub subset_left: HashSet<usize>,
}

impl SplitOp for CategoricalSplitOp {
    fn go_left(&self, col: &dyn ColumnView, row_idx: usize, none_to_left: bool) -> bool {
        match col.get_cat(row_idx) {
            Some(v) => self.subset_left.contains(&v),
            None => none_to_left,
        }
    }

    fn split_rule(
        &self,
        parent_rule: &dyn DynRule,
        none_to_left: bool,
    ) -> (Box<dyn DynRule>, Box<dyn DynRule>) {
        let bt = parent_rule
            .as_any()
            .downcast_ref::<crate::rules::BelongsTo>()
            .expect("CategoricalSplitOp requires a BelongsTo rule");

        let (left, right) = bt.split_subset(self.subset_left.clone(), Some(none_to_left));
        (Box::new(left), Box::new(right))
    }

    fn clone_box(&self) -> Box<dyn SplitOp> {
        Box::new(self.clone())
    }
}

impl fmt::Display for CategoricalSplitOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Categorical(|subset|={})", self.subset_left.len())
    }
}

// ---------------------------------------------------------------------------
// IntegerSplitOp
// ---------------------------------------------------------------------------

/// Split operation for integer columns: `value < threshold` → left.
///
/// The threshold is an integer value. Rows with `value < threshold` go to
/// the left child; rows with `value >= threshold` go to the right.
#[derive(Debug, Clone)]
pub struct IntegerSplitOp {
    /// Split threshold (integer boundary).
    pub threshold: i64,
    /// Position in presorted candidate list (debug / optional).
    pub k_candidate: usize,
    /// Position in presorted XY list (debug / optional).
    pub p_xy: usize,
}

impl SplitOp for IntegerSplitOp {
    fn go_left(&self, col: &dyn ColumnView, row_idx: usize, none_to_left: bool) -> bool {
        match col.get_i64(row_idx) {
            Some(v) => v < self.threshold,
            None => none_to_left,
        }
    }

    fn split_rule(
        &self,
        parent_rule: &dyn DynRule,
        none_to_left: bool,
    ) -> (Box<dyn DynRule>, Box<dyn DynRule>) {
        let ii = parent_rule
            .as_any()
            .downcast_ref::<crate::rules::IntegerInterval>()
            .expect("IntegerSplitOp requires an IntegerInterval rule");

        let (left, right) = <crate::rules::IntegerInterval as crate::rules::Rule<i64>>::split(
            ii,
            self.threshold,
            Some(none_to_left),
        );
        (Box::new(left), Box::new(right))
    }

    fn clone_box(&self) -> Box<dyn SplitOp> {
        Box::new(self.clone())
    }
}

impl fmt::Display for IntegerSplitOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Integer(threshold={})", self.threshold)
    }
}

/// Full description of a successful split search on a single column.
///
/// Produced by a [`ColumnSplitSearcher`](super::column_split::ColumnSplitSearcher)
/// and consumed by the [`TreeBuilder`](super::tree_builder::TreeBuilder) to
/// create child nodes and record split history.
#[derive(Debug, Clone)]
pub struct SplitPoint {
    /// Column that was split.
    pub col_name: String,
    /// Whether this is a feature (X) or target (Y) split.
    pub split_kind: SplitKind,
    /// Whether null values are routed to the left child.
    pub none_to_left: bool,
    /// Information gain of this split.
    pub gain: f64,
    /// Statistics of the left child cell.
    pub left_stats: CellStats,
    /// Statistics of the right child cell.
    pub right_stats: CellStats,
    /// Dtype-specific split operation.
    pub op: Box<dyn SplitOp>,
}

impl fmt::Display for SplitPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SplitPoint({}, {}, gain={:.6}, none_left={}, {})",
            self.col_name, self.split_kind, self.gain, self.none_to_left, self.op
        )
    }
}

// ---------------------------------------------------------------------------
// Split Restrictions
// ---------------------------------------------------------------------------

/// Constraints that a candidate split must satisfy to be accepted.
///
/// These are checked both before searching (via [`can_split`](SplitRestrictions::can_split))
/// and after evaluating each candidate (via
/// [`is_valid_children`](SplitRestrictions::is_valid_children)).
///
/// # Defaults
///
/// The `Default` implementation uses permissive values:
///
/// | Field               | Default         |
/// |---------------------|-----------------|
/// | `min_samples_xy`    | 1.0             |
/// | `min_samples_x`     | 1.0             |
/// | `min_samples_y`     | 1.0             |
/// | `min_gain`          | 0.0             |
/// | `min_volume`        | 0.0             |
/// | `max_depth`         | `usize::MAX`    |
/// | `min_samples_split` | 2.0             |
#[derive(Debug, Clone)]
pub struct SplitRestrictions {
    /// Minimum w_xy in each child.
    pub min_samples_xy: f64,
    /// Minimum w_x in each child.
    pub min_samples_x: f64,
    /// Minimum w_y in each child.
    pub min_samples_y: f64,
    /// Minimum information gain.
    pub min_gain: f64,
    /// Minimum target volume in each child.
    pub min_volume: f64,
    /// Maximum tree depth (inclusive).
    pub max_depth: usize,
    /// Minimum total samples (w_xy) in the parent to attempt a split.
    pub min_samples_split: f64,
}

impl Default for SplitRestrictions {
    fn default() -> Self {
        Self {
            min_samples_xy: 1.0,
            min_samples_x: 1.0,
            min_samples_y: 1.0,
            min_gain: 0.0,
            min_volume: 0.0,
            max_depth: usize::MAX,
            min_samples_split: 2.0,
        }
    }
}

impl SplitRestrictions {
    /// Check whether a node is eligible for splitting (before searching).
    pub fn can_split(&self, w_xy: f64, depth: usize) -> bool {
        w_xy >= self.min_samples_split && depth < self.max_depth
    }

    /// Validate that both children satisfy all restrictions.
    pub fn is_valid_children(&self, left: &CellStats, right: &CellStats, depth: usize) -> bool {
        if depth >= self.max_depth {
            return false;
        }

        // Sample count constraints
        if left.w_xy < self.min_samples_xy || right.w_xy < self.min_samples_xy {
            return false;
        }
        if left.w_x < self.min_samples_x || right.w_x < self.min_samples_x {
            return false;
        }
        if left.w_y < self.min_samples_y || right.w_y < self.min_samples_y {
            return false;
        }

        // Volume constraints
        if left.volume < self.min_volume || right.volume < self.min_volume {
            return false;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Candidate Split (for the priority queue in TreeBuilder)
// ---------------------------------------------------------------------------

/// A candidate split stored in the best-first priority queue.
///
/// Wraps a [`SplitPoint`] together with the index of the node to split.
/// Ordered by gain (highest gain = highest priority) for use in a
/// [`BinaryHeap`](std::collections::BinaryHeap).
#[derive(Debug, Clone)]
pub struct CandidateSplit {
    /// Index into the node arena.
    pub node_index: usize,
    /// The best split found for this node.
    pub split: SplitPoint,
}

impl CandidateSplit {
    pub fn gain(&self) -> f64 {
        self.split.gain
    }
}

impl PartialEq for CandidateSplit {
    fn eq(&self, other: &Self) -> bool {
        self.gain() == other.gain()
    }
}

impl Eq for CandidateSplit {}

impl PartialOrd for CandidateSplit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CandidateSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gain()
            .partial_cmp(&other.gain())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_restrictions_allow_basic_split() {
        let r = SplitRestrictions::default();
        assert!(r.can_split(10.0, 0));
        assert!(!r.can_split(1.0, 0)); // below min_samples_split
    }

    #[test]
    fn restrictions_respect_max_depth() {
        let r = SplitRestrictions {
            max_depth: 3,
            ..Default::default()
        };
        assert!(r.can_split(10.0, 2));
        assert!(!r.can_split(10.0, 3));
    }

    #[test]
    fn is_valid_children_checks_all_constraints() {
        let r = SplitRestrictions {
            min_samples_xy: 5.0,
            min_samples_x: 3.0,
            min_samples_y: 3.0,
            min_volume: 0.1,
            max_depth: 10,
            ..Default::default()
        };

        let left = CellStats::new(10.0, 20.0, 15.0, 1.0);
        let right = CellStats::new(10.0, 20.0, 15.0, 1.0);
        assert!(r.is_valid_children(&left, &right, 0));

        // Fail on min_samples_xy
        let bad = CellStats::new(2.0, 20.0, 15.0, 1.0);
        assert!(!r.is_valid_children(&left, &bad, 0));

        // Fail on min_volume
        let low_vol = CellStats::new(10.0, 20.0, 15.0, 0.01);
        assert!(!r.is_valid_children(&left, &low_vol, 0));
    }

    #[test]
    fn candidate_split_ordering() {
        let make = |gain: f64| CandidateSplit {
            node_index: 0,
            split: SplitPoint {
                col_name: "x".into(),
                split_kind: SplitKind::XSplit,
                none_to_left: true,
                gain,
                left_stats: CellStats::new(0.0, 0.0, 0.0, 0.0),
                right_stats: CellStats::new(0.0, 0.0, 0.0, 0.0),
                op: Box::new(ContinuousSplitOp {
                    threshold: 0.0,
                    k_candidate: 0,
                    p_xy: 0,
                }),
            },
        };

        let a = make(1.0);
        let b = make(2.0);
        assert!(b > a);
    }
}
