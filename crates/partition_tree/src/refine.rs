//! Tree refinement: extend every fitted split coordinate across the whole space.
//!
//! After [`TreeBuilder`](super::tree_builder::TreeBuilder) produces a fitted
//! [`Tree`], some split thresholds only divide a sub-region (the branch that
//! created them). [`Tree::refined`] returns a new tree where every split
//! coordinate fired in any branch is propagated across the entire feature/
//! target space, so no leaf is crossed by a known threshold.
//!
//! Refined leaves preserve the source leaf's density and target volume
//! through [`FittedNode::inherited_target_volume`](crate::tree::FittedNode::inherited_target_volume):
//! `w_xy`, `w_x`, `w_y` are inherited as-is, and `effective_target_volume`
//! returns the source leaf's geometric target volume.
//!
//! ## Algorithm
//!
//! 1. Recover the unique split coordinate of every [`SplitRecord`] in
//!    [`Tree::split_history`].
//! 2. Walk the tree top-down, carrying a candidate set into each subtree
//!    and pruning candidates that fall outside the subtree's region (this
//!    is the "use the tree structure for efficiency" part).
//! 3. At each leaf, recursively apply every remaining candidate via
//!    [`Cell::apply_split`](crate::cell::Cell::apply_split), creating a
//!    synthetic sub-tree of internal nodes plus refined leaves.
//!
//! Synthetic refinement splits are appended to the new tree's
//! [`split_history`](Tree::split_history) with `gain = 0.0` so
//! [`Tree::feature_importances`] (which filters on `gain > 0`) is
//! unaffected.

use std::collections::{HashMap, HashSet};

use crate::cell::Cell;
use crate::conf::TARGET_PREFIX;
use crate::rules::{BelongsTo, ContinuousInterval, IntegerInterval, QuantizedContinuousInterval};
use crate::split_result::{
    CategoricalSplitOp, ContinuousSplitOp, IntegerSplitOp, QuantizedContinuousSplitOp, SplitKind,
    SplitOp,
};
use crate::tree::{FittedNode, SplitRecord, Tree};

// ---------------------------------------------------------------------------
// SplitCoord — a recovered (column, threshold-or-subset) pair
// ---------------------------------------------------------------------------

/// A dtype-erased recovered split coordinate.
///
/// Captures just enough information to re-apply the same split on another
/// cell via [`SplitCoord::to_split_op`] and to detect duplicates via
/// [`SplitCoord::matches`].
#[derive(Debug, Clone)]
enum SplitCoord {
    Continuous {
        threshold: f64,
    },
    Integer {
        threshold: i64,
    },
    Quantized {
        threshold_idx: i64,
        resolution: f64,
    },
    Categorical {
        subset_left: HashSet<usize>,
    },
}

impl SplitCoord {
    /// Bit-for-bit equality (handles NaN consistently for our purposes —
    /// thresholds come from cell rules, which are always finite).
    fn matches(&self, other: &SplitCoord) -> bool {
        match (self, other) {
            (Self::Continuous { threshold: a }, Self::Continuous { threshold: b }) => {
                a.to_bits() == b.to_bits()
            }
            (Self::Integer { threshold: a }, Self::Integer { threshold: b }) => a == b,
            (
                Self::Quantized {
                    threshold_idx: a,
                    resolution: ra,
                },
                Self::Quantized {
                    threshold_idx: b,
                    resolution: rb,
                },
            ) => a == b && ra.to_bits() == rb.to_bits(),
            (
                Self::Categorical { subset_left: a },
                Self::Categorical { subset_left: b },
            ) => a == b,
            _ => false,
        }
    }

    /// Build a fresh `SplitOp` for this coordinate.
    ///
    /// For categorical splits, `parent_active` is the leaf's currently active
    /// category set so that the produced `subset_left` is restricted to
    /// categories the leaf actually contains.
    fn to_split_op(&self, parent_active: Option<&HashSet<usize>>) -> Box<dyn SplitOp> {
        match self {
            Self::Continuous { threshold } => Box::new(ContinuousSplitOp {
                threshold: *threshold,
                k_candidate: 0,
                p_xy: 0,
            }),
            Self::Integer { threshold } => Box::new(IntegerSplitOp {
                threshold: *threshold,
                k_candidate: 0,
                p_xy: 0,
            }),
            Self::Quantized {
                threshold_idx,
                resolution,
            } => Box::new(QuantizedContinuousSplitOp {
                threshold_idx: *threshold_idx,
                resolution: *resolution,
                k_candidate: 0,
                p_xy: 0,
            }),
            Self::Categorical { subset_left } => {
                let subset = match parent_active {
                    Some(active) => subset_left.intersection(active).copied().collect(),
                    None => subset_left.clone(),
                };
                Box::new(CategoricalSplitOp { subset_left: subset })
            }
        }
    }
}

fn split_kind_for_col(col: &str) -> SplitKind {
    if col.starts_with(TARGET_PREFIX) {
        SplitKind::YSplit
    } else {
        SplitKind::XSplit
    }
}

/// Recover a [`SplitCoord`] from a split applied to `parent_cell`, producing
/// `left_cell` on column `col`.
///
/// Returns `None` if the column or its rule type is unsupported (e.g., the
/// column is missing from one of the cells).
fn recover_split_coord(parent_cell: &Cell, left_cell: &Cell, col: &str) -> Option<SplitCoord> {
    let parent_rule = parent_cell.get_rule(col)?;
    let left_rule = left_cell.get_rule(col)?;

    if parent_rule
        .as_any()
        .downcast_ref::<ContinuousInterval>()
        .is_some()
    {
        let li = left_rule.as_any().downcast_ref::<ContinuousInterval>()?;
        // `ContinuousInterval::split(t)` produces left.high == t.
        return Some(SplitCoord::Continuous { threshold: li.high });
    }
    if parent_rule
        .as_any()
        .downcast_ref::<IntegerInterval>()
        .is_some()
    {
        let li = left_rule.as_any().downcast_ref::<IntegerInterval>()?;
        // `IntegerInterval::split(t)` produces left.high == t - 1.
        return Some(SplitCoord::Integer {
            threshold: li.high.saturating_add(1),
        });
    }
    if let Some(pi) = parent_rule
        .as_any()
        .downcast_ref::<QuantizedContinuousInterval>()
    {
        let li = left_rule
            .as_any()
            .downcast_ref::<QuantizedContinuousInterval>()?;
        // `QuantizedContinuousInterval::split_at_index(t)` produces left.high_idx == t.
        return Some(SplitCoord::Quantized {
            threshold_idx: li.high_idx,
            resolution: pi.resolution,
        });
    }
    if parent_rule.as_any().downcast_ref::<BelongsTo>().is_some() {
        let bl = left_rule.as_any().downcast_ref::<BelongsTo>()?;
        return Some(SplitCoord::Categorical {
            subset_left: bl.values.clone(),
        });
    }
    None
}

/// Whether a candidate split coordinate strictly divides `cell`'s rule on
/// `col`. Coordinates that fall on a boundary or outside the rule's range
/// are considered non-interior and produce no refinement.
fn is_interior(cell: &Cell, col: &str, coord: &SplitCoord) -> bool {
    let rule = match cell.get_rule(col) {
        Some(r) => r,
        None => return false,
    };
    match coord {
        SplitCoord::Continuous { threshold } => {
            let ci = match rule.as_any().downcast_ref::<ContinuousInterval>() {
                Some(c) => c,
                None => return false,
            };
            ci.low < *threshold && *threshold < ci.high
        }
        SplitCoord::Integer { threshold } => {
            let ii = match rule.as_any().downcast_ref::<IntegerInterval>() {
                Some(i) => i,
                None => return false,
            };
            // `IntegerSplitOp` with threshold T: left = [lo, T-1], right = [T, hi].
            // Non-trivial: T > lo and T <= hi.
            ii.low < *threshold && *threshold <= ii.high
        }
        SplitCoord::Quantized {
            threshold_idx,
            resolution,
        } => {
            let qi = match rule.as_any().downcast_ref::<QuantizedContinuousInterval>() {
                Some(q) => q,
                None => return false,
            };
            if qi.resolution.to_bits() != resolution.to_bits() {
                return false;
            }
            // Strict interior on lattice indices.
            *threshold_idx > qi.low_idx && *threshold_idx <= qi.high_idx
        }
        SplitCoord::Categorical { subset_left } => {
            let bt = match rule.as_any().downcast_ref::<BelongsTo>() {
                Some(b) => b,
                None => return false,
            };
            let inter_size = subset_left
                .iter()
                .filter(|v| bt.values.contains(*v))
                .count();
            inter_size > 0 && inter_size < bt.values.len()
        }
    }
}

/// Collect deduplicated split coordinates from the tree's split history,
/// grouped by column.
fn collect_unique_split_coords(tree: &Tree) -> HashMap<String, Vec<SplitCoord>> {
    let mut out: HashMap<String, Vec<SplitCoord>> = HashMap::new();
    for rec in &tree.split_history {
        let parent = &tree.nodes[rec.parent_index];
        let left = &tree.nodes[rec.left_child_index];
        let Some(coord) = recover_split_coord(&parent.cell, &left.cell, &rec.col_name) else {
            continue;
        };
        let entry = out.entry(rec.col_name.clone()).or_default();
        if !entry.iter().any(|c| c.matches(&coord)) {
            entry.push(coord);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Refiner — incremental builder for the refined arena
// ---------------------------------------------------------------------------

struct Refiner<'a> {
    source: &'a Tree,
    arena: Vec<FittedNode>,
    /// Map from original arena index → new arena index.
    orig_to_new: HashMap<usize, usize>,
    /// Synthetic refinement splits, appended to `split_history` at the end.
    synthetic_records: Vec<SplitRecord>,
}

impl<'a> Refiner<'a> {
    fn new(source: &'a Tree) -> Self {
        Self {
            source,
            arena: Vec::with_capacity(source.nodes.len() * 2),
            orig_to_new: HashMap::with_capacity(source.nodes.len()),
            synthetic_records: Vec::new(),
        }
    }

    /// Recursively walk the original tree, copying internal nodes and
    /// refining each leaf with the candidates that survive propagation.
    ///
    /// Returns the index in the new arena where this node was placed.
    fn walk(
        &mut self,
        orig_idx: usize,
        candidates: Vec<(String, SplitCoord)>,
        parent_in_new: Option<usize>,
    ) -> usize {
        let original = &self.source.nodes[orig_idx];
        let original_is_leaf = original.is_leaf;
        let original_cell = original.cell.clone();
        let original_w_xy = original.w_xy;
        let original_w_x = original.w_x;
        let original_w_y = original.w_y;
        let original_depth = original.depth;
        let original_split_col = original.split_col.clone();
        let original_split_kind = original.split_kind;
        let original_left = original.left_child;
        let original_right = original.right_child;

        let new_idx = self.arena.len();
        self.arena.push(FittedNode {
            cell: original_cell.clone(),
            w_xy: original_w_xy,
            w_x: original_w_x,
            w_y: original_w_y,
            depth: original_depth,
            parent: parent_in_new,
            left_child: None,
            right_child: None,
            is_leaf: original_is_leaf,
            split_col: original_split_col.clone(),
            split_kind: original_split_kind,
            inherited_target_volume: None,
        });
        self.orig_to_new.insert(orig_idx, new_idx);

        if original_is_leaf {
            // Filter candidates to those strictly interior to this leaf.
            let applicable: Vec<(String, SplitCoord)> = candidates
                .into_iter()
                .filter(|(col, c)| is_interior(&original_cell, col, c))
                .collect();

            if !applicable.is_empty() {
                let inherited_vol = original_cell.target_volume();
                self.refine_leaf_subtree(
                    new_idx,
                    original_w_xy,
                    original_w_x,
                    original_w_y,
                    inherited_vol,
                    applicable,
                );
            }
            return new_idx;
        }

        // Internal node: recover its own split coordinate to prune candidates
        // before recursing into children.
        let split_col = original_split_col.expect("internal node missing split_col");
        let left_orig = original_left.expect("internal node missing left_child");
        let right_orig = original_right.expect("internal node missing right_child");
        let left_cell = self.source.nodes[left_orig].cell.clone();
        let own_coord = recover_split_coord(&original_cell, &left_cell, &split_col);

        let mut left_cands: Vec<(String, SplitCoord)> = Vec::new();
        let mut right_cands: Vec<(String, SplitCoord)> = Vec::new();
        for (col, c) in candidates {
            if col == split_col {
                if let Some(own) = own_coord.as_ref() {
                    if c.matches(own) {
                        // Already applied at this internal node.
                        continue;
                    }
                    match (own, &c) {
                        (
                            SplitCoord::Continuous { threshold: t_own },
                            SplitCoord::Continuous { threshold: t_cand },
                        ) => {
                            if t_cand < t_own {
                                left_cands.push((col, c));
                            } else {
                                right_cands.push((col, c));
                            }
                            continue;
                        }
                        (
                            SplitCoord::Integer { threshold: t_own },
                            SplitCoord::Integer { threshold: t_cand },
                        ) => {
                            if t_cand < t_own {
                                left_cands.push((col, c));
                            } else {
                                right_cands.push((col, c));
                            }
                            continue;
                        }
                        (
                            SplitCoord::Quantized {
                                threshold_idx: t_own,
                                ..
                            },
                            SplitCoord::Quantized {
                                threshold_idx: t_cand,
                                ..
                            },
                        ) => {
                            if t_cand < t_own {
                                left_cands.push((col, c));
                            } else {
                                right_cands.push((col, c));
                            }
                            continue;
                        }
                        _ => {
                            // Categorical (or type mismatch): forward to both;
                            // the leaf-level interior test will prune.
                            left_cands.push((col.clone(), c.clone()));
                            right_cands.push((col, c));
                            continue;
                        }
                    }
                } else {
                    left_cands.push((col.clone(), c.clone()));
                    right_cands.push((col, c));
                    continue;
                }
            }
            // Different column: forward to both subtrees.
            left_cands.push((col.clone(), c.clone()));
            right_cands.push((col, c));
        }

        let new_left = self.walk(left_orig, left_cands, Some(new_idx));
        let new_right = self.walk(right_orig, right_cands, Some(new_idx));
        self.arena[new_idx].left_child = Some(new_left);
        self.arena[new_idx].right_child = Some(new_right);

        new_idx
    }

    /// Subdivide the leaf at `new_idx` by every applicable candidate,
    /// growing a binary sub-tree of synthetic splits and refined leaves.
    ///
    /// All refined sub-leaves keep `w_xy`, `w_x`, `w_y` literally equal to
    /// the source leaf's values — refinement does not redistribute weights.
    /// To keep predictions invariant under refinement,
    /// [`ConditionedCell::from_fitted_node`](crate::predict::conditioned_cell::ConditionedCell::from_fitted_node)
    /// rescales the conditional mass by `cell.target_volume() / inherited_target_volume`
    /// at prediction time, so that refined cells from one source leaf sum
    /// back to the source leaf's mass.
    ///
    /// The caller has already filtered `candidates` to those interior to the
    /// arena[new_idx] cell at entry; deeper recursion re-checks because
    /// sub-cells shrink as we descend.
    fn refine_leaf_subtree(
        &mut self,
        new_idx: usize,
        source_w_xy: f64,
        source_w_x: f64,
        source_w_y: f64,
        inherited_vol: f64,
        candidates: Vec<(String, SplitCoord)>,
    ) {
        let cell = self.arena[new_idx].cell.clone();
        let depth = self.arena[new_idx].depth;

        let chosen_idx = candidates
            .iter()
            .position(|(col, c)| is_interior(&cell, col, c));

        let Some(idx_chosen) = chosen_idx else {
            // No candidate applies here. The node at new_idx stays a leaf;
            // inherited_target_volume was set by the caller when it pushed
            // this node (or left unset for the original-leaf entry).
            return;
        };

        let (col, coord) = candidates[idx_chosen].clone();

        // For categorical splits, restrict the subset to the active categories
        // in the current cell so apply_split produces a non-empty left.
        let parent_active = cell.get_rule(&col).and_then(|rule| {
            rule.as_any()
                .downcast_ref::<BelongsTo>()
                .map(|bt| bt.values.clone())
        });

        let op = coord.to_split_op(parent_active.as_ref());
        let none_to_left = true;
        let (left_cell, right_cell) = cell.apply_split(&col, op.as_ref(), none_to_left);

        let left_idx = self.arena.len();
        self.arena.push(FittedNode {
            cell: left_cell,
            w_xy: source_w_xy,
            w_x: source_w_x,
            w_y: source_w_y,
            depth: depth + 1,
            parent: Some(new_idx),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
            inherited_target_volume: Some(inherited_vol),
        });
        let right_idx = self.arena.len();
        self.arena.push(FittedNode {
            cell: right_cell,
            w_xy: source_w_xy,
            w_x: source_w_x,
            w_y: source_w_y,
            depth: depth + 1,
            parent: Some(new_idx),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
            inherited_target_volume: Some(inherited_vol),
        });

        // Promote `new_idx` from leaf to internal. Clear any inherited
        // marker — internal nodes don't carry leaf-level density semantics.
        let kind = split_kind_for_col(&col);
        {
            let node = &mut self.arena[new_idx];
            node.is_leaf = false;
            node.left_child = Some(left_idx);
            node.right_child = Some(right_idx);
            node.split_col = Some(col.clone());
            node.split_kind = Some(kind);
            node.inherited_target_volume = None;
        }

        self.synthetic_records.push(SplitRecord {
            parent_index: new_idx,
            col_name: col.clone(),
            split_kind: kind,
            gain: 0.0,
            left_child_index: left_idx,
            right_child_index: right_idx,
        });

        let mut remaining = candidates;
        remaining.remove(idx_chosen);

        self.refine_leaf_subtree(
            left_idx,
            source_w_xy,
            source_w_x,
            source_w_y,
            inherited_vol,
            remaining.clone(),
        );
        self.refine_leaf_subtree(
            right_idx,
            source_w_xy,
            source_w_x,
            source_w_y,
            inherited_vol,
            remaining,
        );
    }

    /// Finalize the refined tree.
    fn into_tree(mut self) -> Tree {
        let mut history: Vec<SplitRecord> =
            Vec::with_capacity(self.source.split_history.len() + self.synthetic_records.len());
        for rec in &self.source.split_history {
            // Every node referenced by the original split_history should have
            // been visited; fall back to identity for safety.
            let p = self
                .orig_to_new
                .get(&rec.parent_index)
                .copied()
                .unwrap_or(rec.parent_index);
            let l = self
                .orig_to_new
                .get(&rec.left_child_index)
                .copied()
                .unwrap_or(rec.left_child_index);
            let r = self
                .orig_to_new
                .get(&rec.right_child_index)
                .copied()
                .unwrap_or(rec.right_child_index);
            history.push(SplitRecord {
                parent_index: p,
                col_name: rec.col_name.clone(),
                split_kind: rec.split_kind,
                gain: rec.gain,
                left_child_index: l,
                right_child_index: r,
            });
        }
        history.append(&mut self.synthetic_records);

        let leaves: Vec<usize> = self
            .arena
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_leaf)
            .map(|(i, _)| i)
            .collect();

        Tree {
            nodes: self.arena,
            leaves,
            split_history: history,
        }
    }
}

// ---------------------------------------------------------------------------
// Tree::refined
// ---------------------------------------------------------------------------

impl Tree {
    /// Refine the tree so every split coordinate ever fired (in any branch)
    /// is propagated across the entire space.
    ///
    /// Every leaf in the returned tree has the property that no unique split
    /// coordinate of the original tree lies strictly interior to its rule
    /// for the corresponding column. Refined leaves carry an
    /// [`inherited_target_volume`](FittedNode::inherited_target_volume) equal
    /// to the source leaf's geometric `target_volume`, so
    /// [`FittedNode::conditional_density`] reports the source leaf's density
    /// even though the refined cell is strictly smaller.
    ///
    /// Both feature (`XSplit`) and target (`YSplit`) coordinates participate
    /// in refinement. Categorical splits are propagated as unique active-set
    /// subsets; an interior subset on a categorical leaf must intersect the
    /// leaf's active set as a non-empty proper subset.
    ///
    /// Original split records are carried over (with their indices remapped
    /// into the new arena) and synthetic refinement splits are appended to
    /// [`split_history`](Tree::split_history) with `gain = 0.0`.
    pub fn refined(&self) -> Tree {
        let coords_per_col = collect_unique_split_coords(self);
        let candidates: Vec<(String, SplitCoord)> = coords_per_col
            .into_iter()
            .flat_map(|(col, coords)| {
                let col = col.clone();
                coords.into_iter().map(move |c| (col.clone(), c))
            })
            .collect();

        let mut refiner = Refiner::new(self);
        refiner.walk(0, candidates, None);
        refiner.into_tree()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::rule::DynRule;
    use std::sync::Arc;

    // ── Fixture helpers ────────────────────────────────────────────────

    fn ci(lo: f64, hi: f64, low_closed: bool, high_closed: bool) -> Box<dyn DynRule> {
        Box::new(ContinuousInterval::new(
            lo,
            hi,
            low_closed,
            high_closed,
            Some((0.0, 10.0)),
            false,
        )) as Box<dyn DynRule>
    }

    fn full_ci(domain: (f64, f64)) -> Box<dyn DynRule> {
        Box::new(ContinuousInterval::new(
            domain.0,
            domain.1,
            true,
            true,
            Some(domain),
            false,
        )) as Box<dyn DynRule>
    }

    fn leaf(
        cell: Cell,
        parent: usize,
        depth: usize,
        w_xy: f64,
        w_x: f64,
        w_y: f64,
    ) -> FittedNode {
        FittedNode {
            cell,
            w_xy,
            w_x,
            w_y,
            depth,
            parent: Some(parent),
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
            inherited_target_volume: None,
        }
    }

    fn internal(
        cell: Cell,
        parent: Option<usize>,
        depth: usize,
        w_xy: f64,
        w_x: f64,
        w_y: f64,
        split_col: &str,
        split_kind: SplitKind,
        left_idx: usize,
        right_idx: usize,
    ) -> FittedNode {
        FittedNode {
            cell,
            w_xy,
            w_x,
            w_y,
            depth,
            parent,
            left_child: Some(left_idx),
            right_child: Some(right_idx),
            is_leaf: false,
            split_col: Some(split_col.to_string()),
            split_kind: Some(split_kind),
            inherited_target_volume: None,
        }
    }

    fn split_rec(
        parent_index: usize,
        col_name: &str,
        split_kind: SplitKind,
        gain: f64,
        left_child_index: usize,
        right_child_index: usize,
    ) -> SplitRecord {
        SplitRecord {
            parent_index,
            col_name: col_name.to_string(),
            split_kind,
            gain,
            left_child_index,
            right_child_index,
        }
    }

    /// Tree where the root splits on a target_y rule at 5, leaving the right
    /// branch as a leaf and refining the left branch further on x at 3.
    /// Unique splits: `target__y` = 5 (Y), `x` = 3 (X). Only the right leaf
    /// will be refined (it spans the full x domain).
    fn tree_target_then_x() -> Tree {
        let dom_x = (0.0, 10.0);
        let dom_y = (0.0, 10.0);

        let root_cell = Cell::new()
            .with_rule("x", full_ci(dom_x))
            .with_rule("target__y", full_ci(dom_y));
        let left_cell = Cell::new()
            .with_rule("x", full_ci(dom_x))
            .with_rule("target__y", ci(0.0, 5.0, true, false));
        let right_cell = Cell::new()
            .with_rule("x", full_ci(dom_x))
            .with_rule("target__y", ci(5.0, 10.0, true, true));
        let left_left_cell = Cell::new()
            .with_rule("x", ci(0.0, 3.0, true, false))
            .with_rule("target__y", ci(0.0, 5.0, true, false));
        let left_right_cell = Cell::new()
            .with_rule("x", ci(3.0, 10.0, true, true))
            .with_rule("target__y", ci(0.0, 5.0, true, false));

        let nodes = vec![
            // 0: root (target__y split at 5)
            internal(
                root_cell,
                None,
                0,
                100.0,
                100.0,
                100.0,
                "target__y",
                SplitKind::YSplit,
                1,
                2,
            ),
            // 1: left (x split at 3)
            internal(
                left_cell,
                Some(0),
                1,
                50.0,
                100.0,
                50.0,
                "x",
                SplitKind::XSplit,
                3,
                4,
            ),
            // 2: right (leaf)
            leaf(right_cell, 0, 1, 50.0, 100.0, 50.0),
            // 3: left-left (leaf)
            leaf(left_left_cell, 1, 2, 20.0, 30.0, 50.0),
            // 4: left-right (leaf)
            leaf(left_right_cell, 1, 2, 30.0, 70.0, 50.0),
        ];

        Tree {
            nodes,
            leaves: vec![2, 3, 4],
            split_history: vec![
                split_rec(0, "target__y", SplitKind::YSplit, 2.5, 1, 2),
                split_rec(1, "x", SplitKind::XSplit, 1.0, 3, 4),
            ],
        }
    }

    /// Tree where the root splits on x at 5, with the left branch refined
    /// on target_y at 3 and the right branch left as a leaf.
    /// Unique splits: x = 5 (X), target__y = 3 (Y). Only the right leaf
    /// will be refined (it spans the full target_y domain).
    fn tree_x_then_target() -> Tree {
        let dom_x = (0.0, 10.0);
        let dom_y = (0.0, 10.0);

        let root_cell = Cell::new()
            .with_rule("x", full_ci(dom_x))
            .with_rule("target__y", full_ci(dom_y));
        let left_cell = Cell::new()
            .with_rule("x", ci(0.0, 5.0, true, false))
            .with_rule("target__y", full_ci(dom_y));
        let right_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("target__y", full_ci(dom_y));
        let left_bot_cell = Cell::new()
            .with_rule("x", ci(0.0, 5.0, true, false))
            .with_rule("target__y", ci(0.0, 3.0, true, false));
        let left_top_cell = Cell::new()
            .with_rule("x", ci(0.0, 5.0, true, false))
            .with_rule("target__y", ci(3.0, 10.0, true, true));

        let nodes = vec![
            // 0: root (x split at 5)
            internal(
                root_cell,
                None,
                0,
                100.0,
                100.0,
                100.0,
                "x",
                SplitKind::XSplit,
                1,
                2,
            ),
            // 1: left (target__y split at 3)
            internal(
                left_cell,
                Some(0),
                1,
                40.0,
                50.0,
                100.0,
                "target__y",
                SplitKind::YSplit,
                3,
                4,
            ),
            // 2: right (leaf, w_xy=60, w_x=50, w_y=100, target_volume=10)
            leaf(right_cell, 0, 1, 60.0, 50.0, 100.0),
            // 3: left-bottom (leaf)
            leaf(left_bot_cell, 1, 2, 10.0, 50.0, 30.0),
            // 4: left-top (leaf)
            leaf(left_top_cell, 1, 2, 30.0, 50.0, 70.0),
        ];

        Tree {
            nodes,
            leaves: vec![2, 3, 4],
            split_history: vec![
                split_rec(0, "x", SplitKind::XSplit, 1.5, 1, 2),
                split_rec(1, "target__y", SplitKind::YSplit, 0.8, 3, 4),
            ],
        }
    }

    /// Trivial tree with just a root leaf (no splits).
    fn tree_root_only() -> Tree {
        let cell = Cell::new()
            .with_rule("x", full_ci((0.0, 10.0)))
            .with_rule("target__y", full_ci((0.0, 10.0)));
        let root = FittedNode {
            cell,
            w_xy: 100.0,
            w_x: 100.0,
            w_y: 100.0,
            depth: 0,
            parent: None,
            left_child: None,
            right_child: None,
            is_leaf: true,
            split_col: None,
            split_kind: None,
            inherited_target_volume: None,
        };
        Tree {
            nodes: vec![root],
            leaves: vec![0],
            split_history: vec![],
        }
    }

    /// Categorical tree where one branch refines the categorical column
    /// further than the other, leaving the other branch's leaf with a
    /// strictly-larger active set than the deeper branches' leaves.
    ///
    /// Structure (color domain = {R, G, B, Y} → codes 0..3):
    ///
    /// ```text
    /// root (x split at 5)
    /// ├── left (x<5, color ∈ {R,G,B,Y})  [LEAF]
    /// └── right (x>=5)
    ///     └── color split, subset_left = {R, G}
    ///         ├── right-left  color={R,G}
    ///         │   └── color split, subset_left = {R}
    ///         │       ├── color={R}  [LEAF]
    ///         │       └── color={G}  [LEAF]
    ///         └── right-right color={B,Y} [LEAF]
    /// ```
    ///
    /// Unique splits: x=5 (X), color={R,G} (X), color={R} (X). Refining the
    /// `left` leaf should yield three sub-leaves matching the equivalence
    /// classes induced by the two categorical subsets: {R}, {G}, {B,Y}.
    fn tree_categorical() -> Tree {
        use std::collections::HashSet;

        let domain: Arc<Vec<usize>> = Arc::new(vec![0, 1, 2, 3]);
        let names: Arc<Vec<String>> = Arc::new(
            ["R", "G", "B", "Y"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );

        let bt = |codes: &[usize]| -> Box<dyn DynRule> {
            Box::new(BelongsTo::new(
                codes.iter().copied().collect::<HashSet<_>>(),
                Arc::clone(&domain),
                Arc::clone(&names),
                false,
            )) as Box<dyn DynRule>
        };

        let dom_x = (0.0, 10.0);
        let root_cell = Cell::new()
            .with_rule("x", full_ci(dom_x))
            .with_rule("color", bt(&[0, 1, 2, 3]));
        let left_cell = Cell::new()
            .with_rule("x", ci(0.0, 5.0, true, false))
            .with_rule("color", bt(&[0, 1, 2, 3]));
        let right_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("color", bt(&[0, 1, 2, 3]));
        let rl_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("color", bt(&[0, 1]));
        let rr_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("color", bt(&[2, 3]));
        let rll_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("color", bt(&[0]));
        let rlr_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("color", bt(&[1]));

        let nodes = vec![
            // 0: root (x split at 5)
            internal(
                root_cell,
                None,
                0,
                100.0,
                100.0,
                100.0,
                "x",
                SplitKind::XSplit,
                1,
                2,
            ),
            // 1: left (leaf)
            leaf(left_cell, 0, 1, 50.0, 50.0, 100.0),
            // 2: right (color split, subset_left={R,G})
            internal(
                right_cell,
                Some(0),
                1,
                50.0,
                50.0,
                100.0,
                "color",
                SplitKind::XSplit,
                3,
                4,
            ),
            // 3: right-left (color split, subset_left={R})
            internal(
                rl_cell,
                Some(2),
                2,
                30.0,
                30.0,
                30.0,
                "color",
                SplitKind::XSplit,
                5,
                6,
            ),
            // 4: right-right (leaf)
            leaf(rr_cell, 2, 2, 20.0, 20.0, 20.0),
            // 5: right-left-left (leaf, color={R})
            leaf(rll_cell, 3, 3, 18.0, 18.0, 18.0),
            // 6: right-left-right (leaf, color={G})
            leaf(rlr_cell, 3, 3, 12.0, 12.0, 12.0),
        ];

        Tree {
            nodes,
            leaves: vec![1, 4, 5, 6],
            split_history: vec![
                split_rec(0, "x", SplitKind::XSplit, 1.0, 1, 2),
                split_rec(2, "color", SplitKind::XSplit, 0.5, 3, 4),
                split_rec(3, "color", SplitKind::XSplit, 0.3, 5, 6),
            ],
        }
    }

    fn assert_no_interior_crossing(tree: &Tree) {
        let coords = collect_unique_split_coords(tree);
        for &leaf_idx in &tree.leaves {
            let cell = &tree.nodes[leaf_idx].cell;
            for (col, list) in &coords {
                for coord in list {
                    assert!(
                        !is_interior(cell, col, coord),
                        "leaf {leaf_idx}: unique split {coord:?} on column {col} \
                         lies strictly interior to its rule"
                    );
                }
            }
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn refined_root_only_is_unchanged() {
        let tree = tree_root_only();
        let refined = tree.refined();

        assert_eq!(refined.n_nodes(), 1);
        assert_eq!(refined.n_leaves(), 1);
        assert!(refined.nodes[0].inherited_target_volume.is_none());
        assert!(refined.split_history.is_empty());
    }

    #[test]
    fn refined_propagates_x_split_to_target_branch_leaf() {
        // tree_target_then_x: only the right leaf (target_y in [5,10], x full)
        // should be refined by the X-split at x=3.
        let tree = tree_target_then_x();
        let refined = tree.refined();

        // Original: 3 leaves. After refining the right leaf into 2 sub-leaves: 4.
        assert_eq!(refined.n_leaves(), 4);
        assert_no_interior_crossing(&refined);

        // Synthetic record for the X refinement should be present with gain=0.
        let synthetic: Vec<_> = refined
            .split_history
            .iter()
            .filter(|r| r.gain == 0.0 && r.col_name == "x")
            .collect();
        assert_eq!(synthetic.len(), 1);
        assert_eq!(synthetic[0].split_kind, SplitKind::XSplit);

        // The refined sub-leaves of the original right leaf should carry the
        // source leaf's target_volume (= 5) as inherited_target_volume, and
        // their cell.target_volume() should equal it because an X-split does
        // not change target_volume.
        let inherited_leaves: Vec<&FittedNode> = refined
            .leaf_nodes()
            .into_iter()
            .filter(|n| n.inherited_target_volume.is_some())
            .collect();
        assert_eq!(inherited_leaves.len(), 2);
        for node in inherited_leaves {
            let geom = node.cell.target_volume();
            let inh = node.inherited_target_volume.unwrap();
            assert!((inh - 5.0).abs() < 1e-10, "inherited vol should be 5.0");
            assert!((geom - inh).abs() < 1e-10, "X-only refinement preserves geometric volume");
            assert_eq!(node.effective_target_volume(), inh);
        }
    }

    #[test]
    fn refined_y_split_uses_inherited_volume_to_preserve_density() {
        // tree_x_then_target: only the right leaf (x in [5,10], target_y full)
        // should be refined by the Y-split at target_y = 3.
        let tree = tree_x_then_target();
        let refined = tree.refined();

        assert_eq!(refined.n_leaves(), 4);
        assert_no_interior_crossing(&refined);

        let inherited_leaves: Vec<&FittedNode> = refined
            .leaf_nodes()
            .into_iter()
            .filter(|n| n.inherited_target_volume.is_some())
            .collect();
        assert_eq!(inherited_leaves.len(), 2);

        // Source right leaf: target_y full → target_volume = 10, density =
        // 60 / (50 * 10) = 0.12. Each refined sub-leaf must report that
        // density even though its geometric target_volume is 3 or 7.
        let source_density = 60.0 / (50.0 * 10.0);
        let mut geom_volumes: Vec<f64> = inherited_leaves
            .iter()
            .map(|n| n.cell.target_volume())
            .collect();
        geom_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((geom_volumes[0] - 3.0).abs() < 1e-10);
        assert!((geom_volumes[1] - 7.0).abs() < 1e-10);

        for node in inherited_leaves {
            let inh = node.inherited_target_volume.unwrap();
            assert!((inh - 10.0).abs() < 1e-10, "inherited vol should be 10.0");
            assert!(
                node.cell.target_volume() < inh,
                "Y refinement must shrink geometric target_volume below inherited"
            );
            assert_eq!(node.effective_target_volume(), inh);
            assert!(
                (node.conditional_density() - source_density).abs() < 1e-12,
                "refined leaf density {} does not match source density {}",
                node.conditional_density(),
                source_density
            );
        }
    }

    #[test]
    fn refined_density_matches_original_routing() {
        // For every leaf of the original tree, pick an interior point and
        // route it through the refined tree using a HashMap. The refined
        // leaf's density must equal the original leaf's density.
        use crate::rule::DynValue;
        use std::collections::HashMap;

        for tree_fn in [tree_target_then_x as fn() -> Tree, tree_x_then_target] {
            let tree = tree_fn();
            let refined = tree.refined();

            for &orig_leaf_idx in &tree.leaves {
                let orig = &tree.nodes[orig_leaf_idx];
                // Pick the midpoint of every rule as the representative point.
                let mut row: HashMap<String, Option<DynValue>> = HashMap::new();
                for (col, rule) in &orig.cell.rules {
                    if let Some(ci) = rule.as_any().downcast_ref::<ContinuousInterval>() {
                        let mid = (ci.low + ci.high) / 2.0;
                        row.insert(col.clone(), Some(DynValue::Continuous(mid)));
                    }
                }
                let refined_idx = refined.predict_leaf_from_map(&row);
                let refined_leaf = &refined.nodes[refined_idx];
                assert!(refined_leaf.is_leaf);

                let orig_density = orig.conditional_density();
                let refined_density = refined_leaf.conditional_density();
                assert!(
                    (orig_density - refined_density).abs() < 1e-12,
                    "original leaf {orig_leaf_idx} density {orig_density} \
                     differs from refined leaf {refined_idx} density {refined_density}"
                );
            }
        }
    }

    #[test]
    fn refined_invariant_no_interior_crossings_after_one_pass() {
        for tree_fn in [
            tree_target_then_x as fn() -> Tree,
            tree_x_then_target,
            tree_categorical,
            tree_root_only,
        ] {
            let refined = tree_fn().refined();
            assert_no_interior_crossing(&refined);
        }
    }

    #[test]
    fn refined_is_idempotent() {
        for tree_fn in [
            tree_target_then_x as fn() -> Tree,
            tree_x_then_target,
            tree_categorical,
            tree_root_only,
        ] {
            let once = tree_fn().refined();
            let twice = once.refined();
            assert_eq!(
                once.n_leaves(),
                twice.n_leaves(),
                "refined().refined() should not produce additional leaves"
            );
            assert_eq!(once.n_nodes(), twice.n_nodes());
        }
    }

    #[test]
    fn refined_categorical_partitions_into_equivalence_classes() {
        use std::collections::HashSet;

        let tree = tree_categorical();
        let refined = tree.refined();

        // The original left leaf (x < 5, color = {R,G,B,Y}) is the only leaf
        // affected by the categorical refinement. After applying both unique
        // subsets ({R,G} and {R}), the equivalence classes are {R}, {G}, {B,Y}.
        // Combined with the x < 5 prefix, refined_left has 3 sub-leaves.
        // Plus the 3 unchanged right-side leaves: total = 6.
        assert_eq!(refined.n_leaves(), 6);
        assert_no_interior_crossing(&refined);

        // Among leaves with x rule [0, 5), collect their color active sets.
        let mut left_color_sets: Vec<HashSet<usize>> = Vec::new();
        for &leaf_idx in &refined.leaves {
            let cell = &refined.nodes[leaf_idx].cell;
            let x_rule = cell.get_rule("x").unwrap();
            let ci = x_rule
                .as_any()
                .downcast_ref::<ContinuousInterval>()
                .unwrap();
            if ci.low == 0.0 && ci.high == 5.0 {
                let color_rule = cell.get_rule("color").unwrap();
                let bt = color_rule.as_any().downcast_ref::<BelongsTo>().unwrap();
                left_color_sets.push(bt.values.clone());
            }
        }
        assert_eq!(left_color_sets.len(), 3);
        let expected: HashSet<Vec<usize>> = [vec![0], vec![1], vec![2, 3]]
            .into_iter()
            .map(|mut v| {
                v.sort();
                v
            })
            .collect();
        let got: HashSet<Vec<usize>> = left_color_sets
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort();
                v
            })
            .collect();
        assert_eq!(got, expected);

        // Density preservation on the refined left-side leaves: with no
        // target__ rules in this tree, target_volume defaults to 1.0 and
        // the source left leaf's density is w_xy/w_x = 50/50 = 1.0.
        let source_density = 50.0 / 50.0;
        for &leaf_idx in &refined.leaves {
            let node = &refined.nodes[leaf_idx];
            if node.inherited_target_volume.is_some() {
                assert!(
                    (node.conditional_density() - source_density).abs() < 1e-12,
                    "refined categorical leaf density {} != source {}",
                    node.conditional_density(),
                    source_density,
                );
            }
        }
    }

    #[test]
    fn refined_serde_roundtrip_preserves_inherited_target_volume() {
        let refined = tree_x_then_target().refined();

        let bytes = bincode::serialize(&refined).expect("serialize");
        let restored: Tree = bincode::deserialize(&bytes).expect("deserialize");

        assert_eq!(refined.n_nodes(), restored.n_nodes());
        assert_eq!(refined.n_leaves(), restored.n_leaves());
        assert_eq!(refined.split_history.len(), restored.split_history.len());

        for (a, b) in refined.nodes.iter().zip(restored.nodes.iter()) {
            match (a.inherited_target_volume, b.inherited_target_volume) {
                (Some(x), Some(y)) => assert!((x - y).abs() < 1e-12),
                (None, None) => {}
                _ => panic!("inherited_target_volume mismatch after roundtrip"),
            }
            assert!((a.conditional_density() - b.conditional_density()).abs() < 1e-12);
        }
    }

    #[test]
    fn refined_split_history_remaps_indices() {
        // After refinement, original SplitRecords should refer to nodes in the
        // new arena (not the original). The mapped indices must satisfy:
        // - parent_index points to an internal node
        // - left/right_child_index point to nodes whose `parent` field equals
        //   the mapped parent_index.
        let refined = tree_target_then_x().refined();
        for rec in &refined.split_history {
            let parent = &refined.nodes[rec.parent_index];
            assert!(!parent.is_leaf);
            let left = &refined.nodes[rec.left_child_index];
            let right = &refined.nodes[rec.right_child_index];
            assert_eq!(left.parent, Some(rec.parent_index));
            assert_eq!(right.parent, Some(rec.parent_index));
        }
    }

    #[test]
    fn refined_feature_importances_unchanged_by_synthetic_splits() {
        // Synthetic refinement splits use gain=0.0, so they must not contribute
        // to feature importances (which filter on gain > 0).
        let tree = tree_target_then_x();
        let importances = tree.feature_importances(false);

        let refined = tree.refined();
        let refined_importances = refined.feature_importances(false);

        assert_eq!(importances.len(), refined_importances.len());
        for (k, v) in &importances {
            let r = refined_importances.get(k).copied().unwrap_or(0.0);
            assert!((v - r).abs() < 1e-12, "importance for {k} changed: {v} vs {r}");
        }
    }

    #[test]
    fn refined_uses_collect_unique_split_coords_deduplication() {
        // Build a tree where the same threshold appears in two branches.
        // After collection it must be deduplicated.
        let dom_x = (0.0, 10.0);
        let dom_y = (0.0, 10.0);

        let root_cell = Cell::new()
            .with_rule("x", full_ci(dom_x))
            .with_rule("target__y", full_ci(dom_y));
        let left_cell = Cell::new()
            .with_rule("x", ci(0.0, 5.0, true, false))
            .with_rule("target__y", full_ci(dom_y));
        let right_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("target__y", full_ci(dom_y));
        // Both sides split target_y at the same threshold = 3.
        let ll_cell = Cell::new()
            .with_rule("x", ci(0.0, 5.0, true, false))
            .with_rule("target__y", ci(0.0, 3.0, true, false));
        let lr_cell = Cell::new()
            .with_rule("x", ci(0.0, 5.0, true, false))
            .with_rule("target__y", ci(3.0, 10.0, true, true));
        let rl_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("target__y", ci(0.0, 3.0, true, false));
        let rr_cell = Cell::new()
            .with_rule("x", ci(5.0, 10.0, true, true))
            .with_rule("target__y", ci(3.0, 10.0, true, true));

        let nodes = vec![
            internal(
                root_cell,
                None,
                0,
                100.0,
                100.0,
                100.0,
                "x",
                SplitKind::XSplit,
                1,
                2,
            ),
            internal(
                left_cell,
                Some(0),
                1,
                50.0,
                50.0,
                100.0,
                "target__y",
                SplitKind::YSplit,
                3,
                4,
            ),
            internal(
                right_cell,
                Some(0),
                1,
                50.0,
                50.0,
                100.0,
                "target__y",
                SplitKind::YSplit,
                5,
                6,
            ),
            leaf(ll_cell, 1, 2, 15.0, 50.0, 30.0),
            leaf(lr_cell, 1, 2, 35.0, 50.0, 70.0),
            leaf(rl_cell, 2, 2, 15.0, 50.0, 30.0),
            leaf(rr_cell, 2, 2, 35.0, 50.0, 70.0),
        ];

        let tree = Tree {
            nodes,
            leaves: vec![3, 4, 5, 6],
            split_history: vec![
                split_rec(0, "x", SplitKind::XSplit, 1.0, 1, 2),
                split_rec(1, "target__y", SplitKind::YSplit, 0.5, 3, 4),
                split_rec(2, "target__y", SplitKind::YSplit, 0.5, 5, 6),
            ],
        };

        let coords = collect_unique_split_coords(&tree);
        // target__y should appear once (despite two records), x once.
        assert_eq!(coords.get("target__y").map(|v| v.len()), Some(1));
        assert_eq!(coords.get("x").map(|v| v.len()), Some(1));

        // Refining: every leaf already respects both unique coords, so the
        // refined tree must equal the original in leaf count.
        let refined = tree.refined();
        assert_eq!(refined.n_leaves(), 4);
        assert_no_interior_crossing(&refined);
    }
}
