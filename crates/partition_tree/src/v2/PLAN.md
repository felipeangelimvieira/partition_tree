# Partition Tree v2 — Implementation Plan

## Goal

Refactor the partition tree into a trait-driven, modular system where every dtype-specific
concern lives behind a trait object, and the tree builder is a generic orchestrator.
Focus on single-tree first; Forest is deferred to a follow-up.

---

## Module Layout

```
src/v2/
  mod.rs              — re-exports all public types
  loss.rs             — LossFunc trait + ConditionalLogLoss, BalancedLogLoss
  rule.rs             — re-exports v1 Rule/ContinuousInterval/BelongsToGeneric/RuleType
  cell.rs             — Cell (HashMap<String, RuleType>)
  split_result.rs     — SplitKind, SplitPoint, SplitDetail, SplitRestrictions, CellStats
  dataset_view.rs     — DatasetView + ColumnView traits, PolarsDatasetView + PolarsColumnView
  node.rs             — SortedIndices, Node, presorted-index propagation
  column_split.rs     — ColumnSplitSearcher trait + ContinuousColumnSplitSearcher + CategoricalColumnSplitSearcher
  split_searcher.rs   — SplitSearcher (orchestrates per-column searchers via DTypeRegistry)
  dtype_plugin.rs     — DTypePlugin trait, DTypeRegistry, ContinuousPlugin, CategoricalPlugin
  tree_builder.rs     — TreeBuilder (best-first loop, produces Tree)
  tree.rs             — FittedNode, Tree (predict API, leaf info)
```

---

## Components & Contracts

### 1. `LossFunc` (loss.rs)

```rust
pub struct CellStats {
    pub w_xy: f64,
    pub w_x: f64,
    pub w_y: f64,
    pub volume: f64,
}

pub trait LossFunc: Send + Sync {
    fn cell_loss(&self, stats: &CellStats) -> f64;
    fn gain(&self, parent: &CellStats, left: &CellStats, right: &CellStats) -> f64 {
        self.cell_loss(parent) - (self.cell_loss(left) + self.cell_loss(right))
    }
}
```

**Implementations:**
- `ConditionalLogLoss` — $-w_{xy} \ln\!\bigl(\frac{w_{xy}}{w_x \cdot \mathrm{vol}}\bigr)$
- `BalancedLogLoss` — $-w_{xy} \ln\!\bigl(\frac{w_{xy}}{w_x \cdot w_y}\bigr)$

---

### 2. Rules (rule.rs)

Re-export from `crate::rules`:
- `Rule<T>` trait
- `ContinuousInterval`
- `BelongsToGeneric<T>`, `BelongsTo` (= `BelongsToGeneric<usize>`)
- `RuleType` enum (`Continuous | BelongsTo`)

Add convenience methods on `RuleType`:
- `accept_none() -> bool`
- `volume() -> f64`
- `relative_volume() -> f64`
- `phi_volume() -> f64`
- `mean() -> Vec<f64>`
- `split_continuous(threshold, none_to_left) -> (RuleType, RuleType)`
- `split_categorical(subset, none_to_left) -> (RuleType, RuleType)`
- `evaluate_continuous(value: Option<f64>) -> bool`
- `evaluate_categorical(value: Option<usize>) -> bool`

---

### 3. `Cell` (cell.rs)

```rust
pub struct Cell {
    pub rules: HashMap<String, RuleType>,
}
```

**Methods:**
- `new() -> Self`
- `with_rule(col, rule) -> Self`
- `get_rule(col) -> Option<&RuleType>`
- `volume() -> f64` — product of per-rule volumes
- `relative_volume() -> f64`
- `phi_volume() -> f64`
- `split(col, detail, none_to_left) -> (Cell, Cell)` — clone rules, replace split-column rule in each child
- `contains_continuous(col, value) -> bool`
- `contains_categorical(col, value) -> bool`
- `target_rules() -> impl Iterator` — rules on `target__`-prefixed columns

---

### 4. `SplitResult` types (split_result.rs)

```rust
pub enum SplitKind { XSplit, YSplit }

pub enum SplitDetail {
    Continuous { threshold: f64, k_candidate: usize, p_xy: usize },
    Categorical { subset_left: HashSet<usize> },
}

pub struct SplitPoint {
    pub col_name: String,
    pub split_kind: SplitKind,
    pub none_to_left: bool,
    pub gain: f64,
    pub left_stats: CellStats,
    pub right_stats: CellStats,
    pub detail: SplitDetail,
}

pub struct SplitRestrictions {
    pub min_samples_xy: f64,
    pub min_samples_x: f64,
    pub min_samples_y: f64,
    pub min_gain: f64,
    pub min_volume: f64,
    pub max_depth: usize,
    pub min_samples_split: f64,
}
```

`SplitRestrictions::is_valid_children(left, right, depth) -> bool`

---

### 5. `DatasetView` / `ColumnView` (dataset_view.rs)

```rust
pub enum LogicalDType { Continuous, Categorical }

pub trait ColumnView: Send + Sync {
    fn name(&self) -> &str;
    fn len(&self) -> usize;
    fn logical_dtype(&self) -> LogicalDType;
    fn get_f64(&self, idx: usize) -> Option<f64>;
    fn get_cat(&self, idx: usize) -> Option<usize>;
    fn is_null(&self, idx: usize) -> bool;
}

pub trait DatasetView: Send + Sync {
    fn n_rows(&self) -> usize;
    fn n_columns(&self) -> usize;
    fn column(&self, name: &str) -> Option<&dyn ColumnView>;
    fn columns(&self) -> Vec<&dyn ColumnView>;
    fn feature_columns(&self) -> Vec<&dyn ColumnView>;
    fn target_columns(&self) -> Vec<&dyn ColumnView>;
    fn column_names(&self) -> Vec<&str>;
    fn initial_sorted_indices(&self, col: &str) -> Vec<u32>;
}
```

**Polars implementations:** `PolarsColumnView`, `PolarsDatasetView`.

---

### 6. `Node` (node.rs)

```rust
pub struct SortedIndices {
    pub sorted_x:  HashMap<String, Vec<u32>>,
    pub sorted_y:  HashMap<String, Vec<u32>>,
    pub sorted_xy: HashMap<String, Vec<u32>>,
}

pub struct Node {
    pub cell: Cell,
    pub w_xy: f64,
    pub w_x: f64,
    pub w_y: f64,
    pub depth: usize,
    pub sorted: SortedIndices,
}
```

**Key methods:**
- `Node::root(dataset, weights_xy, weights_x, weights_y) -> Node`
- `Node::propagate_children(parent, split_point, dataset, weights) -> (Node, Node)` — stable partition of all presorted lists using the `go_left` oracle
- `conditional_density() -> f64`

---

### 7. `ColumnSplitSearcher` (column_split.rs)

```rust
pub trait ColumnSplitSearcher: Send + Sync {
    fn search(
        &self,
        node: &Node,
        col: &dyn ColumnView,
        split_kind: SplitKind,
        dataset: &dyn DatasetView,
        loss: &dyn LossFunc,
        restrictions: &SplitRestrictions,
        weights_xy: &[f64],
        weights_x: &[f64],
        weights_y: &[f64],
    ) -> Option<SplitPoint>;
}
```

**Implementations:**
- `ContinuousColumnSplitSearcher` — presorted scan, moving pointer on sorted_xy, prefix sums
- `CategoricalColumnSplitSearcher` — per-category (a_c, b_c, r_c), sort by r_c, prefix scan

---

### 8. `DTypePlugin` + `DTypeRegistry` (dtype_plugin.rs)

```rust
pub trait DTypePlugin: Send + Sync {
    fn logical_dtype(&self) -> LogicalDType;
    fn default_rule(&self, col: &dyn ColumnView) -> RuleType;
    fn split_searcher(&self) -> &dyn ColumnSplitSearcher;
}

pub struct DTypeRegistry {
    plugins: HashMap<LogicalDType, Box<dyn DTypePlugin>>,
}
```

Built-in plugins: `ContinuousPlugin`, `CategoricalPlugin`.

---

### 9. `SplitSearcher` (split_searcher.rs)

```rust
pub struct SplitSearcher {
    pub registry: Arc<DTypeRegistry>,
}
```

`fn find_best_split(node, dataset, loss, restrictions, weights) -> Option<SplitPoint>`

Iterates all columns, dispatches to the appropriate `ColumnSplitSearcher` via the registry.
Feature columns → `XSplit`; target columns → `YSplit`. Column-level parallelism via `rayon`.

---

### 10. `TreeBuilder` (tree_builder.rs)

```rust
pub struct TreeBuilder {
    pub split_searcher: SplitSearcher,
    pub loss: Box<dyn LossFunc>,
    pub restrictions: SplitRestrictions,
    pub max_leaves: usize,
}
```

`fn build(dataset, weights_xy, weights_x, weights_y) -> Tree`

Algorithm:
1. Create root Node with full indices, root Cell from DTypeRegistry defaults.
2. Find best split for root, push into max-heap keyed by gain.
3. Loop (pop best candidate → split cell → propagate sorted indices → find best splits for children → push viable children).
4. Stop when heap empty or max_leaves reached.
5. Collect FittedNodes + leaves → Tree.

---

### 11. `Tree` (tree.rs)

```rust
pub struct FittedNode {
    pub cell: Cell,
    pub w_xy: f64,
    pub w_x: f64,
    pub w_y: f64,
    pub depth: usize,
    pub parent: Option<usize>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub is_leaf: bool,
}

pub struct Tree {
    pub nodes: Vec<FittedNode>,
    pub leaves: Vec<usize>,
}
```

**Methods:**
- `predict_leaf(row) -> usize` — traverse from root
- `leaf_nodes() -> Vec<&FittedNode>`
- `depth() -> usize`
- `n_leaves() -> usize`

---

## Implementation Order

| Step | File                | Dependencies                                               |
| ---- | ------------------- | ---------------------------------------------------------- |
| 1    | `loss.rs`           | none                                                       |
| 2    | `rule.rs`           | `crate::rules`                                             |
| 3    | `cell.rs`           | `rule.rs`                                                  |
| 4    | `split_result.rs`   | `loss.rs`                                                  |
| 5    | `dataset_view.rs`   | `rule.rs`                                                  |
| 6    | `node.rs`           | `cell.rs`, `dataset_view.rs`, `split_result.rs`            |
| 7    | `column_split.rs`   | `node.rs`, `dataset_view.rs`, `loss.rs`, `split_result.rs` |
| 8    | `dtype_plugin.rs`   | `column_split.rs`, `dataset_view.rs`, `rule.rs`            |
| 9    | `split_searcher.rs` | `dtype_plugin.rs`, `node.rs`                               |
| 10   | `tree_builder.rs`   | `split_searcher.rs`, `node.rs`                             |
| 11   | `tree.rs`           | `cell.rs`                                                  |
| 12   | `mod.rs`            | all of the above                                           |

---

## Design Principles

1. **Trait objects for extension** — `LossFunc`, `ColumnSplitSearcher`, `DTypePlugin`, `DatasetView`, `ColumnView` are all trait-based.
2. **No `dyn Any` downcasting** — `RuleType` enum replaces the type-erased `Box<dyn DynOneDimPartition>` pattern.
3. **Densities removed from rules** — densities are computed from node weights and cell volume, not stored per-rule.
4. **Presorted index propagation** — O(d·N log N) total, no per-node re-sorting.
5. **Separation of concerns** — split search, tree building, and prediction are distinct modules.
6. **v1 coexistence** — v2 lives in `src/v2/`, v1 modules remain untouched.
