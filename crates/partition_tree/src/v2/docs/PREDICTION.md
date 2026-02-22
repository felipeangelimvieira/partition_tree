# Prediction — v2 Design Plan

## 1. Overview

Once a tree is fitted, it is defined by a partition (leaf cells) $\pi$ of $\mathcal{Z} := \mathcal{X} \times \mathcal{Y}$.
Users can do queries of the form $\pi[X=x]$, $\pi[X=x, Y=y]$, $\pi[X=x, y \geq y_0]$ (for continuous $Y$), $\pi[X=x, y \in S]$ (for categorical $Y$), etc.
Such conditioning filters subsets of leaf cells in $\pi$, generating a new partition $\pi'$.

When doing conditional prediction, we compute $\pi[X=x]$ for each row, producing a set of matching leaf cells — a **piecewise-constant conditional distribution** over $\mathcal{Y}$.
Each cell $c$ carries:

- **conditional mass**: $m_c = w_{xy}^c / w_x^c$
- **conditional density** (continuous targets): $\hat{f}(y|x) = w_{xy}^c / (w_x^c \cdot V_{\text{target}}^c)$

The prediction module must construct such distributions and expose ergonomic APIs for point prediction, distribution queries, and ensemble averaging.

---

## 2. Architecture

```
src/v2/predict/
  mod.rs                     — re-exports, Tree extension methods
  piecewise_distribution.rs  — PiecewiseConstantDistribution
  target_projector.rs        — TargetProjector (extract per-target-dim info from cells)
```

### Dependency graph

```text
Tree (tree.rs)
  │
  ├─ predict_leaves(dataset)     → Vec<usize>          (already exists)
  │
  └─ predict_distributions(dataset) → Vec<PiecewiseConstantDistribution>  (NEW)
       │
       ▼
  PiecewiseConstantDistribution
       │
       ├─ mean_vector()      → HashMap<String, Vec<f64>>
       ├─ mean_df()          → DataFrame
       ├─ pdf(y_dataset)     → Vec<f64>
       ├─ mass(y_dataset)    → Vec<f64>
       └─ ensemble_with(..) → PiecewiseConstantDistribution
```

---

## 3. Key Types

### 3.1 `ConditionedCell`

A lightweight view of a leaf cell after conditioning on $X = x$.
Only the **target rules** are retained (features are irrelevant after conditioning).

```rust
pub struct ConditionedCell {
    /// Target rules only (column name → DynRule).
    pub target_rules: HashMap<String, Box<dyn DynRule>>,
    /// Conditional mass: w_xy / w_x for this cell.
    pub mass: f64,
}
```

**Key methods:**

- `target_volume() -> f64` — product of target rule volumes
- `conditional_density() -> f64` — `mass / target_volume()`
- `mean_map() -> HashMap<String, Vec<f64>>` — per-target-column mean from `DynRule::mean()`
- `contains_target(row, columns) -> bool` — evaluate target rules against a data row

**Construction** from a `FittedNode`:

```rust
impl ConditionedCell {
    pub fn from_fitted_node(node: &FittedNode) -> Self {
        let target_rules: HashMap<String, Box<dyn DynRule>> = node.cell
            .target_rules()
            .map(|(k, r)| (k.clone(), r.clone_box()))
            .collect();
        let mass = if node.w_x > 0.0 { node.w_xy / node.w_x } else { 0.0 };
        Self { target_rules, mass }
    }
}
```

### 3.2 `PiecewiseConstantDistribution`

The core prediction object.
One per sample row — represents $\hat{P}(Y | X = x)$ as a union of conditioned cells.

```rust
pub struct PiecewiseConstantDistribution {
    /// Conditioned cells from matched leaves.
    pub cells: Vec<ConditionedCell>,
}
```

**Construction flow:**

1. `Tree::predict_leaves(dataset)` → leaf index per row.
2. For each row, collect the matching leaf's `FittedNode` → build `ConditionedCell`.
3. Wrap in `PiecewiseConstantDistribution`.

> **Note:** In v2, each row matches exactly **one** leaf (deterministic tree traversal).
> The distribution may still contain multiple cells in the future if we support
> partial-match (e.g., soft routing or missing features), or when ensembling.

### 3.3 `MeanVector`

A type alias for the per-target-dimension mean representation:

```rust
/// Per-target-column mean vector.
/// Key: target column name (e.g., "target__y1").
/// Value: Vec<f64> whose semantics depend on dtype:
///   - Continuous: [midpoint]    (length 1)
///   - Integer:    [midpoint]    (length 1)
///   - Categorical: [p_0, p_1, ..., p_{K-1}]  (one-hot probability over sorted domain)
pub type MeanVector = HashMap<String, Vec<f64>>;
```

---

## 4. API Design

### 4.1 `PiecewiseConstantDistribution` methods

```rust
impl PiecewiseConstantDistribution {
    /// Weighted mean vector across all cells.
    ///
    /// For each target column, compute:
    ///   mean_col = Σ_c (mass_c / total_mass) * cell_c.mean(col)
    ///
    /// Continuous targets: returns weighted midpoint.
    /// Categorical targets: returns probability vector over sorted domain.
    ///
    /// Handles zero total mass by falling back to uniform weighting.
    pub fn mean_vector(&self) -> MeanVector;

    /// Total mass: Σ_c mass_c.
    pub fn total_mass(&self) -> f64;

    /// Number of cells in this distribution.
    pub fn n_cells(&self) -> usize;

    /// Evaluate the piecewise-constant pdf at target points given as
    /// column views + row indices. For each row, find the matching cell
    /// and return mass / target_volume.
    ///
    /// Returns a Vec<f64> of length `n_rows`.
    pub fn pdf(&self, target_columns: &[&dyn ColumnView], row_indices: &[usize]) -> Vec<f64>;

    /// Like pdf but without dividing by volume (returns mass).
    pub fn mass_at(&self, target_columns: &[&dyn ColumnView], row_indices: &[usize]) -> Vec<f64>;

    /// Create a merged distribution from multiple distributions (for ensembling).
    /// Each source distribution's masses are divided by the number of sources.
    pub fn ensemble(distributions: &[&PiecewiseConstantDistribution]) -> Self;

    /// Iterator over (cell, mass) pairs.
    pub fn iter(&self) -> impl Iterator<Item = &ConditionedCell>;

    /// For continuous targets: return Vec<(pdf_value, low, high)> segments.
    pub fn pdf_segments(&self) -> Vec<(f64, f64, f64)>;

    /// For categorical targets: return category probabilities as
    /// HashMap<String, Vec<(String, f64)>> where keys are target column names
    /// and values are (category_name, probability) pairs.
    /// Requires domain_names from the DynRule (BelongsTo) for decoding.
    pub fn category_probabilities(&self) -> HashMap<String, Vec<f64>>;
}
```

### 4.2 `Tree` extension methods

```rust
impl Tree {
    /// Predict conditional distributions for all rows in a dataset.
    ///
    /// 1. Routes each row to its leaf via `predict_leaves`.
    /// 2. Builds a `PiecewiseConstantDistribution` from each leaf's `FittedNode`.
    ///
    /// Returns a Vec of length `dataset.n_rows()`.
    pub fn predict_distributions(
        &self,
        dataset: &dyn DatasetView,
    ) -> Vec<PiecewiseConstantDistribution>;

    /// Predict mean vectors for all rows.
    ///
    /// Shorthand for `predict_distributions` followed by `mean_vector` on each.
    pub fn predict_mean_vectors(
        &self,
        dataset: &dyn DatasetView,
    ) -> Vec<MeanVector>;

    /// Predict means as a Polars DataFrame.
    ///
    /// Continuous targets → single f64 column per target.
    /// Categorical targets → the category with highest probability (argmax).
    /// Integer targets → rounded midpoint.
    ///
    /// The output DataFrame has `n_rows` rows and one column per target dimension.
    pub fn predict_mean(
        &self,
        dataset: &dyn DatasetView,
    ) -> DataFrame;
}
```

---

## 5. Mean Vector Semantics

The `mean_vector` is the common currency for both single-tree and ensemble prediction.
Its shape is **deterministic** given the tree's target schema:

| Target dtype | `DynRule::mean()` returns  | Meaning                                  |
| ------------ | -------------------------- | ---------------------------------------- |
| Continuous   | `[(low + high) / 2.0]`     | Midpoint of interval                     |
| Integer      | `[(low + high) / 2.0]`     | Midpoint of integer range (as f64)       |
| Categorical  | `[p_0, p_1, ..., p_{K-1}]` | One-hot indicator over **sorted domain** |

The weighted `mean_vector` combines these per-cell means:

$$
\text{mean\_vector}[\text{col}] = \sum_{c} \frac{m_c}{\sum_c m_c} \cdot \text{rule}_c[\text{col}].\text{mean}()
$$

For **categorical targets** after weighting, the vector entries become proper **probability estimates** over the domain categories.

### Fixed-length guarantee

Because every tree built from the same schema uses the same sorted domain for categorical rules and the same dimensionality for continuous/integer rules, `mean_vector` always has the same structure across trees. This makes ensemble averaging trivial:

$$
\text{ensemble\_mean}[\text{col}] = \frac{1}{T} \sum_{t=1}^{T} \text{mean\_vector}_t[\text{col}]
$$

---

## 6. `predict_mean` — DataFrame Output

`predict_mean` converts `mean_vector` into a Polars `DataFrame`:

| Target dtype | DataFrame column type | Conversion                                       |
| ------------ | --------------------- | ------------------------------------------------ |
| Continuous   | `Float64`             | Direct: `mean_vector[col][0]`                    |
| Integer      | `Int64`               | Round: `mean_vector[col][0].round() as i64`      |
| Categorical  | `Enum` / `Utf8`       | Argmax: `domain_names[argmax(mean_vector[col])]` |

**Implementation:**

To decode categorical predictions back to category names, we need access to the domain names from the `BelongsTo` rule. The tree's root cell contains the full domain for each target column, so we can extract `domain_names` from `root.cell.target_rules()` at prediction time.

```rust
fn decode_categorical_mean(mean_vec: &[f64], domain_names: &[String]) -> String {
    let argmax = mean_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    domain_names[argmax].clone()
}
```

---

## 7. Ensembling

Ensembling is a first-class concern. Two approaches:

### 7.1 Mean-vector averaging (preferred for point prediction)

```rust
fn ensemble_mean_vectors(trees: &[Tree], dataset: &dyn DatasetView) -> Vec<MeanVector> {
    let all_means: Vec<Vec<MeanVector>> = trees
        .iter()
        .map(|t| t.predict_mean_vectors(dataset))
        .collect();

    let n = dataset.n_rows();
    let k = trees.len() as f64;

    (0..n)
        .map(|i| {
            let mut combined = MeanVector::new();
            for tree_means in &all_means {
                for (col, vec) in &tree_means[i] {
                    combined.entry(col.clone())
                        .and_modify(|e: &mut Vec<f64>| {
                            for (j, v) in vec.iter().enumerate() {
                                e[j] += v / k;
                            }
                        })
                        .or_insert_with(|| vec.iter().map(|v| v / k).collect());
                }
            }
            combined
        })
        .collect()
}
```

### 7.2 Distribution merging (for full distribution queries)

```rust
fn ensemble_distributions(
    distributions: &[PiecewiseConstantDistribution],
) -> PiecewiseConstantDistribution {
    let mut all_cells = Vec::new();
    let k = distributions.len() as f64;
    for dist in distributions {
        for cell in &dist.cells {
            all_cells.push(ConditionedCell {
                target_rules: cell.target_rules.clone(),
                mass: cell.mass / k,
            });
        }
    }
    PiecewiseConstantDistribution { cells: all_cells }
}
```

This preserves the full piecewise structure and supports pdf/mass queries on the ensemble.

---

## 8. Implementation Order

| Step | File                                | Description                                                     | Dependencies                    |
| ---- | ----------------------------------- | --------------------------------------------------------------- | ------------------------------- |
| 1    | `predict/mod.rs`                    | Module declaration, re-exports                                  | —                               |
| 2    | `predict/conditioned_cell.rs`       | `ConditionedCell` struct + methods                              | `cell.rs`, `rule.rs`, `tree.rs` |
| 3    | `predict/piecewise_distribution.rs` | `PiecewiseConstantDistribution` + `MeanVector`                  | `conditioned_cell.rs`           |
| 4    | `tree.rs` (extension)               | `predict_distributions`, `predict_mean_vectors`, `predict_mean` | `predict/`                      |
| 5    | `mod.rs` (update)                   | Add `pub mod predict;` + re-exports                             | all above                       |

### Step-by-step details:

**Step 1 — `predict/mod.rs`:**
- `pub mod conditioned_cell;`
- `pub mod piecewise_distribution;`
- Re-export `ConditionedCell`, `PiecewiseConstantDistribution`, `MeanVector`.

**Step 2 — `predict/conditioned_cell.rs`:**
- `ConditionedCell` struct with `target_rules: HashMap<String, Box<dyn DynRule>>` and `mass: f64`.
- `from_fitted_node(node: &FittedNode) -> Self`.
- `target_volume() -> f64`.
- `conditional_density() -> f64`.
- `mean_map() -> HashMap<String, Vec<f64>>`.
- `contains_target_row(row_idx, columns) -> bool` — evaluate target rules against data columns.

**Step 3 — `predict/piecewise_distribution.rs`:**
- `MeanVector` type alias.
- `PiecewiseConstantDistribution` with `cells: Vec<ConditionedCell>`.
- `mean_vector()`, `total_mass()`, `n_cells()`.
- `ensemble(distributions: &[&Self]) -> Self`.
- `pdf_segments()` for continuous targets.
- `category_probabilities()` for categorical targets.

**Step 4 — `tree.rs` extension:**
- `Tree::predict_distributions(&self, dataset) -> Vec<PiecewiseConstantDistribution>`.
  Implementation: call `predict_leaves`, group by leaf, build `ConditionedCell` per leaf, wrap.
  Optimization: build `ConditionedCell` once per unique leaf, clone for each row.
- `Tree::predict_mean_vectors(&self, dataset) -> Vec<MeanVector>`.
- `Tree::predict_mean(&self, dataset) -> DataFrame`.
  Needs root cell's target rules to extract domain names for categorical decoding.

**Step 5 — `mod.rs`:**
- Add `pub mod predict;` to the v2 module.
- Re-export `PiecewiseConstantDistribution`, `MeanVector`, `ConditionedCell`.

---

## 9. Performance Considerations

1. **Leaf deduplication.** Many rows map to the same leaf. Build `ConditionedCell` once per
   unique leaf index and reuse via `clone()` or `Arc`.

2. **Avoid re-traversal.** `predict_distributions` should reuse `predict_leaves` rather than
   traversing the tree twice.

3. **Lazy mean computation.** `mean_vector()` iterates cells and calls `DynRule::mean()` which
   allocates. For batch prediction, prefer `predict_mean_vectors` which can amortize allocations.

4. **Parallelism.** `predict_leaves` is already row-independent. The distribution construction
   and mean computation are also trivially parallelizable with `rayon` if needed.

---

## 10. Design Principles

1. **v1 parity.** The v2 `PiecewiseConstantDistribution` covers all v1 capabilities (mean, pdf,
   mass, ensemble) but with a cleaner API that leverages `DynRule` instead of `dyn Any` downcasting.

2. **Ensemble-friendly.** Fixed-shape `MeanVector` makes averaging across trees trivial —
   no alignment or domain negotiation required.

3. **Dtype-agnostic.** All dtype-specific logic lives in `DynRule::mean()` and `DynRule::contains()`.
   The prediction module never matches on concrete rule types except for DataFrame decoding
   in `predict_mean`.

4. **No stored schema.** Unlike v1 which required a separate `schema` object, v2 derives all
   metadata from the tree's root cell (domain sizes, target column names, dtypes via `DynRule::as_any()`).

5. **Composable.** `ConditionedCell` and `PiecewiseConstantDistribution` are standalone types
   usable outside the tree context (e.g., for custom post-processing or visualization).