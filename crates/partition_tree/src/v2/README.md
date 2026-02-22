# Partition Tree v2

This file is meant to be a refactoring proposal for Partition Tree
The goal is to make the code more modular and easier to extend,
keeping clear responsabilities for each component.

## Main components

Overall flow:

1. The user initializes Tree with hyperparameters.
2. The user creates a `DatasetView` with their dataset. This dataset view stores sorted indices for each column, which will be used for efficient split searching.
3. The user calls `fit` with a dataset. Target columns are prefixed with `target__` to distinguish them from feature columns.
4. We create a TreeBuilder instance, and call `build_tree` passing the DatasetView, gain objective, and split restrictions.
5. The TreeBuilder uses a SplitSearcher to find the best split for the current node, which in turn uses ColumnSplitSearchers for each column type to find the best split point for that column.
6. The TreeBuilder creates child nodes based on the best split, and recursively builds the tree until a stopping criterion is met (e.g., max depth, min samples, etc.), in best-first fashion.

Important aspects here are that split point search and rule volumes are different for each dtype. For example, for numerical columns we can presort the column and efficiently compute the gain for each potential split point, while for categorical columns we might need to use a different approach (e.g., grouping categories together). The code should allow easily extending to new dtypes (based on Polars datatypes).

### LossFunc

A contract for computing the gain of splits. This will allow us to easily add new gain objectives in the future.
Given:
* w_xy: xy-samples weights
* w_x: x-samples weights
* w_y: y-samples weights
* volume: the volume of the cell d

It returns the objective value of the split (the smaller the better).

*Conditional Log loss*:

The log loss computes:

$$
- \log\left(\frac{w_{xy}}{w_x \cdot \text{volume}}\right)
$$

*Balanced Log Loss*:

$$
- \log\left(\frac{w_{xy}}{w_x \cdot w_y}\right)
$$

### Rule

Represents a partition constraint for a single column.
Rules are responsible for:

* evaluating whether values belong to the partition
* computing partition volume and relative volume
* providing a numeric mean/embedding representation
* splitting themselves into left/right child rules

Current rule contract (see `src/rules.rs`) is generic over value type `T`:

* `_evaluate_some(&T) -> bool`: membership for concrete values
* `_evaluate_none() -> bool`: whether missing values are accepted
* `evaluate(&[Option<T>]) -> Vec<bool>`: vectorized evaluation
* `volume() -> f64`: raw geometric/cardinality volume
* `relative_volume() -> f64`: normalized by domain
* `phi_volume() -> f64`: transformed volume via `phi_length` (dtype-specific)
* `mean() -> Vec<f64>`: vector representation used in downstream decoding
* `split(point, none_to_left) -> (left, right)`: creates child rules
* `inverse_one_hot(&Vec<f64>) -> T`: decode vector back to value space

Implemented rule families:

* `ContinuousInterval` (for `f64`):
    * interval `[low, high]` with open/closed bounds
    * tracks full domain `(min, max)` and `accept_none`
    * split at threshold creates left `(low, point)` and right `[point, high)` style children
    * volume is interval length; relative volume is interval/domain ratio

* `BelongsToGeneric<T>` (categorical membership):
    * stores selected category set `values` and complete ordered `domain`
    * supports physical category types (`usize`, integers, bool, String)
    * split by subset: left receives subset, right receives current values not in subset
    * volume is selected category count; relative volume is count/domain size

Type-erased enum used by tree/cell APIs:

* `RuleType::Continuous(ContinuousInterval)`
* `RuleType::BelongsTo(BelongsTo)` (`BelongsTo` = `BelongsToGeneric<usize>`)

Notes:

* Both rule families support `accept_none` to route missing values.
* `BitAnd` is implemented for both families to intersect constraints when combining cells.

### Cell

Represents the region of feature space covered by a node.

* `HashMap<String, RuleType>`

where:

* key = column name
* value = the rule restricting that column for the current node

Semantics:

* A sample belongs to a `Cell` iff it satisfies **all** rules in the map.
* Missing columns in the map are unconstrained (implicitly full-domain for that column).
* Child cells are created by cloning parent rules and replacing only the split column rule.

Suggested responsibilities:

* `contains(row) -> bool`: evaluate conjunction of per-column rules.
* `split(col, point_or_subset, none_to_left) -> (left_cell, right_cell)`: delegate split to the selected column rule and update the map in each child.
* `volume() -> f64`: aggregate per-rule volumes (typically multiplicative under independence assumptions).
* `relative_volume() -> f64`: aggregate normalized per-rule relative volumes.

This representation keeps the structure simple, explicit, and dtype-extensible via `RuleType`.

### Node

Contains:
* 1 Cell
* w_xy, w_x, w_y: the weights of the samples in the cell
* depth: depth of the node in the tree
* reference to child nodes (if any)


Allows computing the densities:

- conditional density: w_xy / (w_x * volume)
- balancing weight: w_y / (w_x * w_y)



### Split Restrictions

Struct that validate if, given the information of a candidate split point, it is valid or not.
Types of split restrictions:

* min_samples_xy: minimum number of xy-samples weights in the split
* min_samples_x: minimum number of x-samples weights in the split
* min_samples_y: minimum number of y-samples weights in the split
* min_gain: minimum gain of the split
* min_volume: minimum volume of the split
* max_depth: maximum depth of the tree
* min_samples_split: minimum number of samples in the split (n_xy of the parent node)


### Column Split Searcher

Represents the contract for searching for the best split for a given column type. 
This will allow us to easily add new split search algorithms for different variable types in the future.

#### Continuous columns (node-local presorted scan, supports X / Y splits)

Below is a concrete template for a **continuous** `ColumnSplitSearcher` that respects Partition Tree bookkeeping:
each node carries **three** index sets (and their node-local presorted orders for each column):

- `idx_x`  : indices used to estimate `w_x`   (membership in `A_X`)
- `idx_y`  : indices used to estimate `w_y`   (membership in `A_Y`)
- `idx_xy` : indices used to estimate `w_xy`  (membership in `A_X × A_Y`)

For each column `col`, the node stores presorted lists restricted to these sets:

- `sorted_x[col]`   : `idx_x`  sorted by `col` values
- `sorted_y[col]`   : `idx_y`  sorted by `col` values
- `sorted_xy[col]`  : `idx_xy` sorted by `col` values

The scan enumerates candidate thresholds (midpoints between consecutive *distinct* values)
and updates left-mass using prefix sums + moving pointers on the other presorted lists.

> Conventions:
> - `wx[i], wy[i], wxy[i]` are per-sample weights in their respective measures.
> - `cell.split_volumes(col, t, none_to_left, split_kind)` returns `(vol_left, vol_right)`.
>   For **X-split**, volumes are typically unchanged (same `A_Y`), so it can return `(vol_parent, vol_parent)`.
>   For **Y-split**, volumes change (split `A_Y`), so it returns the actual child volumes.
> - `loss.cell_loss(w_xy, w_x, w_y, vol)` returns the (weighted) node loss contribution.
>   Then `gain = parent_loss - (left_loss + right_loss)`.

---

##### 1) Best split search for one continuous column

```pseudo
fn search_best_split_point_continuous(
    node,
    cell,
    col,                 // ColumnView
    split_kind,          // XSplit(col is feature) OR YSplit(col is target)
    wx, wy, wxy,         // global per-sample weights
    loss_func,
    restrictions,
    none_to_left         // routing of nulls for this column
) -> Option<SplitResult> {

    // --- 0) Fetch node-local presorted index lists for this column ---
    sx   = node.sorted_x[col.name]    // Vec<idx> sorted by col over idx_x
    sy   = node.sorted_y[col.name]    // Vec<idx> sorted by col over idx_y
    sxy  = node.sorted_xy[col.name]   // Vec<idx> sorted by col over idx_xy

    // --- 1) Separate NULLs (so scans run on "some" values only) ---
    // We assume presorted lists place NULLs at the end (or we can filter).
    (sx_some,  wx_null)  = split_nulls_and_sum_weights(sx,  col, wx)
    (sy_some,  wy_null)  = split_nulls_and_sum_weights(sy,  col, wy)
    (sxy_some, wxy_null) = split_nulls_and_sum_weights(sxy, col, wxy)

    // Total weights at parent
    w_x_parent  = sum_over_indices(sx_some,  wx)  + wx_null
    w_y_parent  = sum_over_indices(sy_some,  wy)  + wy_null
    w_xy_parent = sum_over_indices(sxy_some, wxy) + wxy_null

    // Parent volume and loss
    vol_parent = cell.volume()  // or cell.phi_volume()
    parent_loss = loss_func.cell_loss(w_xy_parent, w_x_parent, w_y_parent, vol_parent)

    best = None
    best_gain = -INF

    // --- 2) Choose candidate list depending on split kind ---
    candidates = (split_kind == XSplit) ? sx_some : sy_some
    if candidates.len < 2:
        return None

    // --- 3) Prefix sums on candidate list ---
    if split_kind == XSplit:
        prefix_w = cumsum([ wx[i] for i in candidates ])
    else:
        prefix_w = cumsum([ wy[i] for i in candidates ])

    // --- 4) Prefix sums on sxy_some for O(1) left extraction after pointer moves ---
    prefix_wxy = cumsum([ wxy[i] for i in sxy_some ])

    // pointer into sxy_some
    p_xy = 0

    // --- 5) Iterate candidate thresholds ---
    for k in 0 .. candidates.len - 2:
        i0 = candidates[k]
        i1 = candidates[k+1]
        v0 = col.value(i0)
        v1 = col.value(i1)
        if v0 == v1:
            continue

        t = 0.5 * (v0 + v1)

        // ---- 5.1) Candidate-measure left weights ----
        w_candidate_left = prefix_w[k]
        if split_kind == XSplit:
            w_x_left  = w_candidate_left + (none_to_left ? wx_null : 0.0)
            w_x_right = w_x_parent - w_x_left
        else:
            w_y_left  = w_candidate_left + (none_to_left ? wy_null : 0.0)
            w_y_right = w_y_parent - w_y_left

        // ---- 5.2) Move pointer on sxy_some to compute w_xy_left ----
        while p_xy < sxy_some.len and col.value(sxy_some[p_xy]) <= t:
            p_xy += 1
        w_xy_left_some = (p_xy > 0) ? prefix_wxy[p_xy - 1] : 0.0
        w_xy_left  = w_xy_left_some + (none_to_left ? wxy_null : 0.0)
        w_xy_right = w_xy_parent - w_xy_left

        // ---- 5.3) Fill in unchanged measures ----
        if split_kind == XSplit:
            w_y_left = w_y_parent; w_y_right = w_y_parent
        else:
            w_x_left = w_x_parent; w_x_right = w_x_parent

        // ---- 5.4) Child volumes ----
        (vol_left, vol_right) = cell.split_volumes(col.name, t, none_to_left, split_kind)

        // ---- 5.5) Validate split restrictions ----
        if not restrictions.is_valid_children(
            w_xy_left,  w_x_left,  w_y_left,  vol_left,
            w_xy_right, w_x_right, w_y_right, vol_right,
            node.depth
        ):
            continue

        // ---- 5.6) Gain ----
        left_loss  = loss_func.cell_loss(w_xy_left,  w_x_left,  w_y_left,  vol_left)
        right_loss = loss_func.cell_loss(w_xy_right, w_x_right, w_y_right, vol_right)
        gain = parent_loss - (left_loss + right_loss)
        if gain < restrictions.min_gain:
            continue

        // ---- 5.7) Best ----
        if gain > best_gain:
            best_gain = gain
            best = SplitResult {
                col_name: col.name,
                split_kind: split_kind,
                split_point: t,
                none_to_left: none_to_left,
                gain: gain,

                left_stats:  (w_xy_left,  w_x_left,  w_y_left,  vol_left),
                right_stats: (w_xy_right, w_x_right, w_y_right, vol_right),

                // Store split positions to accelerate propagation for THIS column
                // If candidates == sx_some, then k determines boundary in sorted_x[col]
                // If candidates == sy_some, then k determines boundary in sorted_y[col]
                k_candidate: k,
                p_xy: p_xy,
            }
        end if
    end for

    return best
}
```

##### 2) Propagating `sorted_x`, `sorted_y`, `sorted_xy` to children

After `TreeBuilder` chooses the best `SplitResult` for node `A`,
it must create child nodes `A_L`, `A_R`, including their node-local presorted lists for **all columns**.

The key idea is:

* You never rebuild child sorted lists from scratch.
* For each column `c`, you obtain `child.sorted_*[c]` by a **stable partition** of `parent.sorted_*[c]`.
* The predicate used in this partition depends only on the chosen split rule `(split_col, split_point/subset, none_to_left)`.

This keeps propagation **linear in node size per column**.

```pseudo
fn propagate_sorted_lists_to_children(
    parent_node,
    parent_cell,
    split_result,         // chosen split (col, kind, point/subset, none_to_left)
    dataset_view          // gives access to column values
) -> (left_node_sorted, right_node_sorted) {

    split_col = split_result.col_name
    split_kind = split_result.split_kind
    t = split_result.split_point
    none_to_left = split_result.none_to_left

    // Access split column values once
    split_column_view = dataset_view.column(split_col)

    // Build an O(1) routing oracle for indices in THIS NODE:
    // go_left(i) := whether sample i is sent to left child by the chosen rule.
    // For continuous:
    //   if value is NULL => none_to_left
    //   else value <= t  => left
    // For categorical: value ∈ subset, etc.
    fn go_left(i):
        v = split_column_view.get(i)
        if v is NULL: return none_to_left
        else: return (v <= t)

    // Initialize child maps
    left_sorted_x  = HashMap<col, Vec<idx>>()
    right_sorted_x = HashMap<col, Vec<idx>>()
    left_sorted_y  = HashMap<col, Vec<idx>>()
    right_sorted_y = HashMap<col, Vec<idx>>()
    left_sorted_xy  = HashMap<col, Vec<idx>>()
    right_sorted_xy = HashMap<col, Vec<idx>>()

    // For every column c in the schema, propagate all three families.
    for c in dataset_view.columns():
        // --- propagate sorted_x[c] ---
        parent_list = parent_node.sorted_x[c]
        (l, r) = stable_partition(parent_list, go_left)
        left_sorted_x[c] = l
        right_sorted_x[c] = r

        // --- propagate sorted_y[c] ---
        parent_list = parent_node.sorted_y[c]
        (l, r) = stable_partition(parent_list, go_left)
        left_sorted_y[c] = l
        right_sorted_y[c] = r

        // --- propagate sorted_xy[c] ---
        parent_list = parent_node.sorted_xy[c]
        (l, r) = stable_partition(parent_list, go_left)
        left_sorted_xy[c] = l
        right_sorted_xy[c] = r
    end for

    return (
      (left_sorted_x, left_sorted_y, left_sorted_xy),
      (right_sorted_x, right_sorted_y, right_sorted_xy)
    )
}

fn stable_partition(list, predicate) -> (left, right) {
    left = []
    right = []
    for i in list:
        if predicate(i): left.push(i) else right.push(i)
    return (left, right)
}
```

**Notes (important for correctness and efficiency):**

1. **X-split vs Y-split semantics**

   * In the standard product-cell semantics, an **X-split** only refines `A_X`; an **Y-split** only refines `A_Y`.
   * However, the *routing oracle* `go_left(i)` is still defined on full samples. It correctly partitions all three index sets:

     * `idx_x` moves under X-splits (by definition).
     * `idx_y` may remain unchanged under X-splits if `idx_y` depends only on `A_Y`. If so, `stable_partition` is still correct but redundant.
       An optimization is to skip propagating `sorted_y` on X-splits and just alias parent lists.
     * `idx_xy` always moves under both X and Y splits.

2. **Optimization for the split column**

   * For the chosen split column, you already have split boundaries (`k_candidate`, `p_xy`) computed during scanning.
   * You can split `parent.sorted_x[split_col]`, `parent.sorted_y[split_col]`, `parent.sorted_xy[split_col]` by slicing at those boundaries
     instead of re-checking `go_left` for each element.
   * For other columns, you still need stable_partition, because their sort order is not aligned with the split coordinate.

3. **Complexity**

   * Propagation for one node split costs:

     * (O(n_x \cdot d)) for `sorted_x`, (O(n_y \cdot d)) for `sorted_y`, (O(n_{xy} \cdot d)) for `sorted_xy`,
       where (d) is the number of columns you maintain sorted lists for.
   * With balanced trees and global presorting, this yields the typical (O(d,N\log N)) behavior (up to constant factors from carrying three families).

4. **Correctness with NULLs**

   * The routing oracle must implement the same NULL rule (`none_to_left`) used by the split rule.
   * Presorted lists should either:

     * place NULLs at the end, or
     * allow `col.get(i)` to return NULL and route accordingly.

This propagation step is invoked by `TreeBuilder` immediately after selecting the best split, before pushing child nodes into the best-first queue.

#### Categorical columns (sorted-prefix subset search; supports X / Y splits)

For categorical coordinates, enumerating all subset splits \(S \subset \Sigma\) is exponential (\(2^{|\Sigma|}-2\)).
Partition Trees use a **sort-and-scan** procedure: compute per-category statistics inside the current node,
sort categories by a score \(r_c\), and scan the \(|\Sigma|-1\) prefix splits. This attains the best subset split
for the conditional log-loss gain (and its weighted analogue) under the same “optimal prefix” property used in the paper.

We distinguish two cases (both produce a prefix-optimal scan after sorting by \(r_c\)):

- **Categorical X-split** (split a feature \(x_\ell\)): \(A_Y\) is unchanged, so \(\mu_Y(A_Y)\) is constant across children.
  Define for each category \(c\in\Sigma\) (restricted to samples in the node):
  - \(a_c := w_{xy}(A \cap \{x_\ell=c\})\)  (xy-mass in category)
  - \(b_c := w_{x}(A \cap \{x_\ell=c\})\)   (x-mass in category)
  - \(r_c := a_c/b_c\) for \(b_c>0\)

- **Categorical Y-split** (split a target \(y_\ell\)): \(A_X\) is unchanged. The denominator uses \(\mu_Y(A_Y)\),
  which varies by the selected subset. Define:
  - \(a_c := w_{xy}(A \cap \{y_\ell=c\})\)
  - \(b_c := \mu_Y(A_Y \cap \{y_\ell=c\})\) (category “volume”; for counting measure, typically 1 per allowed category)
  - \(r_c := a_c/b_c\) for \(b_c>0\)

For **missing values** (None), treat them as a dedicated “category” bucket if `accept_none` is true.
To support `none_to_left`, you can either:
- force the None-bucket into the chosen side after selecting a subset, or
- include None in the sorted list with its own \((a_{\text{None}}, b_{\text{None}}, r_{\text{None}})\) and handle routing explicitly.

---

##### 1) Best subset split search for one categorical column (conditional log-loss-compatible)

```pseudo
fn search_best_split_subset_categorical(
    node,
    cell,
    col,                 // ColumnView categorical with domain Σ
    split_kind,          // XSplit OR YSplit
    wx, wy, wxy,         // per-sample weights (global arrays)
    loss_func,
    restrictions,
    none_to_left
) -> Option<SplitResult> {

    Σ = cell.rule(col.name).domain_categories()   // ordered domain used by the rule
    if Σ.len < 2: return None

    // --- 0) Accumulate per-category stats inside the node ---
    // a_c always from idx_xy
    map_a = HashMap<cat, f64>(default 0)
    for i in node.idx_xy:
        v = col.get(i)                   // Option<cat>
        if v is None:
            if cell.rule(col.name).accept_none:
                map_a[None] += wxy[i]
        else:
            map_a[v] += wxy[i]

    map_b = HashMap<cat, f64>(default 0)
    if split_kind == XSplit:
        // b_c from idx_x
        for i in node.idx_x:
            v = col.get(i)
            if v is None:
                if cell.rule(col.name).accept_none:
                    map_b[None] += wx[i]
            else:
                map_b[v] += wx[i]
    else: // YSplit
        // b_c from μY mass implied by the current Y-rule
        // (for counting measure and BelongsTo, this is typically 1 per allowed category)
        for c in Σ:
            map_b[c] = cell.rule(col.name).muY_mass_of_category(c)
        if cell.rule(col.name).accept_none:
            map_b[None] = cell.rule(col.name).muY_mass_of_none()

    // Active categories are those with b_c > 0
    cats = []   // entries: (cat, a_c, b_c, r_c)
    for c in Σ plus maybe None:
        b = map_b[c]
        if b <= 0: continue
        a = map_a[c]            // may be 0
        r = a / b
        cats.push((c, a, b, r))

    if cats.len < 2: return None

    // --- 1) Sort categories by r_c (ascending) ---
    sort cats by r ascending

    // --- 2) Prefix sums for a and b over the sorted categories ---
    A_pref = cumsum([a for (_,a,_,_) in cats])
    B_pref = cumsum([b for (_,_,b,_) in cats])
    A_total = A_pref[last]
    B_total = B_pref[last]

    // Parent stats for gain
    w_xy_parent = node.w_xy
    w_x_parent  = node.w_x
    w_y_parent  = node.w_y
    vol_parent  = cell.volume()
    parent_loss = loss_func.cell_loss(w_xy_parent, w_x_parent, w_y_parent, vol_parent)

    best = None
    best_gain = -INF

    // --- 3) Scan prefix splits S_t = {cats[0..t]} for t = 0..m-2 ---
    for t in 0 .. cats.len - 2:
        A_left = A_pref[t]
        B_left = B_pref[t]
        A_right = A_total - A_left
        B_right = B_total - B_left

        // Build subset (prefix) representation for the rule:
        subset_left = { cats[0].cat, ..., cats[t].cat }

        // Optionally enforce none_to_left by moving None in/out of subset_left here.
        if cell.rule(col.name).accept_none:
            if none_to_left: subset_left.insert(None) else subset_left.remove(None)

        // --- 3.1) Child weights/volumes ---
        if split_kind == XSplit:
            // X-split: A_Y unchanged
            w_xy_left  = A_left
            w_xy_right = w_xy_parent - w_xy_left

            w_x_left   = B_left
            w_x_right  = w_x_parent - w_x_left

            w_y_left   = w_y_parent
            w_y_right  = w_y_parent

            vol_left   = vol_parent
            vol_right  = vol_parent

        else: // YSplit
            // Y-split: A_X unchanged, μY changes by subset
            w_xy_left  = A_left
            w_xy_right = w_xy_parent - w_xy_left

            w_x_left   = w_x_parent
            w_x_right  = w_x_parent

            // If objective needs w_y (e.g., BalancedLogLoss), compute via idx_y aggregation.
            // Otherwise (conditional log-loss) w_y is bookkeeping.
            (w_y_left, w_y_right) = compute_wy_split_if_needed(node.idx_y, col, wy, subset_left)

            vol_left   = B_left    // B_left is μY(A_Y_left)
            vol_right  = B_right

        // --- 3.2) Restrictions (both children) ---
        if not restrictions.is_valid_children(
            w_xy_left,  w_x_left,  w_y_left,  vol_left,
            w_xy_right, w_x_right, w_y_right, vol_right,
            node.depth
        ):
            continue

        // --- 3.3) Gain ---
        left_loss  = loss_func.cell_loss(w_xy_left,  w_x_left,  w_y_left,  vol_left)
        right_loss = loss_func.cell_loss(w_xy_right, w_x_right, w_y_right, vol_right)
        gain = parent_loss - (left_loss + right_loss)
        if gain < restrictions.min_gain: continue

        if gain > best_gain:
            best_gain = gain
            best = SplitResult {
                col_name: col.name,
                split_kind: split_kind,
                subset_left: subset_left,
                none_to_left: none_to_left,
                gain: gain,
                left_stats:  (w_xy_left,  w_x_left,  w_y_left,  vol_left),
                right_stats: (w_xy_right, w_x_right, w_y_right, vol_right),
            }
        end if
    end for

    return best
}
```

2) Complexity
	•	Accumulating ((a_c)): (O(|\text{idx}_{xy}(A)|))
	•	Accumulating ((b_c)): (O(|\text{idx}_{x}(A)|)) for X-splits, or (O(|\Sigma|)) for Y-splits (volume lookups)
	•	Sorting categories: (O(|\Sigma|\log|\Sigma|))
	•	Scanning prefix splits: (O(|\Sigma|))

This avoids enumerating all (2^{|\Sigma|}-2) subsets while still finding the best categorical split under the objective.



### Split Searcher

Given a datasetview, a gain objective, and a set of split restrictions, it finds the best split point-column-gain combination for the current node. It uses the column split searcher for each column to find the best split point for that column, and then compares the gains to find the best overall split.

### Tree builder

Given a datasetview, a gain objective, and a set of split restrictions, it builds the partition tree by recursively finding the best split using the Split Searcher and creating child nodes until a stopping criterion is met (e.g., max depth, min samples, etc.). The split is done in best-first
fashion, meaning that we always split the node with the highest gain first, which allows us to build a more balanced tree and potentially reduce the depth of the tree.

The `find_best_split` returns a `SplitResult` struct that contains the best split point, the column it belongs to, and the gain of the split. 

Tree Builder also propagates the split results to child nodes, updating their statistics and continuing the recursive partitioning process until stopping criteria are met.


### DTypePlugin

DType-plugin avoids if-else clauses 
```rust
pub trait DTypePlugin: Send + Sync {
    fn logical_dtype(&self) -> LogicalDType;
    fn default_rule(&self, column: &ColumnView, cfg: &PartitionDefaults) -> DynPartition;
    fn split_searcher(&self) -> &dyn SplitSearcher;
    // Other dtype-specific methods
}
```

A `DTypeRegistry` maps `LogicalDType -> DTypePlugin`.

This replaces large `match` blocks and lets new dtypes be added by plugin registration.


### DatasetView & ColumnView

Layer of abstraction over the dataset and columns, with the core based on Polars. Provides an interface for presorting columns.
ColumnView provides methods for accessing column data, presorted indices, and other column-specific operations needed for split searching.



## Remarks

- The design should follow best code design practices
- We will not consider exploration splits, as in the first implementation
- Always add unit tests at the right abstraction level. Avoid testing implementation details, but test the contracts and expected behavior of each component.

### Best test practices

Add tests for this Rust library/module following these practices:
	•	Prioritize public API behavior and regression tests for known bugs.
	•	Prefer fast unit tests first; use integration tests only for cross-module/public API flows.
	•	Keep tests deterministic (no time/network/randomness unless controlled).
	•	Test edge cases, error paths, and boundary conditions.
	•	Avoid over-testing private implementation details that make refactors brittle.
	•	Use clear test names and assertions so failures are easy to diagnose.
	•	Reuse shared setup/helpers instead of duplicating fixtures.

Use the test-case crate for modular, parameterized tests:
	•	Replace repeated “same test, different inputs” functions with #[test_case(...)].
	•	Group related cases in one test function with descriptive case labels.
	•	Keep each case focused on one behavior/invariant.
	•	Use normal helper functions/builders for shared setup; use test-case mainly for input/output matrices.

Goal: produce a test suite that is readable, fast, maintainable, and easy to extend as the library grows.