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
````

---

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

```