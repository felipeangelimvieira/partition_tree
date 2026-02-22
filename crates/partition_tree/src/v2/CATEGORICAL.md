#### Categorical columns (sorted-prefix subset search; supports X / Y splits)

For categorical coordinates, enumerating all subset splits \(S \subset \Sigma\) costs \(2^{|\Sigma|}\).
Partition Trees instead use a **sort-and-scan** procedure: compute per-category statistics in the node,
sort categories by a score \(r_c\), and scan the \(|\Sigma|-1\) prefix splits. This attains the best subset split
for the **conditional log-loss** objective. :contentReference[oaicite:0]{index=0}

We distinguish two cases:

- **Categorical X-split** (split a feature): \(A_Y\) is unchanged, so \(\mu_Y(A_Y)\) cancels in the gain. :contentReference[oaicite:1]{index=1}  
  Define for each category \(c\in\Sigma\):
  - \(a_c := w_{xy}(A \cap \{x_\ell=c\})\) (xy-mass in category)
  - \(b_c := w_{x}(A \cap \{x_\ell=c\})\)  (x-mass in category)
  - \(r_c := a_c/b_c\) for \(b_c>0\)

- **Categorical Y-split** (split a target category): \(A_X\) is unchanged, so \(w_x(A)\) cancels; \(\mu_Y\) terms vary across children. :contentReference[oaicite:2]{index=2}  
  Define:
  - \(a_c := w_{xy}(A \cap \{y_\ell=c\})\)
  - \(b_c := \mu_Y(A_Y \cap \{y_\ell=c\})\)  (category “volume”; for counting measure this is typically 1 for allowed categories, 0 otherwise)
  - \(r_c := a_c/b_c\) for \(b_c>0\)

For both cases, the best subset split can be chosen as a **prefix** after sorting by \(r_c\). :contentReference[oaicite:3]{index=3}

---

##### 1) Best subset split search for one categorical column (conditional log-loss)

```pseudo
fn search_best_split_subset_categorical(
    node,
    cell,
    col,                  // ColumnView (categorical) with domain Σ = [c1..cK]
    split_kind,           // XSplit OR YSplit
    wx, wy, wxy,          // per-sample weights
    loss_func,            // must be ConditionalLogLoss-compatible for this routine
    restrictions,
    none_to_left
) -> Option<SplitResult> {

    Σ = col.domain_categories()             // ordered list of categories
    K = Σ.len
    if K < 2: return InvalidSplit   // no split possible

    // ---- 0) Compute per-category (a_c, b_c) inside this node ----
    // a_c always comes from idx_xy. b_c depends on split kind:
    //   XSplit: b_c = sum wx over idx_x with value==c
    //   YSplit: b_c = μY(A_Y ∩ {col=c}) from current cell/rule
    //
    // Also handle NULLs via a special bucket "None" if accept_none is true.
    map_a = HashMap<cat, f64>(default 0)
    map_b = HashMap<cat, f64>(default 0)

    // 0.1) a_c from idx_xy
    for i in node.idx_xy:
        v = col.get(i)                  // Option<Category>
        if v is None:
            if cell.rule(col).accept_none:
                map_a[None] += wxy[i]
            else:
                continue
        else:
            map_a[v] += wxy[i]

    // 0.2) b_c
    if split_kind == XSplit:
        for i in node.idx_x:
            v = col.get(i)
            if v is None:
                if cell.rule(col).accept_none:
                    map_b[None] += wx[i]
                else:
                    continue
            else:
                map_b[v] += wx[i]
    else: // YSplit
        // b_c is volume contribution from the current Y-rule
        // (for counting measure and a BelongsTo rule, b_c is typically 1 for allowed categories, 0 otherwise)
        for c in Σ:
            map_b[c] = cell.rule(col).muY_mass_of_category(c)
        if cell.rule(col).accept_none:
            map_b[None] = cell.rule(col).muY_mass_of_none()

    // ---- 1) Build list of active categories with b_c > 0 ----
    cats = []
    for cat in Σ plus maybe None:
        b = map_b[cat]
        if b > 0:
            a = map_a[cat]           // possibly 0
            r = a / b
            cats.push( (cat, a, b, r) )

    if cats.len < 2:
        return None

    // ---- 2) Sort categories by r_c (ascending) ----
    sort cats by r ascending

    // ---- 3) Prefix sums of a and b in sorted order ----
    A_pref[t] = sum_{j<=t} a_j
    B_pref[t] = sum_{j<=t} b_j
    A_pref = cumsum([a for (_,a,_,_) in cats])
    B_pref = cumsum([b for (_,_,b,_) in cats])

    A_total = A_pref[last]
    B_total = B_pref[last]

    // Parent masses (needed for loss/gain + restrictions)
    w_xy_parent = node.w_xy
    w_x_parent  = node.w_x
    w_y_parent  = node.w_y
    vol_parent  = cell.volume()
    parent_loss = loss_func.cell_loss(w_xy_parent, w_x_parent, w_y_parent, vol_parent)

    best = None
    best_gain = -INF

    // ---- 4) Scan prefix splits S_t = {cats[0..t]} for t=0..m-2 ----
    for t in 0 .. cats.len - 2:
        // subset S is prefix categories
        A_left = A_pref[t]
        B_left = B_pref[t]
        A_right = A_total - A_left
        B_right = B_total - B_left

        // Handle none_to_left: if None exists as a category bucket, it is already positioned in sorted order.
        // If you want to force None to left/right regardless of r, treat None separately and merge into the chosen side here.

        // ---- 4.1) Derive child weights/volumes depending on split kind ----
        if split_kind == XSplit:
            // X-split: A_Y unchanged
            w_xy_left = A_left
            w_xy_right = w_xy_parent - w_xy_left

            w_x_left = B_left
            w_x_right = w_x_parent - w_x_left

            w_y_left = w_y_parent
            w_y_right = w_y_parent

            vol_left = vol_parent
            vol_right = vol_parent

        else: // YSplit
            // Y-split: A_X unchanged; μY differs across children
            w_xy_left = A_left
            w_xy_right = w_xy_parent - w_xy_left

            w_x_left = w_x_parent
            w_x_right = w_x_parent

            // If your objective needs w_y (e.g., BalancedLogLoss), compute it from idx_y analogously:
            // w_y_left = sum wy over idx_y with category in S; w_y_right = w_y_parent - w_y_left.
            // For conditional log-loss, w_y may be unused; keep as bookkeeping:
            w_y_left = compute_wy_left_from_prefix_if_needed(node.idx_y, col, wy, S_t, none_to_left)
            w_y_right = w_y_parent - w_y_left

            vol_left  = B_left      // because b_c := μY(category), so B_left is μY(A_Y_left)
            vol_right = B_right

        // ---- 4.2) Validate feasibility / restrictions (both children) ----
        if not restrictions.is_valid_children(
            w_xy_left,  w_x_left,  w_y_left,  vol_left,
            w_xy_right, w_x_right, w_y_right, vol_right,
            node.depth
        ):
            continue

        // ---- 4.3) Gain ----
        left_loss  = loss_func.cell_loss(w_xy_left,  w_x_left,  w_y_left,  vol_left)
        right_loss = loss_func.cell_loss(w_xy_right, w_x_right, w_y_right, vol_right)
        gain = parent_loss - (left_loss + right_loss)
        if gain < restrictions.min_gain:
            continue

        // ---- 4.4) Track best ----
        if gain > best_gain:
            best_gain = gain
            subset_left = { cats[0].cat, ..., cats[t].cat }   // prefix subset
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
````

---

##### 2) Complexity and guarantees

* Computing ((a_c)) and ((b_c)) is linear in node sample counts (typically (O(n_{xy}(A) + n_x(A))) for X-splits).
* Sorting categories costs (O(|\Sigma|\log|\Sigma|)).
* Scanning (|\Sigma|-1) prefix candidates costs (O(|\Sigma|)).

This avoids enumerating (2^{|\Sigma|}-2) subsets and is optimal for the conditional log-loss split gain via the sorted-prefix property. 