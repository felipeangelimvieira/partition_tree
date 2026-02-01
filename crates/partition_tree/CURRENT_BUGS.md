Below are the main potential bugs / logic issues found in the source under src:

Wrong row index mapping in partition::Partition::get_true_indices
It currently returns the position inside subset (0..subset.len()) instead of the actual row id (subset[idx]). This corrupts indices as splits go deeper (children operate on remapped tiny index ranges). This likely explains empty / duplicated leaf sample counts.
Fix shown below (see Patch 1).

Leaf indexing bug in best‑first builder in tree::BestFirstTreeBuilder::build_tree
When a popped candidate is declared a leaf, you do:
let leaf_idx = self.nodes.len(); self.leaves.push(leaf_idx); without pushing (or knowing) the actual node index (except root special case). Nodes already exist in self.nodes (root at 0, later children pushed earlier). This records wrong leaf indices. Printing leaf details then references unrelated nodes. Solution: store each node’s index inside SplitCandidate and push that index instead of self.nodes.len(). (See Patch 2.)

Split selection direction reversed in both split::find_best_split_column and split::_evaluate_split_points
Comments say “pick the min” / “minimum loss” but code uses .max_by. With losses like -34657 (better) vs +34657 (worse), the worse (higher) is selected. Fix by switching to .min_by consistently (Patch 3). Also rename comments (or invert the sign of the scoring function) for clarity.

Inconsistent loss semantics / naming in split::logistic_gain
Function name implies “gain” (higher better) but rest of pipeline treats lower as better (after fixing #3). Either rename to logistic_score and document “lower is better” or return the negative so that higher is better and keep .max_by.

Single unique value treated as viable split in split::find_best_continuous_split
When only one unique value exists, you still “split” at that value (no separation). Should return SplitResult::InvalidSplit (or loss = +∞) to avoid meaningless splits.

Categorical counting ignores return_unique in dataframe::Column::count_categories
Parameter is unused; yet categorical split logic expects unique categories. Clarify intent or remove param.

Unstable / arbitrary category ordering in categorical splits
Using HashMap iteration order for (unique_categories, counts) without sorting makes split reproducibility unstable. Consider sorting categories to stabilize results.

Wrapper signature mismatch for tests
Current public split::search_split_column requires &SplitRestrictions, but tests (and example snippets) call a 3‑argument version. Provide a convenience wrapper with defaults or update tests.

Root leaf case indexing bug
If root becomes a leaf, current logic would assign leaf index = self.nodes.len() (which is already 1 after pushing root) instead of 0. The patch in #2 also resolves this.

Potential misuse of relative volume in partition::Partition::relative_volume
For infinite continuous domains relative volume returns 0.0, which can make restrictions.is_node_valid fail immediately unless min_target_volume is 0. Consider treating infinite domain as 1.0 or excluding from product.

Off‑by‑one style constraint in split::SplitRestrictions::is_node_valid
Uses node.indices_xy.len() > self.min_samples_leaf instead of >=, making exact threshold invalid.

Probability formulas in split::logistic_gain look dimensionally inconsistent ((*xy / (*x * target_volume_left)) * (*xy / total_xy) etc.) which may produce extreme magnitudes. Re‑derive to ensure intended statistical meaning.

Unnecessary allocation / unused variable in partition::Partition::mean (let v: Vec<f64> = ... shadowed / unused).

Performance: Multiple debug println! calls in hot paths (find_best_continuous_split, validity checks) will slow large datasets. Gate behind a feature flag or log level.

Tests likely expect deterministic behavior; current unordered HashSet / HashMap iteration can cause flakiness (means, category ordering, etc.).

Patches for the most critical functional bugs (index mapping, leaf indexing, selection direction) follow.

(Adjust any now‑dead comments referring to “max” accordingly.)

Recommended (but not shown) additional changes:

Add a convenience wrapper or default SplitRestrictions builder for tests.
Return +∞ (and InvalidSplit) when only one unique value in a numeric column.
Implement deterministic ordering for categorical splits (e.g., collect to Vec, sort).
Clarify probability math in logistic_gain and document optimization direction.
Consider changing relative_volume semantics for infinite domains.
Let me know if you want patches for any of the “recommended” items as well.