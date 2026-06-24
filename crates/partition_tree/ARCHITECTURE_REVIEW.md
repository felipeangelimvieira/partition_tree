# partition_tree crate architecture review

## Findings
- The crate is organized around clear domain modules in `src/`: `rules.rs`, `density.rs`, `onedimpartition.rs`, `cell.rs`, `node.rs`, `tree.rs`, `estimator.rs`, and `estimator_forest.rs`.
- `src/lib.rs` exposes many modules directly, which makes internals accessible but also widens the public surface.
- `tree.rs` currently carries two responsibilities: tree storage/traversal and training-time split orchestration.
- `split.rs` and `dtype_adapter.rs` centralize split scoring plus data-type adaptation. This keeps logic in one place but increases coupling to Polars-specific concerns.
- Prediction concerns are already partly separated (`predict/mod.rs`, `predict/probability.rs`), which is a good extensibility hook.
- Serialization is grouped under `serde/` (`schema.rs`, `partitions.rs`, `dataframe_schema.rs`), a strong boundary that can be leveraged for versioning.

## Risks
- **Growth risk in `tree.rs`:** as new training modes appear (e.g., online fit, constrained split policies), one file can become a bottleneck.
- **Type-extensibility friction:** adding new partition/rule combinations likely requires edits in several places (`rules.rs`, `density.rs`, `onedimpartition.rs`, serde modules).
- **Data backend lock-in:** `dataframe.rs` and `dtype_adapter.rs` make Polars a de facto hard dependency for domain-level logic.
- **Public API drift:** broad module exports in `lib.rs` can make future refactors harder without breaking downstream users.

## Recommendations
1. **Stabilize the public API surface**
   - Re-export only intended entry points from `lib.rs` (primarily estimator-facing types).
   - Keep lower-level modules crate-private where possible.
2. **Extract builder responsibilities from `tree.rs`**
   - Introduce a focused builder component (`TreeBuilder`) for fit-time orchestration.
   - Keep `Tree` focused on model representation + inference-oriented traversal.
3. **Introduce pluggable split strategy boundary**
   - Define a trait (e.g., `SplitStrategy`) implemented by current split logic.
   - This enables alternate split criteria without editing core tree code.
4. **Isolate data-backend adaptation**
   - Keep Polars conversion/typing in adapter modules, pass backend-agnostic structures to core split/tree logic when feasible.
5. **Add serialization version marker**
   - Extend `serde/schema.rs` with an explicit model format version to support forward-compatible evolution.

## Refactor Plan
- [ ] **Phase 1 (safe surface hardening):** tighten `lib.rs` exports and document the stable API.
- [ ] **Phase 2 (responsibility split):** move fit orchestration from `tree.rs` into a small builder module while preserving behavior.
- [ ] **Phase 3 (strategy seam):** introduce `SplitStrategy` and wire existing logic as default implementation.
- [ ] **Phase 4 (data boundary):** progressively reduce direct Polars usage in domain internals behind adapters.
- [ ] **Phase 5 (schema evolution):** add versioned serialization and backward-compat checks in serde tests.

## Future Use Cases
- Custom split objectives (cost-sensitive, fairness-aware, monotonic constraints).
- Alternate backends for data ingestion (non-Polars tabular sources).
- Online/incremental training where builder lifecycle differs from static batch fit.
- Lightweight inference runtimes that only require compact tree nodes + prediction modules.
- Long-lived model persistence across crate versions through versioned serde schema.

## Questions/Assumptions
- Assumes current external users primarily consume estimator-level APIs, not deep internals.
- Assumes preserving current prediction behavior is mandatory while refactoring boundaries.
- Assumes Polars remains the default backend, with extensibility as a medium-term goal (not immediate replacement).
- Open question: should `estimators` traits own more of the stable contract, with `partition_tree` narrowing exposed internals?
- Open question: is backward compatibility for serialized models required across all 0.x releases or only minor transitions?
