# Releasing

This repository publishes four distinct artifacts with an explicit order:

1. `estimators` crate to crates.io via the `estimators-v*` tag.
2. `partition_tree` crate to crates.io via the `partition_tree-v*` tag.
3. `pyo3_partition_tree` wheels/sdist to PyPI via the `pyo3-v*` tag.
4. `partition_tree` Python package to PyPI via the `partition_tree-py-v*` tag.

## Why this order

The binding crate depends on the Rust core crate, and the high-level Python
package depends on the published `pyo3-partition-tree` wheels. Releasing in any
other order creates avoidable version skew.

## Local checks

Run these from the repository root before tagging:

```sh
cargo test -p estimators
cargo test -p partition_tree
```

For Python packaging checks:

```sh
cd pyo3_partition_tree && maturin build --release --out dist
cd partition_tree && python -m build --outdir dist
```
