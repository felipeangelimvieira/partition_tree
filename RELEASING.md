# Releasing

This repository publishes four distinct artifacts with an explicit order:

1. `estimators` crate to crates.io via the `estimators-v*` tag.
2. `partition_tree` crate to crates.io via the `partition_tree-v*` tag.
3. `pyo3_partition_tree` wheels/sdist to PyPI via the `pyo3-v*` tag.
4. `partition_tree` Python package to PyPI via the `partition_tree-py-v*` tag.

The Python package workflow also updates the documentation website:

- Pull requests from the main repository publish preview docs to `previews/pr-<number>/`.
- Pushes to `main` publish the development docs to `dev/`.
- Published `partition_tree-py-v*` GitHub releases publish versioned docs to `/<version>/`,
  refresh `latest/`, and update the production root for `partitiontree.com`.
- The `Docs Latest CI` workflow can be triggered manually to rebuild `latest/` and
    the production root directly from the current `main` branch.

## Required credentials

The release automation uses two different credential scopes:

- Local notebook or terminal authentication for `gh release create`. Run `gh auth login`
    first, or provide a `GH_TOKEN` with repository write access.
- GitHub Actions repository secrets for package publishing.

Configure these repository secrets in Settings > Secrets and variables > Actions:

- `CRATES_IO_TOKEN` for publishing the Rust crates.
- `PYPI_API_TOKEN` for publishing both Python distributions.

## Creating the GitHub release

The publish workflows are triggered by a published GitHub Release with the right tag.
You can create that release, with GitHub-generated release notes, from a notebook by
calling the helper script:

```sh
./scripts/create_github_release.sh estimators --wait
./scripts/create_github_release.sh partition_tree --wait
./scripts/create_github_release.sh pyo3 --wait
./scripts/create_github_release.sh python --wait
```

The script uses `gh release create --generate-notes` by default. Pass
`--notes-file <path>` if you want notebook-generated notes instead. If you omit
the version argument, the script infers it from the relevant manifest:
`Cargo.toml` for Rust crates, `pyproject.toml` for Python packages, and both
`pyo3_partition_tree/Cargo.toml` and `pyo3_partition_tree/pyproject.toml` for
the PyO3 package.

The `partition_tree` Python package workflow builds the package artifact, tests
that local artifact against the already-published `pyo3-partition-tree`
dependency from PyPI, and only then publishes the Python package itself.

Use the script in that order. `partition_tree` depends on `estimators`, the
PyO3 package depends on the Rust crates, and the Python package depends on the
published `pyo3-partition-tree` wheel. The `--wait` flag makes the notebook wait
for the matching GitHub Actions release workflow to finish before moving to the
next dependent release.

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
