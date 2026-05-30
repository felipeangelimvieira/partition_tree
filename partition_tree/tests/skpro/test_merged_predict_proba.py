"""Correctness tests for the Rust-merged ``predict_proba`` path."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from partition_tree.skpro.distribution import IntervalDistribution
from partition_tree.skpro.partition_tree import PartitionForestRegressor


@pytest.fixture(scope="module")
def fitted_forest_and_data():
    X, y = make_regression(n_samples=300, n_features=5, noise=0.5, random_state=42)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=0)
    model = PartitionForestRegressor(
        n_estimators=10,
        max_leaves=20,
        max_depth=4,
        random_state=7,
    )
    model.fit(
        pd.DataFrame(X_train, columns=[f"f{i}" for i in range(5)]),
        pd.DataFrame({"y": y_train}),
    )
    X_test_df = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(5)])
    return model, X_test_df


def test_predict_proba_returns_interval_distribution(fitted_forest_and_data):
    model, X_test_df = fitted_forest_and_data
    dist = model.predict_proba(X_test_df)
    assert isinstance(dist, IntervalDistribution)


def test_predict_proba_has_correct_number_of_instances(fitted_forest_and_data):
    model, X_test_df = fitted_forest_and_data
    dist = model.predict_proba(X_test_df)
    assert len(dist.index) == len(X_test_df)


def test_intervals_are_sorted_and_non_overlapping(fitted_forest_and_data):
    model, X_test_df = fitted_forest_and_data
    dist = model.predict_proba(X_test_df)
    for i, intervals in enumerate(dist._intervals):
        for j in range(len(intervals) - 1):
            assert intervals[j].high <= intervals[j + 1].low, (
                f"Instance {i}: intervals not sorted/disjoint at j={j}: "
                f"{intervals[j]} vs {intervals[j+1]}"
            )


def test_pdf_integrates_to_one(fitted_forest_and_data):
    """Normalized densities × widths should sum to 1 for every instance."""
    model, X_test_df = fitted_forest_and_data
    dist = model.predict_proba(X_test_df)
    for i in range(len(dist.index)):
        intervals = dist._intervals[i]
        densities = dist.pdf_values[i]
        norm = dist._normalization_factor[i]
        if norm <= 0:
            continue
        total = sum(
            float(d) * (iv.high - iv.low) / norm for d, iv in zip(densities, intervals)
        )
        assert abs(total - 1.0) < 1e-8, f"Instance {i}: integral={total}"


def test_merged_density_equals_average_of_covering_trees(fitted_forest_and_data):
    """Merged density at a point equals mean of per-tree densities covering it.

    Each sample maps to exactly one leaf per tree, so the merged piecewise
    density on a sub-interval must be ``(1/n_trees) * Σ_t density_t`` over the
    trees whose leaf covers that sub-interval.
    """
    model, X_test_df = fitted_forest_and_data
    dist = model.predict_proba(X_test_df)

    forest = model.partition_forest_
    import polars as pl

    X_pol = pl.DataFrame(X_test_df.astype(np.float64))
    per_tree = forest.predict_trees_proba(X_pol)
    n_trees = len(per_tree)

    rng = np.random.default_rng(0)
    n_check = min(15, len(X_test_df))
    for i in rng.choice(len(X_test_df), size=n_check, replace=False):
        intervals = dist._intervals[i]
        if not intervals:
            continue
        seg_idx = int(rng.integers(len(intervals)))
        iv = intervals[seg_idx]
        mid = 0.5 * (iv.low + iv.high)

        expected = 0.0
        for t in range(n_trees):
            for density, low, high in per_tree[t][i].pdf_segments():
                if low <= mid < high or (mid == high):
                    expected += float(density)
                    break
        expected /= n_trees

        merged_density = float(dist.pdf_values[i][seg_idx])
        assert abs(merged_density - expected) < 1e-6, (
            f"Instance {i}: merged density {merged_density} != "
            f"average-of-covering-trees {expected}"
        )
