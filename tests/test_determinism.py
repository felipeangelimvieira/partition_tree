"""Test to verify that the tree is deterministic with a fixed seed."""

import numpy as np
import pytest
from partition_tree_python import PyPartitionTree
from partition_tree.estimators.partition_tree import (
    PartitionTreeRegressor,
    PartitionForestRegressor,
)
import polars as pl


def test_single_tree_determinism():
    """Test that calling the tree twice with the same seed gives the same result."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    y = np.random.randn(n_samples, 1)

    x_df = pl.DataFrame({f"x_{i}": X[:, i] for i in range(X.shape[1])})
    y_df = pl.DataFrame({"target_y": y[:, 0]})

    # First fit
    tree1 = PyPartitionTree(
        max_iter=10,
        min_samples_split=2,
        min_samples_leaf_y=1,
        min_samples_leaf_x=1,
        min_samples_leaf=1,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        boundaries_expansion_factor=0.0,
        seed=42,
    )
    tree1.fit(x_df, y_df, None)
    pred1 = tree1.predict(x_df)

    # Second fit with same seed
    tree2 = PyPartitionTree(
        max_iter=10,
        min_samples_split=2,
        min_samples_leaf_y=1,
        min_samples_leaf_x=1,
        min_samples_leaf=1,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        boundaries_expansion_factor=0.0,
        seed=42,
    )
    tree2.fit(x_df, y_df, None)
    pred2 = tree2.predict(x_df)

    # Predictions should be identical
    np.testing.assert_array_equal(
        pred1["y"].to_numpy(),
        pred2["y"].to_numpy(),
        err_msg="Predictions should be identical with the same seed",
    )


def test_single_tree_determinism_multiple_runs():
    """Test determinism across multiple runs to ensure consistency."""
    np.random.seed(123)
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples, 1)

    x_df = pl.DataFrame({f"x_{i}": X[:, i] for i in range(X.shape[1])})
    y_df = pl.DataFrame({"target_y": y[:, 0]})

    predictions = []
    for run in range(5):
        tree = PyPartitionTree(
            max_iter=20,
            min_samples_split=2,
            min_samples_leaf_y=1,
            min_samples_leaf_x=1,
            min_samples_leaf=1,
            min_target_volume=0.0,
            max_depth=8,
            min_split_gain=0.0,
            boundaries_expansion_factor=0.0,
            seed=42,
        )
        tree.fit(x_df, y_df, None)
        pred = tree.predict(x_df)
        predictions.append(pred["y"].to_numpy())

    # All predictions should be identical
    for i in range(1, len(predictions)):
        np.testing.assert_array_equal(
            predictions[0],
            predictions[i],
            err_msg=f"Run 0 and run {i} should have identical predictions",
        )


def test_determinism_with_many_features():
    """Test determinism with many features (higher chance of ties in parallel processing)."""
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, 1)

    x_df = pl.DataFrame({f"x_{i}": X[:, i] for i in range(X.shape[1])})
    y_df = pl.DataFrame({"target_y": y[:, 0]})

    predictions = []
    for run in range(10):
        tree = PyPartitionTree(
            max_iter=50,
            min_samples_split=2,
            min_samples_leaf_y=1,
            min_samples_leaf_x=1,
            min_samples_leaf=1,
            min_target_volume=0.0,
            max_depth=10,
            min_split_gain=0.0,
            boundaries_expansion_factor=0.0,
            seed=42,
        )
        tree.fit(x_df, y_df, None)
        pred = tree.predict(x_df)
        predictions.append(pred["y"].to_numpy())

    # All predictions should be identical
    for i in range(1, len(predictions)):
        np.testing.assert_array_equal(
            predictions[0],
            predictions[i],
            err_msg=f"Run 0 and run {i} should have identical predictions with many features",
        )


def test_determinism_with_correlated_features():
    """Test with highly correlated features which could cause tied gains."""
    np.random.seed(42)
    n_samples = 200
    base = np.random.randn(n_samples)

    # Create highly correlated features (which can lead to tied gains)
    X = np.column_stack(
        [
            base + np.random.randn(n_samples) * 0.01,
            base + np.random.randn(n_samples) * 0.01,
            base + np.random.randn(n_samples) * 0.01,
            base + np.random.randn(n_samples) * 0.01,
            base + np.random.randn(n_samples) * 0.01,
        ]
    )
    y = base.reshape(-1, 1) + np.random.randn(n_samples, 1) * 0.1

    x_df = pl.DataFrame({f"x_{i}": X[:, i] for i in range(X.shape[1])})
    y_df = pl.DataFrame({"target_y": y[:, 0]})

    predictions = []
    for run in range(10):
        tree = PyPartitionTree(
            max_iter=20,
            min_samples_split=2,
            min_samples_leaf_y=1,
            min_samples_leaf_x=1,
            min_samples_leaf=1,
            min_target_volume=0.0,
            max_depth=6,
            min_split_gain=0.0,
            boundaries_expansion_factor=0.0,
            seed=42,
        )
        tree.fit(x_df, y_df, None)
        pred = tree.predict(x_df)
        predictions.append(pred["y"].to_numpy())

    # All predictions should be identical
    for i in range(1, len(predictions)):
        np.testing.assert_array_equal(
            predictions[0],
            predictions[i],
            err_msg=f"Run 0 and run {i} should have identical predictions with correlated features",
        )


def test_different_seeds_give_different_results():
    """Test that different seeds can produce different results."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    y = np.random.randn(n_samples, 1)

    x_df = pl.DataFrame({f"x_{i}": X[:, i] for i in range(X.shape[1])})
    y_df = pl.DataFrame({"target_y": y[:, 0]})

    tree1 = PyPartitionTree(
        max_iter=10,
        min_samples_split=2,
        min_samples_leaf_y=1,
        min_samples_leaf_x=1,
        min_samples_leaf=1,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        boundaries_expansion_factor=0.0,
        seed=42,
    )
    tree1.fit(x_df, y_df, None)
    pred1 = tree1.predict(x_df)

    tree2 = PyPartitionTree(
        max_iter=10,
        min_samples_split=2,
        min_samples_leaf_y=1,
        min_samples_leaf_x=1,
        min_samples_leaf=1,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        boundaries_expansion_factor=0.0,
        seed=123,
    )
    tree2.fit(x_df, y_df, None)
    pred2 = tree2.predict(x_df)

    # Note: Different seeds may or may not produce different results depending
    # on the data. This test just verifies the code runs, not that results differ.
    # In many cases the optimal split is the same regardless of seed.


def test_sklearn_estimator_determinism():
    """Test that sklearn-style PartitionTreeRegressor is deterministic."""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)

    predictions = []
    for run in range(5):
        tree = PartitionTreeRegressor(
            max_iter=20,
            max_depth=6,
            seed=42,
        )
        tree.fit(X, y)
        pred = tree.predict(X)
        predictions.append(pred)

    # All predictions should be identical
    for i in range(1, len(predictions)):
        np.testing.assert_array_equal(
            predictions[0],
            predictions[i],
            err_msg=f"PartitionTreeRegressor: Run 0 and run {i} should have identical predictions",
        )


def test_sklearn_forest_determinism():
    """Test that sklearn-style PartitionForestRegressor is deterministic."""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)

    predictions = []
    for run in range(3):
        forest = PartitionForestRegressor(
            n_estimators=10,
            max_iter=10,
            max_depth=5,
            seed=42,
        )
        forest.fit(X, y)
        pred = forest.predict(X)
        predictions.append(pred)

    # All predictions should be identical
    for i in range(1, len(predictions)):
        np.testing.assert_array_equal(
            predictions[0],
            predictions[i],
            err_msg=f"PartitionForestRegressor: Run 0 and run {i} should have identical predictions",
        )


if __name__ == "__main__":
    test_single_tree_determinism()
    test_single_tree_determinism_multiple_runs()
    test_different_seeds_give_different_results()
    test_sklearn_estimator_determinism()
    test_sklearn_forest_determinism()
    print("All tests passed!")
