"""Test that the max_leaves parameter works correctly."""

import numpy as np
from sklearn.datasets import make_regression
from partition_tree.estimators.partition_tree import PartitionTreeRegressor


def test_max_leaves_fits_and_predicts():
    """Test that models with different max_leaves values can fit and predict."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    model1 = PartitionTreeRegressor(max_leaves=20, max_depth=5, random_state=42)
    model1.fit(X, y)
    pred1 = model1.predict(X)
    assert pred1.shape == (100,)

    model2 = PartitionTreeRegressor(max_leaves=10, max_depth=3, random_state=42)
    model2.fit(X, y)
    pred2 = model2.predict(X)
    assert pred2.shape == (100,)


def test_max_leaves_default_none():
    """Test that the default value of max_leaves is None."""
    model = PartitionTreeRegressor(max_depth=3, random_state=42)
    assert model.max_leaves is None


def test_max_leaves_parameter_stored():
    """Test that the max_leaves parameter is properly stored."""
    model = PartitionTreeRegressor(max_leaves=30, max_depth=3, random_state=42)
    assert model.max_leaves == 30


def test_max_leaves_various_values():
    """Test that different max_leaves values can be set."""
    X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)

    for leaves in [5, 10, 25]:
        model = PartitionTreeRegressor(max_leaves=leaves, max_depth=3, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (50,), f"Failed for max_leaves={leaves}"
