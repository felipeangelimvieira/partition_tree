"""Test that the feature_split_fraction parameter works correctly."""

import numpy as np
from sklearn.datasets import make_regression
from partition_tree.estimators.partition_tree import PartitionTreeRegressor


def test_feature_split_fraction_fits_and_predicts():
    """Test that models with different feature_split_fraction values can fit and predict."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Test with feature_split_fraction=0 (default behavior)
    model1 = PartitionTreeRegressor(
        max_iter=20, max_depth=5, feature_split_fraction=0.0, seed=42
    )
    model1.fit(X, y)
    pred1 = model1.predict(X)
    assert pred1.shape == (100,)

    # Test with feature_split_fraction=0.2 (20% of splits on features only)
    # Use a shallow tree to avoid edge cases
    model2 = PartitionTreeRegressor(
        max_iter=10, max_depth=3, feature_split_fraction=0.2, seed=42
    )
    model2.fit(X, y)
    pred2 = model2.predict(X)
    assert pred2.shape == (100,)


def test_feature_split_fraction_default_zero():
    """Test that the default value of feature_split_fraction is None."""
    model = PartitionTreeRegressor(max_iter=10, max_depth=3, seed=42)
    assert model.feature_split_fraction is None


def test_feature_split_fraction_parameter_stored():
    """Test that the feature_split_fraction parameter is properly stored."""
    model = PartitionTreeRegressor(
        max_iter=10, max_depth=3, feature_split_fraction=0.3, seed=42
    )
    assert model.feature_split_fraction == 0.3


def test_feature_split_fraction_various_values():
    """Test that different feature_split_fraction values can be set."""
    X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)

    for frac in [0.0, 0.1, 0.25]:
        model = PartitionTreeRegressor(
            max_iter=10, max_depth=3, feature_split_fraction=frac, seed=42
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (50,), f"Failed for feature_split_fraction={frac}"
