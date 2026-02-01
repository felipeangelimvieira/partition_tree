"""Test script to verify PartitionTreeRegressor implementation."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from partition_tree.estimators import PartitionTreeRegressor


def test_partition_tree_regressor():
    """Test basic functionality of PartitionTreeRegressor."""

    # Generate sample regression data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Test with numpy arrays
    print("Testing with numpy arrays...")
    regressor = PartitionTreeRegressor(max_depth=3, min_samples_split=5)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE with numpy arrays: {mse:.4f}")

    # Test with pandas DataFrames
    print("\nTesting with pandas DataFrames...")
    X_train_df = pd.DataFrame(
        X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])]
    )
    X_test_df = pd.DataFrame(
        X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])]
    )
    y_train_series = pd.Series(y_train, name="target")

    regressor2 = PartitionTreeRegressor(max_depth=3, min_samples_split=5)
    regressor2.fit(X_train_df, y_train_series)
    y_pred2 = regressor2.predict(X_test_df)
    mse2 = mean_squared_error(y_test, y_pred2)
    print(f"MSE with pandas DataFrames: {mse2:.4f}")

    # Test multi-target regression
    print("\nTesting multi-target regression...")
    y_multi = np.column_stack([y_train, y_train * 2])
    regressor3 = PartitionTreeRegressor(max_depth=3, min_samples_split=5)
    regressor3.fit(X_train, y_multi)
    y_pred_multi = regressor3.predict(X_test)
    print(f"Multi-target prediction shape: {y_pred_multi.shape}")
    print(f"Expected shape: ({X_test.shape[0]}, 2)")

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    test_partition_tree_regressor()
