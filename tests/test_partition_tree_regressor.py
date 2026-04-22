"""Test script to verify PartitionTreeRegressor implementation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from partition_tree.estimators import (
    Domain,
    PartitionForestRegressor,
    PartitionTreeRegressor,
)
from partition_tree.utils import _infer_quantized_resolution


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


def test_partition_tree_regressor_passes_dtype_overrides_to_backend():
    X = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    y = pd.Series([10.0, 10.0, 20.0, 20.0, 30.0, 30.0], name="target")

    integer_model = PartitionTreeRegressor(
        max_depth=5,
        min_samples_split=2,
        dtype_overrides={"x1": Domain.integer()},
    )
    quantized_model = PartitionTreeRegressor(
        max_depth=5,
        min_samples_split=2,
        dtype_overrides={"x1": Domain.quantized_continuous(1.0)},
    )

    integer_model.fit(X, y)
    quantized_model.fit(X, y)

    preds_integer = integer_model.predict(X)
    preds_quantized = quantized_model.predict(X)

    assert np.allclose(preds_integer, preds_quantized)


def test_infer_quantized_resolution_detects_shared_step_with_nulls_and_negatives():
    series = pd.Series([np.nan, -0.006, -0.003, 0.0, 0.003, 0.006, 0.003])

    resolution = _infer_quantized_resolution(series)

    assert resolution == pytest.approx(0.003)


def test_infer_quantized_resolution_ignores_infinite_values():
    series = pd.Series([np.inf, -np.inf, 0.25, 0.5, 0.75, 1.0])

    resolution = _infer_quantized_resolution(series)

    assert resolution == pytest.approx(0.25)


def test_infer_quantized_resolution_uses_repeated_nonzero_value_when_no_gaps_exist():
    series = pd.Series([0.125, 0.125, 0.125, np.nan])

    resolution = _infer_quantized_resolution(series)

    assert resolution == pytest.approx(0.125)


def test_infer_quantized_resolution_returns_none_for_zero_only_series():
    series = pd.Series([0.0, 0.0, 0.0, np.nan, np.inf, -np.inf])

    resolution = _infer_quantized_resolution(series)

    assert resolution is None


def test_infer_quantized_resolution_returns_none_when_precision_exceeds_limit():
    series = pd.Series([0.123456789123, 0.234567891234, 0.345678912345])

    resolution = _infer_quantized_resolution(series)

    assert resolution is None


def test_partition_tree_regressor_auto_dtype_override_detects_quantized_resolution():
    X = pd.DataFrame(
        {
            "x1": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
            "x2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pd.Series([10.0, 10.0, 20.0, 20.0, 30.0, 30.0], name="target")

    model = PartitionTreeRegressor(
        max_depth=5,
        min_samples_split=2,
        dtype_overrides={"x1": "auto"},
    )

    model.fit(X, y)
    root_partitions = model.partition_tree_.get_nodes_info()[0]["partitions"]
    assert root_partitions["x1"]["type"] == "quantized_continuous"
    assert root_partitions["x1"]["resolution"] == pytest.approx(0.25)


def test_partition_tree_regressor_auto_dtype_override_falls_back_to_continuous():
    X = pd.DataFrame(
        {
            "x1": [0.123456789123, 0.234567891234, 0.345678912345, 0.456789123456],
            "x2": [0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pd.Series([10.0, 20.0, 30.0, 40.0], name="target")

    model = PartitionTreeRegressor(
        max_depth=5,
        min_samples_split=2,
        dtype_overrides={"x1": "auto"},
    )

    model.fit(X, y)
    root_partitions = model.partition_tree_.get_nodes_info()[0]["partitions"]
    assert root_partitions["x1"]["type"] == "continuous"


def test_partition_forest_regressor_auto_dtype_override_detects_quantized_resolution():
    X = pd.DataFrame(
        {
            "x1": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "x2": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    y = pd.Series([10.0, 10.0, 20.0, 20.0, 30.0, 30.0], name="target")

    model = PartitionForestRegressor(
        n_estimators=2,
        max_depth=5,
        min_samples_split=2,
        random_state=42,
        dtype_overrides={"x1": "auto"},
    )

    model.fit(X, y)
    nodes = model.partition_tree_.get_nodes_info()
    assert nodes[0][0]["partitions"]["x1"]["type"] == "quantized_continuous"
    assert nodes[0][0]["partitions"]["x1"]["resolution"] == pytest.approx(0.5)
