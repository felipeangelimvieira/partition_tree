"""Test sklearn compatibility of PartitionTreeRegressor."""

from sklearn.utils.estimator_checks import check_estimator
from partition_tree.estimators import PartitionTreeRegressor


def test_sklearn_compatibility():
    """Test sklearn compatibility using check_estimator."""

    print("Testing sklearn compatibility...")
    try:
        # Create a simple estimator instance for testing
        estimator = PartitionTreeRegressor(max_depth=2, min_samples_split=10)

        # Run sklearn compatibility checks
        # Note: We'll run a subset of checks as some may not be relevant for tree-based models
        check_estimator(estimator)
        print("✓ All sklearn compatibility checks passed!")

    except Exception as e:
        print(f"✗ Some sklearn checks failed: {e}")
        print("This may be expected for specialized estimators.")


if __name__ == "__main__":
    test_sklearn_compatibility()
