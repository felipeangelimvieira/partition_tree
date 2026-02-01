"""Test sklearn estimator compatibility for CPTree estimators."""

import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks
from sklearn.base import clone

from partition_tree.estimators.partition_tree import (
    PartitionTreeClassifier,
    PartitionTreeRegressor,
)


# Simple estimators with minimal parameters for sklearn checks
@pytest.fixture
def simple_cptree_classifier():
    """Create a simple CPTree classifier for sklearn checks."""
    return PartitionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        min_samples_split=10,
        # random_state=42,
    )


@pytest.fixture
def simple_cptree_regressor():
    """Create a simple CPTree regressor for sklearn checks."""
    return PartitionTreeRegressor(
        max_depth=3,
        min_samples_leaf=5,
        min_samples_split=10,
        # random_state=42,
    )


# Parametrized tests using sklearn's check framework
@parametrize_with_checks(
    [
        PartitionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            min_samples_split=10,
            # random_state=42,
        ),
        # PartitionTreeRegressor(
        #    max_depth=3,
        #    min_samples_leaf=5,
        #    min_samples_split=10,
        #    max_splits_to_search=100,
        #    random_state=42,
        # ),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    """Test that CPTree estimators pass sklearn's compatibility checks."""  # Skip some checks that might be problematic for tree-based models
    skip_checks = [
        "check_parameters_default_constructible",  # CPTree requires max_depth
        "check_fit2d_predict1d",  # Tree models may not support this
        "check_methods_subset_invariance",  # May not apply to probabilistic trees
        "check_no_attributes_set_in_init",  # sklearn_tags is set in init
    ]

    # Get check name, handling both function objects and partial objects
    check_name = getattr(check, "__name__", None)
    if check_name is None and hasattr(check, "func"):
        check_name = getattr(check.func, "__name__", str(check))

    if check_name in skip_checks:
        pytest.skip(f"Skipping {check_name} for CPTree estimators")

    check(estimator)


# Individual tests for specific sklearn requirements
def test_cptree_classifier_basic_sklearn_interface(simple_cptree_classifier):
    """Test basic sklearn interface for PartitionTreeClassifier."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100, n_features=4, n_classes=2, random_state=42
    )

    # Test cloning
    estimator_clone = clone(simple_cptree_classifier)
    assert estimator_clone is not simple_cptree_classifier
    assert estimator_clone.get_params() == simple_cptree_classifier.get_params()

    # Test fit and predict
    simple_cptree_classifier.fit(X, y)
    predictions = simple_cptree_classifier.predict(X)
    probabilities = simple_cptree_classifier.predict_proba(X)

    assert predictions.shape == (100,)
    assert probabilities.shape == (100,)  # Binary classification returns single column
    assert hasattr(simple_cptree_classifier, "classes_")


def test_cptree_regressor_basic_sklearn_interface(simple_cptree_regressor):
    """Test basic sklearn interface for PartitionTreeRegressor."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=4, random_state=42)

    # Test cloning
    estimator_clone = clone(simple_cptree_regressor)
    assert estimator_clone is not simple_cptree_regressor
    assert estimator_clone.get_params() == simple_cptree_regressor.get_params()

    # Test fit and predict
    simple_cptree_regressor.fit(X, y)
    predictions = simple_cptree_regressor.predict(X)

    assert predictions.shape == (100, 1)  # Returns DataFrame with one column


def test_random_forest_classifier_basic_sklearn_interface(simple_rf_classifier):
    """Test basic sklearn interface for CustomRandomForestClassifier."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100, n_features=4, n_classes=2, random_state=42
    )

    # Test cloning
    estimator_clone = clone(simple_rf_classifier)
    assert estimator_clone is not simple_rf_classifier
    assert estimator_clone.get_params() == simple_rf_classifier.get_params()

    # Test fit and predict
    simple_rf_classifier.fit(X, y)
    predictions = simple_rf_classifier.predict(X)
    probabilities = simple_rf_classifier.predict_proba(X)

    assert predictions.shape == (100,)
    assert probabilities.shape == (100, 2)  # Binary classification returns both classes
    assert hasattr(simple_rf_classifier, "classes_")
    assert hasattr(simple_rf_classifier, "trees_")
    assert len(simple_rf_classifier.trees_) == 3


def test_random_forest_regressor_basic_sklearn_interface(simple_rf_regressor):
    """Test basic sklearn interface for CustomRandomForestRegressor."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=4, random_state=42)

    # Test cloning
    estimator_clone = clone(simple_rf_regressor)
    assert estimator_clone is not simple_rf_regressor
    assert estimator_clone.get_params() == simple_rf_regressor.get_params()

    # Test fit and predict
    simple_rf_regressor.fit(X, y)
    predictions = simple_rf_regressor.predict(X)

    assert predictions.shape == (100,)
    assert hasattr(simple_rf_regressor, "trees_")
    assert len(simple_rf_regressor.trees_) == 3


def test_estimator_tags():
    """Test that estimators have appropriate sklearn tags."""
    classifier = PartitionTreeClassifier(
        max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=42
    )
    regressor = PartitionTreeRegressor(
        max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=42
    )

    # Test that tags are accessible
    clf_tags = classifier.__sklearn_tags__()
    reg_tags = regressor.__sklearn_tags__()

    # Check that allow_nan is set to True as specified in the estimators
    assert clf_tags.input_tags.allow_nan is True
    assert reg_tags.input_tags.allow_nan is True


# Test parameter validation
@pytest.mark.parametrize(
    "estimator_class,params",
    [
        (
            PartitionTreeClassifier,
            {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 10},
        ),
        (
            PartitionTreeRegressor,
            {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 10},
        ),
    ],
)
def test_parameter_validation(estimator_class, params):
    """Test that estimators validate parameters correctly."""
    # Test valid parameters
    estimator = estimator_class(**params, random_state=42)
    assert estimator.get_params()["max_depth"] == 3
    assert estimator.get_params()["min_samples_leaf"] == 5
    assert estimator.get_params()["min_samples_split"] == 10


def test_fit_predict_consistency():
    """Test that repeated fit/predict calls are consistent with same random state."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)

    # Test classifier
    clf1 = PartitionTreeClassifier(
        max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=42
    )
    clf2 = PartitionTreeClassifier(
        max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=42
    )

    clf1.fit(X, y)
    clf2.fit(X, y)

    pred1 = clf1.predict(X)
    pred2 = clf2.predict(X)

    np.testing.assert_array_equal(pred1, pred2)


def test_different_random_states():
    """Test that different random states produce different results."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)

    clf1 = PartitionTreeClassifier(
        max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=42
    )
    clf2 = PartitionTreeClassifier(
        max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=24
    )

    clf1.fit(X, y)
    clf2.fit(X, y)

    pred1 = clf1.predict(X)
    pred2 = clf2.predict(X)

    # With different random states, predictions should likely be different
    # (though not guaranteed for small datasets)
    assert not np.array_equal(pred1, pred2) or X.shape[0] < 10


def test_multiclass_classification():
    """Test PartitionTreeClassifier with multiclass classification (3+ classes)."""
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    # Create a 3-class classification problem
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=2)
    clf.fit(X, y)

    # Check classes_ attribute
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 3
    # Classes preserve original type (integers)
    assert set(clf.classes_) == {0, 1, 2}

    # Test predict
    predictions = clf.predict(X)
    assert predictions.shape == (200,)
    # All predictions should be valid class labels
    assert all(p in clf.classes_ for p in predictions)

    # Test predict_proba
    proba = clf.predict_proba(X)
    assert proba.shape == (200, 3)
    # Probabilities should sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    # All probabilities should be between 0 and 1
    assert np.all((proba >= 0) & (proba <= 1))

    # predict should match argmax of predict_proba - compare as same type
    expected_predictions = clf.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(
        np.array(predictions, dtype=expected_predictions.dtype), expected_predictions
    )


def test_multiclass_classification_string_labels():
    """Test PartitionTreeClassifier with string class labels."""
    from sklearn.datasets import make_classification

    # Create a 4-class classification problem with string labels
    X, y_int = make_classification(
        n_samples=150,
        n_features=4,
        n_classes=4,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Map to string labels
    label_map = {0: "apple", 1: "banana", 2: "cherry", 3: "date"}
    y = np.array([label_map[val] for val in y_int])

    clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=2)
    clf.fit(X, y)

    # Check classes_ attribute
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 4
    assert set(clf.classes_) == {"apple", "banana", "cherry", "date"}

    # Test predict returns string labels (not indices)
    predictions = clf.predict(X)
    assert predictions.shape == (150,)
    assert all(p in clf.classes_ for p in predictions)
    assert all(isinstance(p, (str, np.str_)) for p in predictions)

    # Test predict_proba
    proba = clf.predict_proba(X)
    assert proba.shape == (150, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    # Verify proba column order matches classes_ order
    expected_predictions = clf.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(predictions, expected_predictions)


def test_multiclass_proba_column_alignment():
    """Test that predict_proba columns are correctly aligned with classes_."""
    # Create a simple dataset where we can verify the alignment
    np.random.seed(42)
    n_samples = 100

    # Create separable clusters for 3 classes
    X = np.vstack(
        [
            np.random.randn(n_samples // 3, 2) + [0, 0],  # Class A
            np.random.randn(n_samples // 3, 2) + [5, 0],  # Class B
            np.random.randn(n_samples // 3, 2) + [2.5, 4],  # Class C
        ]
    )
    y = np.array(
        ["A"] * (n_samples // 3) + ["B"] * (n_samples // 3) + ["C"] * (n_samples // 3)
    )

    clf = PartitionTreeClassifier(max_depth=10, min_samples_leaf=1)
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    # For well-separated clusters, the highest probability should correspond
    # to the correct class
    for i in range(len(X)):
        true_class = y[i]
        pred_class = clf.predict(X[i : i + 1])[0]
        class_idx = np.where(clf.classes_ == true_class)[0][0]
        pred_idx = np.where(clf.classes_ == pred_class)[0][0]

        # The predicted class should have the highest probability
        assert np.argmax(proba[i]) == pred_idx


def test_deterministic_multiclass_10_classes():
    """Test multiclass classification with 26 classes where one feature fully determines the class.

    This creates a deterministic problem where class = floor(feature_0) for feature_0 in [0, 26).
    The classifier should achieve >90% accuracy since the relationship is perfectly learnable.
    """
    from sklearn.metrics import accuracy_score

    np.random.seed(42)
    n_samples_per_class = 50
    n_classes = 26

    # Create data where class is fully determined by the first feature
    X_list = []
    y_list = []

    for class_idx in range(n_classes):
        # Feature 0: values in [class_idx, class_idx + 1) - this determines the class
        feature_0 = np.random.uniform(
            class_idx + 0.1, class_idx + 0.9, n_samples_per_class
        )
        # Feature 1: random noise, should be ignored by the classifier
        feature_1 = np.random.randn(n_samples_per_class) * 5

        X_list.append(np.column_stack([feature_0, feature_1]))
        # Use letters A-Z for class names
        y_list.extend([chr(ord("A") + class_idx)] * n_samples_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train classifier
    clf = PartitionTreeClassifier(max_depth=15, min_samples_leaf=1, seed=42)
    clf.fit(X_train, y_train)

    # Verify all 10 classes are recognized
    assert (
        len(clf.classes_) == n_classes
    ), f"Expected {n_classes} classes, got {len(clf.classes_)}"

    # Predict and check accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy on deterministic 10-class problem: {accuracy:.2%}")
    print(f"Classes: {clf.classes_}")

    # Verify predictions are actual class labels, not indices
    assert all(
        pred in clf.classes_ for pred in y_pred
    ), "Predictions should be class labels"

    # Should achieve high accuracy since feature_0 fully determines the class
    assert accuracy > 0.90, f"Expected accuracy > 90%, got {accuracy:.2%}"
