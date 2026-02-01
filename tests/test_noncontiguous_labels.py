"""
Tests for PartitionTreeClassifier with non-contiguous class labels.

This module tests that the classifier correctly handles labels that are not
sequential integers (e.g., [0, 2, 5] instead of [0, 1, 2]).
"""

import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from partition_tree.estimators.partition_tree import (
    PartitionTreeClassifier,
    PartitionForestClassifier,
)


class TestNonContiguousIntegerLabels:
    """Tests for non-contiguous integer class labels (e.g., [0, 2, 5])."""

    def test_binary_noncontiguous_labels_0_2(self):
        """Test binary classification with labels [0, 2] instead of [0, 1]."""
        np.random.seed(42)
        n_samples = 100

        # Create separable clusters
        X = np.vstack(
            [
                np.random.randn(n_samples // 2, 2) + [0, 0],  # Class 0
                np.random.randn(n_samples // 2, 2) + [5, 5],  # Class 2
            ]
        )
        y = np.array([0] * (n_samples // 2) + [2] * (n_samples // 2))

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        # Check classes are correct
        assert set(clf.classes_) == {0, 2}
        assert len(clf.classes_) == 2

        # Check predictions
        predictions = clf.predict(X)
        assert set(predictions).issubset({0, 2})

        # Check probabilities
        proba = clf.predict_proba(X)
        assert proba.shape == (n_samples, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        # Verify predictions match argmax of probabilities
        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)

        # Verify high accuracy on separable data
        accuracy = accuracy_score(y, predictions)
        assert accuracy > 0.9, f"Expected accuracy > 0.9, got {accuracy}"

    def test_binary_noncontiguous_labels_5_10(self):
        """Test binary classification with labels [5, 10]."""
        np.random.seed(42)
        n_samples = 100

        X = np.vstack(
            [
                np.random.randn(n_samples // 2, 2) + [-3, -3],  # Class 5
                np.random.randn(n_samples // 2, 2) + [3, 3],  # Class 10
            ]
        )
        y = np.array([5] * (n_samples // 2) + [10] * (n_samples // 2))

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        assert set(clf.classes_) == {5, 10}

        predictions = clf.predict(X)
        assert set(predictions).issubset({5, 10})

        proba = clf.predict_proba(X)
        assert proba.shape == (n_samples, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)

    def test_multiclass_noncontiguous_labels_0_2_5(self):
        """Test multiclass classification with labels [0, 2, 5]."""
        np.random.seed(42)
        n_samples = 150

        X = np.vstack(
            [
                np.random.randn(n_samples // 3, 2) + [0, 0],  # Class 0
                np.random.randn(n_samples // 3, 2) + [5, 0],  # Class 2
                np.random.randn(n_samples // 3, 2) + [2.5, 5],  # Class 5
            ]
        )
        y = np.array(
            [0] * (n_samples // 3) + [2] * (n_samples // 3) + [5] * (n_samples // 3)
        )

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        assert set(clf.classes_) == {0, 2, 5}
        assert len(clf.classes_) == 3

        predictions = clf.predict(X)
        assert set(predictions).issubset({0, 2, 5})

        proba = clf.predict_proba(X)
        assert proba.shape == (n_samples, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)

        accuracy = accuracy_score(y, predictions)
        assert accuracy > 0.85, f"Expected accuracy > 0.85, got {accuracy}"

    def test_multiclass_noncontiguous_large_gaps(self):
        """Test with large gaps between label values [1, 100, 1000]."""
        np.random.seed(42)
        n_samples = 150

        X = np.vstack(
            [
                np.random.randn(n_samples // 3, 2) + [0, 0],
                np.random.randn(n_samples // 3, 2) + [5, 0],
                np.random.randn(n_samples // 3, 2) + [2.5, 5],
            ]
        )
        y = np.array(
            [1] * (n_samples // 3)
            + [100] * (n_samples // 3)
            + [1000] * (n_samples // 3)
        )

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        assert set(clf.classes_) == {1, 100, 1000}

        predictions = clf.predict(X)
        assert set(predictions).issubset({1, 100, 1000})

        proba = clf.predict_proba(X)
        assert proba.shape == (n_samples, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)

    def test_multiclass_negative_labels(self):
        """Test with negative non-contiguous labels [-5, 0, 7]."""
        np.random.seed(42)
        n_samples = 150

        X = np.vstack(
            [
                np.random.randn(n_samples // 3, 2) + [0, 0],
                np.random.randn(n_samples // 3, 2) + [5, 0],
                np.random.randn(n_samples // 3, 2) + [2.5, 5],
            ]
        )
        y = np.array(
            [-5] * (n_samples // 3) + [0] * (n_samples // 3) + [7] * (n_samples // 3)
        )

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        assert set(clf.classes_) == {-5, 0, 7}

        predictions = clf.predict(X)
        assert set(predictions).issubset({-5, 0, 7})

        proba = clf.predict_proba(X)
        assert proba.shape == (n_samples, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)


class TestProbaColumnAlignment:
    """Tests to verify probability columns are aligned correctly with classes_."""

    def test_proba_column_order_matches_classes(self):
        """Verify that proba[:, i] corresponds to classes_[i]."""
        np.random.seed(42)

        # Create data where we know the expected class for each sample
        # Class 3 at center (0, 0)
        # Class 7 at (10, 0)
        # Class 11 at (5, 10)
        X = np.array(
            [
                [0, 0],  # Should be class 3
                [10, 0],  # Should be class 7
                [5, 10],  # Should be class 11
                [0.1, 0],  # Should be class 3
                [9.9, 0],  # Should be class 7
                [5, 9.9],  # Should be class 11
            ]
        )
        y = np.array([3, 7, 11, 3, 7, 11])

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        # classes_ should be sorted
        assert list(clf.classes_) == [3, 7, 11]

        proba = clf.predict_proba(X)

        # For the first sample (0, 0), highest proba should be at index 0 (class 3)
        # For the second sample (10, 0), highest proba should be at index 1 (class 7)
        # For the third sample (5, 10), highest proba should be at index 2 (class 11)
        assert np.argmax(proba[0]) == 0, f"Sample 0 should predict class 3 (idx 0)"
        assert np.argmax(proba[1]) == 1, f"Sample 1 should predict class 7 (idx 1)"
        assert np.argmax(proba[2]) == 2, f"Sample 2 should predict class 11 (idx 2)"

    def test_proba_consistency_with_predict(self):
        """Test that predict returns classes_[argmax(predict_proba)]."""
        np.random.seed(42)
        n_samples = 200

        X = np.vstack(
            [
                np.random.randn(n_samples // 4, 3) + [0, 0, 0],
                np.random.randn(n_samples // 4, 3) + [5, 0, 0],
                np.random.randn(n_samples // 4, 3) + [0, 5, 0],
                np.random.randn(n_samples // 4, 3) + [0, 0, 5],
            ]
        )
        # Non-contiguous labels
        y = np.array(
            [10] * (n_samples // 4)
            + [20] * (n_samples // 4)
            + [30] * (n_samples // 4)
            + [40] * (n_samples // 4)
        )

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        proba = clf.predict_proba(X)

        # predictions should equal classes_[argmax(proba, axis=1)]
        expected = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected)


class TestManyNonContiguousClasses:
    """Tests with many non-contiguous classes."""

    def test_10_noncontiguous_classes(self):
        """Test with 10 non-contiguous class labels."""
        np.random.seed(42)
        n_samples_per_class = 30

        # Non-contiguous labels: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
        labels = list(range(2, 30, 3))
        n_classes = len(labels)

        X_list = []
        y_list = []
        for i, label in enumerate(labels):
            # Create a cluster for each class
            X_list.append(np.random.randn(n_samples_per_class, 2) + [i * 3, 0])
            y_list.extend([label] * n_samples_per_class)

        X = np.vstack(X_list)
        y = np.array(y_list)

        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X, y = X[shuffle_idx], y[shuffle_idx]

        clf = PartitionTreeClassifier(max_depth=10, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        assert len(clf.classes_) == n_classes
        assert set(clf.classes_) == set(labels)

        predictions = clf.predict(X)
        assert set(predictions).issubset(set(labels))

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), n_classes)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)

    def test_fibonacci_labels(self):
        """Test with Fibonacci sequence as labels [1, 2, 3, 5, 8, 13]."""
        np.random.seed(42)
        n_samples_per_class = 25

        labels = [1, 2, 3, 5, 8, 13]
        n_classes = len(labels)

        X_list = []
        y_list = []
        for i, label in enumerate(labels):
            X_list.append(np.random.randn(n_samples_per_class, 2) + [i * 4, i * 2])
            y_list.extend([label] * n_samples_per_class)

        X = np.vstack(X_list)
        y = np.array(y_list)

        clf = PartitionTreeClassifier(max_depth=10, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        assert set(clf.classes_) == set(labels)

        predictions = clf.predict(X)
        proba = clf.predict_proba(X)

        assert proba.shape == (len(X), n_classes)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)


class TestMixedNonContiguousLabels:
    """Tests mixing integer and string-like scenarios."""

    def test_float_labels_that_are_integers(self):
        """Test with float labels that represent integers [0.0, 2.0, 5.0]."""
        np.random.seed(42)
        n_samples = 150

        X = np.vstack(
            [
                np.random.randn(n_samples // 3, 2) + [0, 0],
                np.random.randn(n_samples // 3, 2) + [5, 0],
                np.random.randn(n_samples // 3, 2) + [2.5, 5],
            ]
        )
        y = np.array(
            [0.0] * (n_samples // 3)
            + [2.0] * (n_samples // 3)
            + [5.0] * (n_samples // 3)
        )

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        proba = clf.predict_proba(X)

        assert proba.shape == (n_samples, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)


class TestTrainTestSplit:
    """Tests with proper train/test split for non-contiguous labels."""

    def test_noncontiguous_train_test_split(self):
        """Test train/test split with non-contiguous labels."""
        np.random.seed(42)
        n_samples = 300

        X = np.vstack(
            [
                np.random.randn(n_samples // 3, 4) + [0, 0, 0, 0],
                np.random.randn(n_samples // 3, 4) + [3, 3, 3, 3],
                np.random.randn(n_samples // 3, 4) + [6, 6, 6, 6],
            ]
        )
        # Non-contiguous labels
        y = np.array(
            [0] * (n_samples // 3) + [5] * (n_samples // 3) + [99] * (n_samples // 3)
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X_train, y_train)

        # Test predictions on unseen data
        predictions = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        assert set(predictions).issubset({0, 5, 99})
        assert proba.shape == (len(X_test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)

        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.8, f"Expected accuracy > 0.8, got {accuracy}"


class TestForestNonContiguousLabels:
    """Tests for PartitionForestClassifier with non-contiguous labels."""

    def test_forest_noncontiguous_labels(self):
        """Test forest classifier with non-contiguous labels."""
        np.random.seed(42)
        n_samples = 200

        X = np.vstack(
            [
                np.random.randn(n_samples // 4, 3) + [0, 0, 0],
                np.random.randn(n_samples // 4, 3) + [4, 0, 0],
                np.random.randn(n_samples // 4, 3) + [0, 4, 0],
                np.random.randn(n_samples // 4, 3) + [0, 0, 4],
            ]
        )
        y = np.array(
            [1] * (n_samples // 4)
            + [3] * (n_samples // 4)
            + [7] * (n_samples // 4)
            + [15] * (n_samples // 4)
        )

        clf = PartitionForestClassifier(
            n_estimators=10, max_depth=5, min_samples_leaf=1, seed=42
        )
        clf.fit(X, y)

        assert set(clf.classes_) == {1, 3, 7, 15}

        predictions = clf.predict(X)
        proba = clf.predict_proba(X)

        assert proba.shape == (n_samples, 4)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


class TestEdgeCases:
    """Edge cases for non-contiguous labels."""

    def test_single_sample_per_class(self):
        """Test with just one sample per non-contiguous class."""
        X = np.array(
            [
                [0, 0],
                [5, 5],
                [10, 10],
            ]
        )
        y = np.array([0, 5, 10])

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        assert set(clf.classes_) == {0, 5, 10}

        predictions = clf.predict(X)
        proba = clf.predict_proba(X)

        assert proba.shape == (3, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_only_two_of_three_classes_in_predictions(self):
        """Test where model might only predict 2 of 3 classes."""
        np.random.seed(42)

        # Create a scenario where class 50 is rare and might not be predicted
        X = np.vstack(
            [
                np.random.randn(100, 2) + [0, 0],  # Class 0 (many samples)
                np.random.randn(100, 2) + [10, 10],  # Class 100 (many samples)
                np.array([[5, 5]]),  # Class 50 (single sample)
            ]
        )
        y = np.array([0] * 100 + [100] * 100 + [50])

        clf = PartitionTreeClassifier(max_depth=3, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        # All 3 classes should be in classes_
        assert set(clf.classes_) == {0, 50, 100}

        # Probabilities should still be 3-dimensional
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_unsorted_label_order_in_input(self):
        """Test that label order in input doesn't affect results."""
        np.random.seed(42)

        # Labels appear in non-sorted order in training data
        X = np.array(
            [
                [0, 0],  # Label 5
                [1, 1],  # Label 1
                [2, 2],  # Label 10
                [0.1, 0],  # Label 5
                [1.1, 1],  # Label 1
                [2.1, 2],  # Label 10
            ]
        )
        y = np.array([5, 1, 10, 5, 1, 10])

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        # classes_ should be sorted
        assert list(clf.classes_) == [1, 5, 10]

        proba = clf.predict_proba(X)
        assert proba.shape == (6, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        # Verify proba[:, 0] corresponds to class 1, proba[:, 1] to class 5, etc.
        predictions = clf.predict(X)
        expected_preds = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(predictions, expected_preds)

    def test_duplicate_samples_different_labels(self):
        """Test handling of duplicate samples with different labels (overlapping classes)."""
        np.random.seed(42)

        # Some samples are identical but have different labels
        X = np.array(
            [
                [0, 0],
                [0, 0],  # Same as above but different label
                [5, 5],
                [5, 5],  # Same as above but different label
            ]
        )
        y = np.array([1, 3, 7, 9])

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        assert set(clf.classes_) == {1, 3, 7, 9}

        proba = clf.predict_proba(X)
        assert proba.shape == (4, 4)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


class TestClassesOrderPreservation:
    """Tests to verify classes_ order matches probability column order."""

    def test_classes_order_is_sorted(self):
        """Test that classes_ is always sorted."""
        np.random.seed(42)

        # Create data with unsorted labels
        X = np.random.randn(60, 2)
        y = np.array([99] * 20 + [1] * 20 + [50] * 20)

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y)

        # classes_ should be sorted
        assert list(clf.classes_) == [1, 50, 99]

    def test_proba_index_matches_classes_order(self):
        """Test that proba[:, i] is for classes_[i]."""
        np.random.seed(42)

        # Create perfectly separable data with enough samples
        X = np.vstack(
            [
                np.random.randn(20, 2) + [-10, -10],  # Class 999
                np.random.randn(20, 2) + [0, 0],  # Class 1
                np.random.randn(20, 2) + [10, 10],  # Class 500
            ]
        )
        y_train = np.array([999] * 20 + [1] * 20 + [500] * 20)

        clf = PartitionTreeClassifier(max_depth=5, min_samples_leaf=1, seed=42)
        clf.fit(X, y_train)

        # classes_ should be sorted
        assert list(clf.classes_) == [1, 500, 999]

        # Test on clear examples from each cluster center
        X_test = np.array(
            [
                [-10, -10],  # Should predict class 999 (index 2)
                [0, 0],  # Should predict class 1 (index 0)
                [10, 10],  # Should predict class 500 (index 1)
            ]
        )

        proba = clf.predict_proba(X_test)
        predictions = clf.predict(X_test)

        # Verify predictions match expected classes
        assert predictions[0] == 999
        assert predictions[1] == 1
        assert predictions[2] == 500

        # Verify argmax of proba matches the predicted class index
        for i, pred in enumerate(predictions):
            expected_idx = np.where(clf.classes_ == pred)[0][0]
            assert np.argmax(proba[i]) == expected_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
