"""
Test to investigate why get_leaves_info returns partitions on target without any categories.

This test creates a classification dataset with 10 classes and fits a tree with max_iter=3
to reproduce the issue where categorical partitions end up with empty category sets.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from partition_tree.estimators.partition_tree import PartitionTreeClassifier


def test_leaves_have_nonempty_target_categories():
    """
    Test that all leaves have non-empty category sets for target partitions.

    With 10 classes and max_iter=3, we should have at most 4 leaves.
    Each leaf's target partition should contain at least one category.
    """
    # Create a classification dataset with 10 classes
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Test with various max_iter values to trigger different tree structures
    for max_iter in [3, 10, 20, 50]:
        print(f"\n--- Testing with max_iter={max_iter} ---")
        clf = PartitionTreeClassifier(
            max_iter=max_iter,
            min_samples_split=1,
            min_samples_leaf=0,
            seed=42,
        )
        clf.fit(X, y)
        _check_leaves_have_categories(clf, max_iter)


def _check_leaves_have_categories(clf, max_iter):
    # Get leaves info
    leaves_info = clf.get_leaves_info()

    print(f"  Tree has {len(leaves_info)} leaves")

    # Check each leaf
    empty_found = False
    for i, leaf in enumerate(leaves_info):
        for name, partition in leaf["partitions"].items():
            if "categories" in partition:
                categories = partition["categories"]

                # This is the bug check: no leaf should have empty categories
                if name.startswith("target"):
                    if len(categories) == 0:
                        print(
                            f"  *** BUG: Leaf {i} has empty categories for target partition '{name}'! ***"
                        )
                        empty_found = True

    if not empty_found:
        print(f"  All {len(leaves_info)} leaves have non-empty target categories")

    # Still assert at the end
    for i, leaf in enumerate(leaves_info):
        for name, partition in leaf["partitions"].items():
            if "categories" in partition and name.startswith("target"):
                assert (
                    len(partition["categories"]) > 0
                ), f"max_iter={max_iter}: Leaf {i} has empty categories for target partition '{name}'."


def test_leaves_target_categories_cover_samples():
    """
    Test that each leaf's target categories cover the actual samples in that leaf.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=42,
    )

    clf = PartitionTreeClassifier(
        max_iter=3,
        min_samples_split=1,
        min_samples_leaf=0,
        seed=42,
    )
    clf.fit(X, y)

    leaves_info = clf.get_leaves_info()

    for i, leaf in enumerate(leaves_info):
        indices_xy = leaf["indices_xy"]

        # Get the actual target values for samples in this leaf
        actual_classes = set(str(y[idx]) for idx in indices_xy)

        # Get the categories in the leaf's target partition
        for name, partition in leaf["partitions"].items():
            if name.startswith("target__") and "categories" in partition:
                leaf_categories = set(partition["categories"])

                # Check that actual classes are covered by the partition
                missing = actual_classes - leaf_categories
                if missing:
                    print(
                        f"Leaf {i}: actual classes {actual_classes}, partition categories {leaf_categories}"
                    )
                    print(f"  Missing categories for actual samples: {missing}")

                # At minimum, the categories should cover all samples in the leaf
                # (this is a softer check than requiring exact match)
                assert (
                    actual_classes <= leaf_categories or len(leaf_categories) == 0
                ), f"Leaf {i} has samples with classes {actual_classes} but partition only has {leaf_categories}"


def test_exploration_splits_preserve_categories():
    """
    Test with exploration_split_budget to verify exploration splits don't break categories.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=42,
    )

    print("--- Testing with exploration_split_budget ---")
    for budget in [5, 10, 20]:
        clf = PartitionTreeClassifier(
            max_iter=30,
            min_samples_split=1,
            min_samples_leaf=0,
            exploration_split_budget=budget,
            seed=42,
        )
        clf.fit(X, y)
        print(f"  exploration_budget={budget}:")
        _check_leaves_have_categories(clf, f"budget={budget}")


def test_noncontiguous_class_labels():
    """
    Test with non-contiguous class labels (e.g., 0, 5, 10, 15...).

    This test specifically targets the bug where get_leaves_info used
    category codes as indices into domain_names instead of looking up
    the position in the domain first.
    """
    print("--- Testing with non-contiguous class labels ---")

    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Remap labels to non-contiguous values: 0->10, 1->21, 2->32, etc.
    label_mapping = {i: i * 11 + 10 for i in range(10)}
    y_noncontig = np.array([label_mapping[label] for label in y])

    print(f"  Original labels: {sorted(np.unique(y))}")
    print(f"  Mapped labels: {sorted(np.unique(y_noncontig))}")

    clf = PartitionTreeClassifier(
        max_iter=10,
        min_samples_split=1,
        min_samples_leaf=0,
        seed=42,
    )
    clf.fit(X, y_noncontig)

    leaves_info = clf.get_leaves_info()
    print(f"  Tree has {len(leaves_info)} leaves")

    # Check each leaf has categories
    for i, leaf in enumerate(leaves_info):
        for name, partition in leaf["partitions"].items():
            if "categories" in partition and name.startswith("target"):
                categories = partition["categories"]
                print(
                    f"    Leaf {i}: {len(categories)} categories -> {categories[:5]}{'...' if len(categories) > 5 else ''}"
                )

                assert len(categories) > 0, (
                    f"Leaf {i} has empty categories for partition '{name}'. "
                    f"This indicates the bug where codes were used as indices."
                )


if __name__ == "__main__":
    test_leaves_have_nonempty_target_categories()
    print("\n" + "=" * 60 + "\n")
    test_leaves_target_categories_cover_samples()
    print("\n" + "=" * 60 + "\n")
    test_exploration_splits_preserve_categories()
    print("\n" + "=" * 60 + "\n")
    test_noncontiguous_class_labels()
