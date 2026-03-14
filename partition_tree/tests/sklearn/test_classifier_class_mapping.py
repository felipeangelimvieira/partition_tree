"""Tests for sklearn classifier class mapping.

Verifies that the Partition Tree classifiers correctly map the internal
Rust-backend category codes (cat_0, cat_1, …) back to the original class
labels, regardless of label dtype, class ordering in the data, or number
of classes.

Background
----------
These tests guard against a subtle bug in the Python ↔ Rust FFI boundary.

During classification, the Python layer converts target labels to a Polars
``Enum`` column (lexicographically ordered categories) before passing data
to the Rust backend via **pyo3-polars**.  However, pyo3-polars silently
converts ``Enum`` to ``Categorical`` at the FFI boundary, and ``Categorical``
assigns **physical integer codes by insertion order** (i.e. the order rows
appear in the data), not by the lexicographic order of the ``Enum``.

The Rust backend then builds a probability domain with names ``cat_0``,
``cat_1``, … corresponding to the *sorted physical codes* it receives.
If the Python layer naively assumes ``cat_N`` maps to the N-th category
in the original ``Enum`` ordering, the mapping is wrong whenever the
insertion order differs from lexicographic order — which is almost always
the case in practice (e.g. Iris, where class ``1`` is the first sample
seen in most train splits, so physical code 0 → class ``'1'``, not
class ``'0'``).

The fix captures the actual ``Categorical`` physical-to-logical mapping
*before* casting to ``Enum``, so ``cat_N`` is resolved to the correct
original class label.

Test organisation
-----------------
- **TestClassMapping**: Predicted labels are a subset of the training labels
  and preserve the original dtype (int, string, binary).
- **TestProbabilities**: ``predict_proba`` output has the right shape, rows
  sum to 1, and ``argmax`` matches ``predict``.
- **TestAccuracy**: Accuracy exceeds a threshold that any systematic class
  permutation would violate (a cyclic permutation gives ~0 % on Iris).
- **TestDataOrdering**: Shuffling or reversing the training data does not
  break the mapping — directly targets the insertion-order bug.
- **TestWineDataset**: Sanity check on a second dataset with a different
  feature space and class distribution.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from partition_tree.sklearn import (
    PartitionForestClassifier,
    PartitionTreeClassifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def iris_data():
    """Iris dataset split into train/test (integer labels 0, 1, 2)."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture()
def iris_string_labels(iris_data):
    """Iris with string class labels: 'setosa', 'versicolor', 'virginica'."""
    X_train, X_test, y_train, y_test = iris_data
    mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
    y_train_str = np.array([mapping[c] for c in y_train])
    y_test_str = np.array([mapping[c] for c in y_test])
    return X_train, X_test, y_train_str, y_test_str


@pytest.fixture()
def binary_data():
    """Simple binary classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TREE_PARAMS = dict(max_leaves=20, max_depth=5)
FOREST_PARAMS = dict(n_estimators=30, max_leaves=20, max_depth=5, random_state=42)


def _fit_predict(estimator_cls, params, X_train, y_train, X_test):
    """Fit an estimator and return (predictions, probabilities)."""
    model = estimator_cls(**params)
    model.fit(X_train, y_train)
    return model, model.predict(X_test), model.predict_proba(X_test)


# ---------------------------------------------------------------------------
# Tests: class label consistency
# ---------------------------------------------------------------------------


class TestClassMapping:
    """Predictions use the original class labels, not internal cat_N codes."""

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_predictions_are_subset_of_original_classes(self, iris_data, cls, params):
        X_train, X_test, y_train, y_test = iris_data
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        assert set(y_pred).issubset(set(y_train))

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_predicted_classes_dtype_matches_input(self, iris_data, cls, params):
        X_train, X_test, y_train, y_test = iris_data
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        assert y_pred.dtype == y_train.dtype

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_string_labels_preserved(self, iris_string_labels, cls, params):
        X_train, X_test, y_train_str, y_test_str = iris_string_labels
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train_str, X_test)
        valid = {"setosa", "versicolor", "virginica"}
        assert set(y_pred).issubset(valid)

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_binary_labels(self, binary_data, cls, params):
        X_train, X_test, y_train, y_test = binary_data
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        assert set(y_pred).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests: probability shape and ordering
# ---------------------------------------------------------------------------


class TestProbabilities:
    """predict_proba returns valid, correctly ordered probability arrays."""

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_proba_shape_matches_classes(self, iris_data, cls, params):
        X_train, X_test, y_train, _ = iris_data
        model, _, proba = _fit_predict(cls, params, X_train, y_train, X_test)
        assert proba.shape == (len(X_test), len(model.classes_))

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_proba_rows_sum_to_one(self, iris_data, cls, params):
        X_train, X_test, y_train, _ = iris_data
        _, _, proba = _fit_predict(cls, params, X_train, y_train, X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_argmax_proba_equals_predict(self, iris_data, cls, params):
        """predict() must return the class with highest predict_proba()."""
        X_train, X_test, y_train, _ = iris_data
        model, y_pred, proba = _fit_predict(cls, params, X_train, y_train, X_test)
        expected = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(y_pred, expected)


# ---------------------------------------------------------------------------
# Tests: better-than-random accuracy (guards against permuted classes)
# ---------------------------------------------------------------------------


class TestAccuracy:
    """Classifiers must beat random on easy datasets.

    A systematic class permutation (the original bug) would yield ~0%
    accuracy on Iris and ~50% on a binary task, so a simple accuracy
    threshold catches any remaining mapping errors.
    """

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_iris_accuracy_above_80_percent(self, iris_data, cls, params):
        X_train, X_test, y_train, y_test = iris_data
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc > 0.80, f"Accuracy {acc:.2%} too low — possible class permutation"

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_binary_accuracy_above_75_percent(self, binary_data, cls, params):
        X_train, X_test, y_train, y_test = binary_data
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc > 0.75, f"Accuracy {acc:.2%} too low — possible class permutation"

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_string_labels_accuracy_above_80_percent(
        self, iris_string_labels, cls, params
    ):
        X_train, X_test, y_train, y_test = iris_string_labels
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc > 0.80, f"Accuracy {acc:.2%} too low — possible class permutation"


# ---------------------------------------------------------------------------
# Tests: shuffled data ordering (guards against first-seen-order bugs)
# ---------------------------------------------------------------------------


class TestDataOrdering:
    """Class mapping must be independent of row ordering in the data."""

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_shuffled_train_data_same_accuracy(self, iris_data, cls, params):
        """Shuffling training rows must not break class mapping."""
        X_train, X_test, y_train, y_test = iris_data

        # Fit on original order
        _, y_pred_orig, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        acc_orig = accuracy_score(y_test, y_pred_orig)

        # Fit on shuffled order
        rng = np.random.RandomState(99)
        idx = rng.permutation(len(y_train))
        _, y_pred_shuf, _ = _fit_predict(
            cls, params, X_train[idx], y_train[idx], X_test
        )
        acc_shuf = accuracy_score(y_test, y_pred_shuf)

        # Both must beat random — a broken mapping gives ~0%
        assert acc_orig > 0.80
        assert acc_shuf > 0.80

    @pytest.mark.parametrize(
        "cls,params",
        [
            (PartitionTreeClassifier, TREE_PARAMS),
            (PartitionForestClassifier, FOREST_PARAMS),
        ],
        ids=["tree", "forest"],
    )
    def test_reversed_class_order_same_accuracy(self, cls, params):
        """When the first sample seen belongs to the last class, mapping
        must still be correct (catches insertion-order physical code bugs)."""
        X, y = load_iris(return_X_y=True)
        # Reverse so class 2 is seen first
        X_rev, y_rev = X[::-1].copy(), y[::-1].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X_rev, y_rev, test_size=0.3, random_state=42
        )
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc > 0.80, f"Accuracy {acc:.2%} with reversed class order"


# ---------------------------------------------------------------------------
# Tests: many classes (wine dataset, 3 classes with different int labels)
# ---------------------------------------------------------------------------


class TestWineDataset:
    """Verify correct mapping on Wine (classes 0/1/2, different feature space).

    The accuracy threshold is intentionally conservative (>50%) because the
    purpose of this test is to guard against class *permutation* bugs (which
    would yield ~33% for a 3-class cyclic permutation), not to benchmark peak
    accuracy.  ``PartitionTreeClassifier`` with ``max_leaves=20`` is capacity-
    limited on Wine and achieves ~65 %; ``PartitionForestClassifier`` exceeds
    70 % comfortably.  Both are well above the permutation baseline of ~33 %.
    """

    @pytest.mark.parametrize(
        "cls,params,threshold",
        [
            (PartitionTreeClassifier, TREE_PARAMS, 0.55),
            (PartitionForestClassifier, FOREST_PARAMS, 0.70),
        ],
        ids=["tree", "forest"],
    )
    def test_wine_accuracy_above_threshold(self, cls, params, threshold):
        X, y = load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        _, y_pred, _ = _fit_predict(cls, params, X_train, y_train, X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc > threshold, (
            f"Wine accuracy {acc:.2%} too low (threshold {threshold:.0%}) — "
            "possible class permutation bug"
        )
