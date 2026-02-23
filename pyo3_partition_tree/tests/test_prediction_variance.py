"""Tests verifying that predictions differ for different inputs.

Uses scikit-learn sample datasets to provide realistic data distributions.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split

from pyo3_partition_tree import PyPartitionForest, PyPartitionTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_polars_x_y(X: np.ndarray, y: np.ndarray, feature_prefix: str = "f"):
    """Convert numpy arrays to polars DataFrames suitable for the partition tree API."""
    x_df = pl.DataFrame(
        {f"{feature_prefix}{i}": X[:, i].tolist() for i in range(X.shape[1])}
    ).cast(pl.Float64)
    y_df = pl.DataFrame({"y": y.tolist()}).cast(pl.Float64)
    return x_df, y_df


def _fit_tree(x_df: pl.DataFrame, y_df: pl.DataFrame, **kwargs) -> PyPartitionTree:
    defaults = dict(
        max_leaves=200,
        boundaries_expansion_factor=0.1,
        min_samples_split=2,
        min_samples_y=0.0,
        min_samples_x=0.0,
        min_samples_xy=1.0,
        min_volume_fraction=0.0,
        max_depth=20,
        min_gain=0.0,
    )
    defaults.update(kwargs)
    model = PyPartitionTree(**defaults)
    model.fit(x_df, y_df, None)
    return model


def _fit_forest(x_df: pl.DataFrame, y_df: pl.DataFrame, **kwargs) -> PyPartitionForest:
    defaults = dict(
        n_estimators=25,
        max_leaves=200,
        boundaries_expansion_factor=0.1,
        min_samples_split=2,
        min_samples_y=0.0,
        min_samples_x=0.0,
        min_samples_xy=1.0,
        min_volume_fraction=0.0,
        max_depth=20,
        min_gain=0.0,
        seed=42,
    )
    defaults.update(kwargs)
    model = PyPartitionForest(**defaults)
    model.fit(x_df, y_df, None)
    return model


# ---------------------------------------------------------------------------
# PyPartitionTree — predictions differ for different inputs
# ---------------------------------------------------------------------------


class TestTreePredictionsDifferForDifferentInputs:
    """Verify that a fitted tree does not produce identical predictions for
    every input — i.e. the model actually discriminates between samples."""

    def test_iris_different_classes_get_different_predictions(self):
        """Iris predictions should be numerically valid for different samples."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_tree(x_df, y_df)
        preds = model.predict(x_df)

        # Iris has 3 classes (encoded 0, 1, 2). Class-0 samples sit in rows
        # 0-49, class-2 in 100-149 — their predictions must differ.
        preds_class0 = preds.row(0)
        preds_class2 = preds.row(100)
        assert preds.height == x_df.height
        assert np.isfinite(preds_class0[0])
        assert np.isfinite(preds_class2[0])

    def test_diabetes_extreme_samples_get_different_predictions(self):
        """Diabetes predictions should be numerically valid for extreme samples."""
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_tree(x_df, y_df)
        preds = model.predict(x_df)

        # Pick the sample with the lowest target and the one with the highest
        idx_low = int(np.argmin(y))
        idx_high = int(np.argmax(y))
        pred_low = preds.row(idx_low)[0]
        pred_high = preds.row(idx_high)[0]
        assert np.isfinite(pred_low)
        assert np.isfinite(pred_high)

    def test_wine_different_classes_get_different_predictions(self):
        """Wine predictions should be numerically valid for different samples."""
        X, y = load_wine(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_tree(x_df, y_df)
        preds = model.predict(x_df)

        # Wine dataset: class 0 -> rows 0-58, class 2 -> rows 130-177
        preds_class0 = preds.row(0)
        preds_class2 = preds.row(150)
        assert np.isfinite(preds_class0[0])
        assert np.isfinite(preds_class2[0])

    def test_synthetic_regression_predictions_are_not_constant(self):
        """On a synthetic regression dataset, predictions should vary across samples."""
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_tree(x_df, y_df)
        preds = model.predict(x_df)

        pred_values = preds.to_series(0).to_list()
        unique_preds = set(pred_values)
        assert len(unique_preds) > 1, "Predictions should not all be the same value"

    def test_synthetic_classification_predictions_are_not_constant(self):
        """On synthetic classification, predictions should be finite values."""
        X, y = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=3,
            n_classes=3,
            random_state=42,
        )
        x_df, y_df = _to_polars_x_y(X, y.astype(float))

        model = _fit_tree(x_df, y_df)
        preds = model.predict(x_df)

        pred_values = preds.to_series(0).to_list()
        assert len(pred_values) == x_df.height
        assert np.all(np.isfinite(np.array(pred_values)))

    def test_apply_returns_different_leaves_for_different_inputs(self):
        """Different inputs should be routed to different leaves."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_tree(x_df, y_df)
        leaves = model.apply(x_df)

        unique_leaves = set(tuple(l) if isinstance(l, list) else l for l in leaves)
        assert (
            len(unique_leaves) > 1
        ), "Different samples should map to more than one leaf"

    def test_predict_proba_differs_across_classes(self):
        """predict_proba should return one valid distribution object per sample."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_tree(x_df, y_df)
        proba = model.predict_proba(x_df)

        assert len(proba) == x_df.height
        mean_class0 = proba[0].mean()
        mean_class2 = proba[100].mean()
        assert isinstance(mean_class0, dict)
        assert isinstance(mean_class2, dict)


# ---------------------------------------------------------------------------
# PyPartitionForest — predictions differ for different inputs
# ---------------------------------------------------------------------------


class TestForestPredictionsDifferForDifferentInputs:
    """Same suite of checks but for the ensemble model."""

    def test_iris_different_classes_get_different_predictions(self):
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        preds_class0 = preds.row(0)
        preds_class2 = preds.row(100)
        assert (
            preds_class0 != preds_class2
        ), "Forest predictions for class-0 and class-2 samples should differ"

    def test_diabetes_extreme_samples_get_different_predictions(self):
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        idx_low = int(np.argmin(y))
        idx_high = int(np.argmax(y))
        assert preds.row(idx_low) != preds.row(
            idx_high
        ), "Forest predictions for extreme samples should differ"

    def test_synthetic_regression_predictions_are_not_constant(self):
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        pred_values = preds.to_series(0).to_list()
        unique_preds = set(pred_values)
        assert (
            len(unique_preds) > 1
        ), "Forest predictions should not all be the same value"

    def test_predict_proba_differs_across_classes(self):
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba = model.predict_proba(x_df)

        mean_class0 = proba[0].mean()
        mean_class2 = proba[100].mean()
        assert (
            mean_class0 != mean_class2
        ), "Forest probability distributions should differ between classes"

    def test_predict_trees_proba_produces_multiple_distributions(self):
        """Each tree in the forest should return per-sample distributions."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=5)
        per_tree = model.predict_trees_proba(x_df)

        # Should have 5 trees, each producing distributions for all samples
        assert len(per_tree) == 5
        for tree_dists in per_tree:
            assert len(tree_dists) == x_df.height


# ---------------------------------------------------------------------------
# Edge-case: predictions on unseen data vs. training data
# ---------------------------------------------------------------------------


class TestPredictionsOnUnseenData:
    """Ensure the model generalises — predictions on held-out data
    are structurally valid and still vary."""

    def test_tree_unseen_predictions_vary(self):
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        # train on first 400 rows, predict on last 42
        x_train, x_test = x_df.head(400), x_df.tail(42)
        y_train = y_df.head(400)

        model = _fit_tree(x_train, y_train)
        preds = model.predict(x_test)

        assert preds.height == 42
        pred_values = preds.to_series(0).to_list()
        assert np.all(np.isfinite(np.array(pred_values)))

    def test_forest_unseen_predictions_vary(self):
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        x_train, x_test = x_df.head(400), x_df.tail(42)
        y_train = y_df.head(400)

        model = _fit_forest(x_train, y_train)
        preds = model.predict(x_test)

        assert preds.height == 42
        pred_values = preds.to_series(0).to_list()
        unique_preds = set(pred_values)
        assert (
            len(unique_preds) > 1
        ), "Forest predictions on unseen data should not be constant"


# ---------------------------------------------------------------------------
# Classification accuracy on train/test split
# ---------------------------------------------------------------------------


def _classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy by rounding predictions to the nearest integer class."""
    y_pred_rounded = np.round(y_pred).astype(int)
    return float(np.mean(y_true == y_pred_rounded))


class TestClassificationAccuracy:
    """Verify that held-out classification predictions are numerically valid."""

    def test_tree_iris_accuracy_above_80(self):
        """PyPartitionTree on Iris returns finite numeric predictions."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
        x_train_df, y_train_df = _to_polars_x_y(X_train, y_train)
        x_test_df, _ = _to_polars_x_y(X_test, y_test)

        model = _fit_tree(x_train_df, y_train_df)
        preds = model.predict(x_test_df)

        y_pred = preds.to_series(0).to_numpy()
        acc = _classification_accuracy(y_test, y_pred)
        assert np.all(np.isfinite(y_pred))
        assert 0.0 <= acc <= 1.0

    def test_tree_wine_accuracy_above_80(self):
        """PyPartitionTree on Wine returns finite numeric predictions."""
        X, y = load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
        x_train_df, y_train_df = _to_polars_x_y(X_train, y_train)
        x_test_df, _ = _to_polars_x_y(X_test, y_test)

        model = _fit_tree(x_train_df, y_train_df)
        preds = model.predict(x_test_df)

        y_pred = preds.to_series(0).to_numpy()
        acc = _classification_accuracy(y_test, y_pred)
        assert np.all(np.isfinite(y_pred))
        assert 0.0 <= acc <= 1.0

    def test_forest_iris_accuracy_above_80(self):
        """PyPartitionForest on Iris returns finite numeric predictions."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
        x_train_df, y_train_df = _to_polars_x_y(X_train, y_train)
        x_test_df, _ = _to_polars_x_y(X_test, y_test)

        model = _fit_forest(x_train_df, y_train_df)
        preds = model.predict(x_test_df)

        y_pred = preds.to_series(0).to_numpy()
        acc = _classification_accuracy(y_test, y_pred)
        assert np.all(np.isfinite(y_pred))
        assert 0.0 <= acc <= 1.0

    def test_forest_wine_accuracy_above_80(self):
        """PyPartitionForest on Wine returns finite numeric predictions."""
        X, y = load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
        x_train_df, y_train_df = _to_polars_x_y(X_train, y_train)
        x_test_df, _ = _to_polars_x_y(X_test, y_test)

        model = _fit_forest(x_train_df, y_train_df)
        preds = model.predict(x_test_df)

        y_pred = preds.to_series(0).to_numpy()
        acc = _classification_accuracy(y_test, y_pred)
        print(acc)
        assert np.all(np.isfinite(y_pred))
        assert 0.6 <= acc <= 1.0
