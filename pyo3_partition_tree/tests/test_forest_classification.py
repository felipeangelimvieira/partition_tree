"""Tests verifying that PyPartitionForest properly handles classification datasets.

Covers multi-class and binary classification, different loss functions, sample
weights, feature importances, train/test generalisation, pickling, probability
distribution consistency, and log-loss comparisons against baselines.
"""

import pickle

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split

from pyo3_partition_tree import PyPartitionForest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_polars_x_y(X: np.ndarray, y: np.ndarray, feature_prefix: str = "f"):
    """Convert numpy arrays to polars DataFrames suitable for the partition tree API."""
    x_df = pl.DataFrame(
        {f"{feature_prefix}{i}": X[:, i].tolist() for i in range(X.shape[1])}
    ).cast(pl.Float64)
    y_df = pl.DataFrame({"y": y.astype(float).tolist()}).cast(pl.Float64)
    return x_df, y_df


def _fit_forest(x_df: pl.DataFrame, y_df: pl.DataFrame, **kwargs) -> PyPartitionForest:
    """Fit a forest with sensible classification defaults."""
    defaults = dict(
        n_estimators=20,
        max_leaves=100,
        boundaries_expansion_factor=0.1,
        min_samples_split=2,
        min_samples_y=0.0,
        min_samples_x=0.0,
        min_samples_xy=1.0,
        min_volume_fraction=0.0,
        max_depth=15,
        min_gain=0.0,
        max_samples=0.8,
        max_features=0.8,
        seed=42,
    )
    defaults.update(kwargs)
    model = PyPartitionForest(**defaults)
    model.fit(x_df, y_df, None)
    return model


def _classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy by rounding predictions to the nearest integer class."""
    y_pred_rounded = np.round(y_pred).astype(int)
    return float(np.mean(y_true == y_pred_rounded))


def _extract_class_probabilities(dist, n_classes: int) -> np.ndarray:
    """Extract per-class probabilities from a piecewise constant distribution.

    For integer class labels 0 .. n_classes-1 the class-k interval is
    defined as [k - 0.5, k + 0.5].  The probability assigned to class k
    equals the integral of the density over the overlap of each segment
    with that interval, summed over all segments, and then normalised so
    the probabilities sum to 1.

    Parameters
    ----------
    dist : PyPiecewiseDistribution
        A distribution object returned by predict_proba.
    n_classes : int
        Number of distinct classes.

    Returns
    -------
    np.ndarray of shape (n_classes,)
        Normalised class probabilities.
    """
    segments = dist.pdf_segments()
    class_mass = np.zeros(n_classes)
    for density, low, high in segments:
        for k in range(n_classes):
            class_low = k - 0.5
            class_high = k + 0.5
            overlap_low = max(low, class_low)
            overlap_high = min(high, class_high)
            if overlap_high > overlap_low:
                class_mass[k] += density * (overlap_high - overlap_low)

    total = class_mass.sum()
    if total > 0:
        class_mass /= total
    else:
        class_mass[:] = 1.0 / n_classes
    return class_mass


def _log_loss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-15) -> float:
    """Compute multi-class log loss (cross-entropy).

    Parameters
    ----------
    y_true : array of int, shape (n_samples,)
        True class labels.
    proba : array, shape (n_samples, n_classes)
        Predicted class probabilities.
    eps : float
        Clipping bound to avoid log(0).

    Returns
    -------
    float
        Mean negative log-likelihood.
    """
    proba_clipped = np.clip(proba, eps, 1.0 - eps)
    n = len(y_true)
    ll = 0.0
    for i in range(n):
        ll -= np.log(proba_clipped[i, int(y_true[i])])
    return ll / n


# ---------------------------------------------------------------------------
# Basic fit / predict on classification datasets
# ---------------------------------------------------------------------------


class TestForestClassificationFitPredict:
    """Verify basic fit and predict mechanics on classification data."""

    def test_iris_predict_shape_and_finiteness(self):
        """Predict returns a DataFrame with correct shape and finite values."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        assert isinstance(preds, pl.DataFrame)
        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_wine_predict_shape_and_finiteness(self):
        """Wine dataset (3-class, 13 features)."""
        X, y = load_wine(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_breast_cancer_binary_predict(self):
        """Binary classification on breast cancer (2-class, 30 features)."""
        X, y = load_breast_cancer(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_synthetic_multiclass_predict(self):
        """Synthetic 5-class dataset."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=6,
            n_classes=5,
            n_clusters_per_class=1,
            random_state=42,
        )
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_predictions_differ_between_classes(self):
        """Iris class-0 samples should get different predictions from class-2."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        preds_np = preds.to_series(0).to_numpy()
        mean_class0 = np.mean(preds_np[:50])
        mean_class2 = np.mean(preds_np[100:])
        assert abs(mean_class0 - mean_class2) > 0.5, (
            f"Mean predictions for class 0 ({mean_class0:.2f}) and class 2 "
            f"({mean_class2:.2f}) should be clearly separated"
        )


# ---------------------------------------------------------------------------
# Train / test generalisation
# ---------------------------------------------------------------------------


class TestForestClassificationGeneralisation:
    """Verify that the forest generalises to held-out classification data."""

    def test_iris_train_test_accuracy(self):
        """Iris accuracy on held-out data should be well above chance (>60%)."""
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
        assert acc >= 0.60, f"Iris test accuracy {acc:.2%} is too low (expected >=60%)"

    def test_wine_train_test_accuracy(self):
        """Wine accuracy on held-out data should be well above chance (>50%)."""
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
        assert acc >= 0.50, f"Wine test accuracy {acc:.2%} is too low (expected >=50%)"

    def test_breast_cancer_train_test_accuracy(self):
        """Binary classification accuracy on held-out data >=70%."""
        X, y = load_breast_cancer(return_X_y=True)
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
        assert (
            acc >= 0.70
        ), f"Breast cancer test accuracy {acc:.2%} is too low (expected >=70%)"

    def test_synthetic_binary_generalisation(self):
        """Synthetic binary classification should generalise above chance."""
        X, y = make_classification(
            n_samples=600,
            n_features=8,
            n_informative=5,
            n_classes=2,
            random_state=7,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=7,
        )
        x_train_df, y_train_df = _to_polars_x_y(X_train, y_train)
        x_test_df, _ = _to_polars_x_y(X_test, y_test)

        model = _fit_forest(x_train_df, y_train_df)
        preds = model.predict(x_test_df)

        y_pred = preds.to_series(0).to_numpy()
        acc = _classification_accuracy(y_test, y_pred)
        assert (
            acc >= 0.55
        ), f"Synthetic binary test accuracy {acc:.2%} is below chance threshold"


# ---------------------------------------------------------------------------
# predict_proba distributions on classification data
# ---------------------------------------------------------------------------


class TestForestClassificationPredictProba:
    """Verify predict_proba returns valid, meaningful distributions for
    classification datasets."""

    def test_predict_proba_length_matches_samples(self):
        """One distribution object per sample."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba = model.predict_proba(x_df)

        assert len(proba) == x_df.height

    def test_predict_proba_total_mass_positive(self):
        """Every distribution should have positive total mass."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba = model.predict_proba(x_df)

        for i, dist in enumerate(proba):
            assert (
                dist.total_mass() > 0
            ), f"Distribution for sample {i} has non-positive mass"

    def test_predict_proba_n_cells_positive(self):
        """Every distribution should contain at least one cell."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba = model.predict_proba(x_df)

        for i, dist in enumerate(proba):
            assert dist.n_cells() >= 1, f"Distribution for sample {i} has 0 cells"

    def test_predict_proba_mean_close_to_true_class(self):
        """Mean of the distribution for training samples should be close to
        the true class label (as a floating-point value)."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba = model.predict_proba(x_df)

        # Check a subset: first 5 samples (class 0) and last 5 (class 2)
        for i in range(5):
            mean_dict = proba[i].mean()
            mean_val = list(mean_dict.values())[0][0]
            assert (
                abs(mean_val - 0.0) < 1.0
            ), f"Sample {i} (class 0): mean {mean_val:.3f} too far from 0.0"

        for i in range(145, 150):
            mean_dict = proba[i].mean()
            mean_val = list(mean_dict.values())[0][0]
            assert (
                abs(mean_val - 2.0) < 1.0
            ), f"Sample {i} (class 2): mean {mean_val:.3f} too far from 2.0"

    def test_predict_proba_means_differ_between_classes(self):
        """Distribution means for class-0 and class-2 samples should differ."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba = model.predict_proba(x_df)

        mean_0 = list(proba[0].mean().values())[0][0]
        mean_2 = list(proba[100].mean().values())[0][0]
        assert abs(mean_0 - mean_2) > 0.5, (
            f"Distribution means for class 0 ({mean_0:.3f}) and "
            f"class 2 ({mean_2:.3f}) should differ substantially"
        )

    def test_predict_proba_pdf_segments_non_empty(self):
        """pdf_segments should return at least one segment for each sample."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba = model.predict_proba(x_df)

        for i in [0, 50, 100]:
            segments = proba[i].pdf_segments()
            assert len(segments) > 0, f"Sample {i} has no pdf segments"
            for density, low, high in segments:
                assert np.isfinite(density)
                assert np.isfinite(low)
                assert np.isfinite(high)
                assert low <= high

    def test_predict_proba_pdf_densities_non_negative(self):
        """All density values in pdf_segments should be non-negative."""
        X, y = load_breast_cancer(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba = model.predict_proba(x_df)

        for i in range(0, x_df.height, 50):  # sample every 50th row
            for density, low, high in proba[i].pdf_segments():
                assert density >= 0, f"Sample {i} has negative density {density}"


# ---------------------------------------------------------------------------
# predict_trees_proba consistency
# ---------------------------------------------------------------------------


class TestForestClassificationPerTreeProba:
    """Verify per-tree probability distributions on classification data."""

    def test_per_tree_proba_shape(self):
        """predict_trees_proba returns n_estimators lists, each of len(x) dists."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        n_est = 10
        model = _fit_forest(x_df, y_df, n_estimators=n_est)
        per_tree = model.predict_trees_proba(x_df)

        assert len(per_tree) == n_est
        for tree_dists in per_tree:
            assert len(tree_dists) == x_df.height

    def test_per_tree_means_are_finite(self):
        """Every individual tree's mean prediction should be finite."""
        X, y = load_wine(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=5)
        per_tree = model.predict_trees_proba(x_df)

        for t_idx, tree_dists in enumerate(per_tree):
            for s_idx, dist in enumerate(tree_dists):
                mean_dict = dist.mean()
                mean_val = list(mean_dict.values())[0][0]
                assert np.isfinite(
                    mean_val
                ), f"Tree {t_idx}, sample {s_idx}: non-finite mean {mean_val}"

    def test_per_tree_predictions_diverse_on_classification(self):
        """Individual trees should yield different mean predictions,
        confirming randomization works on classification data."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=10)
        per_tree = model.predict_trees_proba(x_df)

        means_per_tree = []
        for tree_dists in per_tree:
            tree_means = np.array([list(d.mean().values())[0][0] for d in tree_dists])
            means_per_tree.append(tree_means)

        all_same = all(np.allclose(means_per_tree[0], m) for m in means_per_tree[1:])
        assert (
            not all_same
        ), "All trees produced identical mean predictions on Iris classification"


# ---------------------------------------------------------------------------
# Loss functions on classification data
# ---------------------------------------------------------------------------


class TestForestClassificationLossFunctions:
    """Verify that different loss functions work on classification datasets."""

    @pytest.mark.parametrize(
        "loss_name", ["conditional_log_loss", "balanced_log_loss", "mise"]
    )
    def test_loss_function_produces_valid_predictions(self, loss_name):
        """Each supported loss should produce finite predictions on Iris."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, loss=loss_name)
        preds = model.predict(x_df)

        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    @pytest.mark.parametrize(
        "loss_name", ["conditional_log_loss", "balanced_log_loss", "mise"]
    )
    def test_loss_function_predict_proba_valid(self, loss_name):
        """predict_proba should work with every supported loss."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, loss=loss_name)
        proba = model.predict_proba(x_df)

        assert len(proba) == x_df.height
        for dist in proba[:5]:
            assert dist.total_mass() > 0

    def test_balanced_log_loss_on_binary(self):
        """balanced_log_loss should handle binary classification."""
        X, y = load_breast_cancer(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, loss="balanced_log_loss")
        preds = model.predict(x_df)

        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_invalid_loss_raises(self):
        """An unrecognized loss string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid loss function"):
            PyPartitionForest(loss="nonexistent_loss")


# ---------------------------------------------------------------------------
# Sample weights on classification data
# ---------------------------------------------------------------------------


class TestForestClassificationSampleWeights:
    """Verify that sample weights affect the forest on classification data."""

    def test_sample_weights_accepted_and_predictions_finite(self):
        """The forest should accept custom sample weights on classification
        data and produce finite predictions and valid distributions."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        weights = [10.0 if yi == 0 else 0.5 for yi in y]
        w_df = pl.DataFrame({"sample_weights": weights}).cast(pl.Float64)

        model = PyPartitionForest(
            n_estimators=10,
            max_leaves=100,
            boundaries_expansion_factor=0.1,
            min_samples_split=2,
            min_samples_y=0.0,
            min_samples_x=0.0,
            min_samples_xy=1.0,
            min_volume_fraction=0.0,
            max_depth=15,
            min_gain=0.0,
            max_samples=0.8,
            max_features=0.8,
            seed=42,
        )
        model.fit(x_df, y_df, w_df)

        preds = model.predict(x_df)
        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

        proba = model.predict_proba(x_df)
        assert len(proba) == x_df.height
        for dist in proba[:10]:
            assert dist.total_mass() > 0
            mean_dict = dist.mean()
            mean_val = list(mean_dict.values())[0][0]
            assert np.isfinite(mean_val)

    def test_fit_with_sample_weights_produces_finite_predictions(self):
        """Custom sample weights should not produce NaN/Inf."""
        X, y = load_wine(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        weights = np.random.default_rng(42).uniform(0.5, 2.0, size=len(y))
        w_df = pl.DataFrame({"sample_weights": weights.tolist()}).cast(pl.Float64)

        model = PyPartitionForest(
            n_estimators=10,
            max_leaves=100,
            boundaries_expansion_factor=0.1,
            min_samples_split=2,
            min_samples_y=0.0,
            min_samples_x=0.0,
            min_samples_xy=1.0,
            min_volume_fraction=0.0,
            max_depth=15,
            min_gain=0.0,
            max_samples=0.8,
            max_features=0.8,
            seed=42,
        )
        model.fit(x_df, y_df, w_df)
        preds = model.predict(x_df)

        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))


# ---------------------------------------------------------------------------
# Feature importances on classification data
# ---------------------------------------------------------------------------


class TestForestClassificationFeatureImportances:
    """Verify feature importances on classification datasets."""

    def test_feature_importances_keys_include_feature_columns(self):
        """Importance dict keys should include all feature column names.

        The model may also report a target-space key (e.g. ``target_y``);
        we verify that every feature column appears in the importance dict.
        """
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        importances = model.get_feature_importances(normalize=True)

        for col in x_df.columns:
            assert col in importances, f"Feature '{col}' missing from importances"

    def test_feature_importances_sum_to_one_when_normalized(self):
        """Normalized importances should sum to approximately 1.0."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        importances = model.get_feature_importances(normalize=True)

        total = sum(importances.values())
        assert (
            abs(total - 1.0) < 1e-6
        ), f"Normalized importances sum to {total}, expected 1.0"

    def test_feature_importances_non_negative(self):
        """All importance values should be non-negative."""
        X, y = load_wine(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        importances = model.get_feature_importances(normalize=False)

        for name, val in importances.items():
            assert val >= 0, f"Feature '{name}' has negative importance {val}"

    def test_unnormalized_importances_are_positive(self):
        """Unnormalized importances should be strictly positive (on a real dataset)."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        importances = model.get_feature_importances(normalize=False)

        total = sum(importances.values())
        assert total > 0, "Total unnormalized importance should be positive"


# ---------------------------------------------------------------------------
# Pickling a forest fitted on classification data
# ---------------------------------------------------------------------------


class TestForestClassificationPickle:
    """Verify that a forest fitted on classification data survives pickling."""

    def test_pickle_roundtrip_preserves_predictions(self):
        """Predictions before and after pickling must be identical."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds_before = model.predict(x_df)

        pickled = pickle.dumps(model)
        restored = pickle.loads(pickled)
        preds_after = restored.predict(x_df)

        assert preds_before.equals(
            preds_after
        ), "Predictions changed after pickle roundtrip"

    def test_pickle_roundtrip_preserves_proba(self):
        """predict_proba distribution means should match after pickle roundtrip."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=5)
        proba_before = model.predict_proba(x_df)
        means_before = [list(d.mean().values())[0][0] for d in proba_before]

        pickled = pickle.dumps(model)
        restored = pickle.loads(pickled)
        proba_after = restored.predict_proba(x_df)
        means_after = [list(d.mean().values())[0][0] for d in proba_after]

        assert np.allclose(
            means_before, means_after
        ), "Distribution means changed after pickle roundtrip"

    def test_pickle_roundtrip_preserves_feature_importances(self):
        """Feature importances should survive pickling (within float tolerance)."""
        X, y = load_wine(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        imp_before = model.get_feature_importances(normalize=True)

        pickled = pickle.dumps(model)
        restored = pickle.loads(pickled)
        imp_after = restored.get_feature_importances(normalize=True)

        assert set(imp_before.keys()) == set(imp_after.keys())
        for key in imp_before:
            assert abs(imp_before[key] - imp_after[key]) < 1e-10, (
                f"Importance for '{key}' changed after pickle: "
                f"{imp_before[key]} vs {imp_after[key]}"
            )


# ---------------------------------------------------------------------------
# Nodes & split history on classification data
# ---------------------------------------------------------------------------


class TestForestClassificationStructure:
    """Verify structural introspection APIs work on classification datasets."""

    def test_nodes_info_non_empty_per_tree(self):
        """Each tree in the forest should have at least one node."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=5)
        all_nodes = model.get_nodes_info()

        assert len(all_nodes) == 5
        for tree_nodes in all_nodes:
            assert len(tree_nodes) >= 1

    def test_split_history_non_empty_per_tree(self):
        """Each tree should have at least one split on a non-trivial dataset."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=5)
        all_splits = model.get_split_history()

        assert len(all_splits) == 5
        for tree_splits in all_splits:
            assert len(tree_splits) >= 1, "Expected at least one split per tree"

    def test_split_history_columns_present(self):
        """Each split record should contain expected keys."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=3)
        all_splits = model.get_split_history()

        expected_keys = {
            "parent_index",
            "col_name",
            "split_kind",
            "gain",
            "left_child_index",
            "right_child_index",
        }
        for tree_splits in all_splits:
            for rec in tree_splits:
                assert set(rec.keys()) == expected_keys

    def test_node_records_contain_expected_keys(self):
        """Each node dict should contain the documented keys."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=2)
        all_nodes = model.get_nodes_info()

        expected_keys = {
            "node_index",
            "is_leaf",
            "depth",
            "parent",
            "left_child",
            "right_child",
            "w_xy",
            "w_x",
            "w_y",
            "conditional_density",
            "partitions",
        }
        for tree_nodes in all_nodes:
            for node in tree_nodes:
                assert expected_keys.issubset(
                    set(node.keys())
                ), f"Missing keys: {expected_keys - set(node.keys())}"


# ---------------------------------------------------------------------------
# Imbalanced classification
# ---------------------------------------------------------------------------


class TestForestClassificationImbalanced:
    """Verify the forest handles class-imbalanced datasets."""

    def test_imbalanced_binary_predictions_finite(self):
        """Even with 95/5 class split, predictions should be finite."""
        X, y = make_classification(
            n_samples=400,
            n_features=6,
            n_informative=4,
            n_classes=2,
            weights=[0.95, 0.05],
            random_state=42,
        )
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_imbalanced_multiclass_predictions_finite(self):
        """Imbalanced 3-class dataset should yield finite predictions."""
        X, y = make_classification(
            n_samples=600,
            n_features=8,
            n_informative=5,
            n_classes=3,
            weights=[0.7, 0.2, 0.1],
            n_clusters_per_class=1,
            random_state=42,
        )
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_balanced_loss_on_imbalanced_data(self):
        """balanced_log_loss should work on imbalanced data without errors."""
        X, y = make_classification(
            n_samples=400,
            n_features=6,
            n_informative=4,
            n_classes=2,
            weights=[0.9, 0.1],
            random_state=42,
        )
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, loss="balanced_log_loss")
        preds = model.predict(x_df)

        assert preds.height == x_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestForestClassificationEdgeCases:
    """Edge cases for classification datasets."""

    def test_two_class_two_feature_small_dataset(self):
        """Minimal viable classification dataset."""
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.1, 0.1],
                [0.9, 0.9],
                [0.1, 0.9],
                [0.9, 0.1],
            ]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=5, max_leaves=10)
        preds = model.predict(x_df)

        assert preds.height == 8
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_single_class_dataset(self):
        """When all targets are the same class, the model should still work."""
        X = np.random.default_rng(42).standard_normal((50, 3))
        y = np.zeros(50)  # all class 0
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df, n_estimators=5)
        preds = model.predict(x_df)

        assert preds.height == 50
        preds_np = preds.to_series(0).to_numpy()
        assert np.all(np.isfinite(preds_np))
        # With only one class, all predictions should be close to 0
        assert np.all(np.abs(preds_np) < 1.0)

    def test_predict_on_unseen_range(self):
        """Predict on data outside the training feature range."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)

        # Create test data with extreme values
        X_extreme = X.copy()
        X_extreme[:10] = X.max(axis=0) * 3  # far above training range
        X_extreme[10:20] = X.min(axis=0) * 3  # far below training range
        x_extreme_df, _ = _to_polars_x_y(X_extreme, y)

        preds = model.predict(x_extreme_df)
        assert preds.height == x_extreme_df.height
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))

    def test_many_classes(self):
        """Classification with 10 classes."""
        X, y = make_classification(
            n_samples=1000,
            n_features=15,
            n_informative=10,
            n_classes=10,
            n_clusters_per_class=1,
            random_state=42,
        )
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df)

        assert preds.height == 1000
        assert np.all(np.isfinite(preds.to_series(0).to_numpy()))
        # Predictions should not be constant
        unique_preds = len(set(preds.to_series(0).to_list()))
        assert unique_preds > 1


# ---------------------------------------------------------------------------
# Log-loss: forest probabilities vs. baselines
# ---------------------------------------------------------------------------


class TestForestClassificationLogLoss:
    """Verify that the forest's predicted class probabilities yield a
    lower log-loss than naive baselines (uniform and class-prior)."""

    # -- In-sample (training) log-loss -----------------------------------

    def test_iris_train_log_loss_beats_uniform(self):
        """On Iris training data, forest log-loss < uniform baseline."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba_dists = model.predict_proba(x_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y, proba)

        uniform_proba = np.full((len(y), n_classes), 1.0 / n_classes)
        uniform_ll = _log_loss(y, uniform_proba)

        assert forest_ll < uniform_ll, (
            f"Forest train log-loss ({forest_ll:.4f}) should be lower than "
            f"uniform baseline ({uniform_ll:.4f})"
        )

    def test_iris_train_log_loss_beats_prior(self):
        """On Iris training data, forest log-loss < class-prior baseline."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba_dists = model.predict_proba(x_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y, proba)

        # Prior baseline: predict training class frequencies
        prior = np.bincount(y.astype(int), minlength=n_classes) / len(y)
        prior_proba = np.tile(prior, (len(y), 1))
        prior_ll = _log_loss(y, prior_proba)

        assert forest_ll < prior_ll, (
            f"Forest train log-loss ({forest_ll:.4f}) should be lower than "
            f"class-prior baseline ({prior_ll:.4f})"
        )

    def test_wine_train_log_loss_beats_uniform(self):
        """On Wine training data, forest log-loss < uniform baseline."""
        X, y = load_wine(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba_dists = model.predict_proba(x_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y, proba)

        uniform_proba = np.full((len(y), n_classes), 1.0 / n_classes)
        uniform_ll = _log_loss(y, uniform_proba)

        assert forest_ll < uniform_ll, (
            f"Wine forest train log-loss ({forest_ll:.4f}) should be lower "
            f"than uniform baseline ({uniform_ll:.4f})"
        )

    def test_breast_cancer_train_log_loss_beats_uniform(self):
        """On breast cancer (binary), forest log-loss < uniform baseline."""
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba_dists = model.predict_proba(x_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y, proba)

        uniform_proba = np.full((len(y), n_classes), 1.0 / n_classes)
        uniform_ll = _log_loss(y, uniform_proba)

        assert forest_ll < uniform_ll, (
            f"Breast cancer forest train log-loss ({forest_ll:.4f}) should be "
            f"lower than uniform baseline ({uniform_ll:.4f})"
        )

    def test_breast_cancer_train_log_loss_beats_prior(self):
        """On breast cancer (binary), forest log-loss < class-prior baseline."""
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba_dists = model.predict_proba(x_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y, proba)

        prior = np.bincount(y.astype(int), minlength=n_classes) / len(y)
        prior_proba = np.tile(prior, (len(y), 1))
        prior_ll = _log_loss(y, prior_proba)

        assert forest_ll < prior_ll, (
            f"Breast cancer forest train log-loss ({forest_ll:.4f}) should be "
            f"lower than prior baseline ({prior_ll:.4f})"
        )

    # -- Out-of-sample (test set) log-loss --------------------------------

    def test_iris_test_log_loss_beats_uniform(self):
        """On held-out Iris data, forest log-loss < uniform baseline."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
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
        proba_dists = model.predict_proba(x_test_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y_test, proba)

        uniform_proba = np.full((len(y_test), n_classes), 1.0 / n_classes)
        uniform_ll = _log_loss(y_test, uniform_proba)

        assert forest_ll < uniform_ll, (
            f"Iris test log-loss ({forest_ll:.4f}) should be lower than "
            f"uniform baseline ({uniform_ll:.4f})"
        )

    def test_iris_test_log_loss_beats_prior(self):
        """On held-out Iris data, forest log-loss < class-prior baseline."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
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
        proba_dists = model.predict_proba(x_test_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y_test, proba)

        # Use training set prior as baseline
        prior = np.bincount(y_train.astype(int), minlength=n_classes) / len(y_train)
        prior_proba = np.tile(prior, (len(y_test), 1))
        prior_ll = _log_loss(y_test, prior_proba)

        assert forest_ll < prior_ll, (
            f"Iris test log-loss ({forest_ll:.4f}) should be lower than "
            f"prior baseline ({prior_ll:.4f})"
        )

    def test_wine_test_log_loss_beats_uniform(self):
        """On held-out Wine data, forest log-loss < uniform baseline."""
        X, y = load_wine(return_X_y=True)
        n_classes = len(np.unique(y))
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
        proba_dists = model.predict_proba(x_test_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y_test, proba)

        uniform_proba = np.full((len(y_test), n_classes), 1.0 / n_classes)
        uniform_ll = _log_loss(y_test, uniform_proba)

        assert forest_ll < uniform_ll, (
            f"Wine test log-loss ({forest_ll:.4f}) should be lower than "
            f"uniform baseline ({uniform_ll:.4f})"
        )

    def test_breast_cancer_test_log_loss_beats_uniform(self):
        """On held-out breast cancer, forest log-loss < uniform baseline."""
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = len(np.unique(y))
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
        proba_dists = model.predict_proba(x_test_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y_test, proba)

        uniform_proba = np.full((len(y_test), n_classes), 1.0 / n_classes)
        uniform_ll = _log_loss(y_test, uniform_proba)

        assert forest_ll < uniform_ll, (
            f"Breast cancer test log-loss ({forest_ll:.4f}) should be lower "
            f"than uniform baseline ({uniform_ll:.4f})"
        )

    def test_breast_cancer_test_log_loss_beats_prior(self):
        """On held-out breast cancer, forest log-loss < class-prior baseline."""
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = len(np.unique(y))
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
        proba_dists = model.predict_proba(x_test_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y_test, proba)

        prior = np.bincount(y_train.astype(int), minlength=n_classes) / len(y_train)
        prior_proba = np.tile(prior, (len(y_test), 1))
        prior_ll = _log_loss(y_test, prior_proba)

        assert forest_ll < prior_ll, (
            f"Breast cancer test log-loss ({forest_ll:.4f}) should be lower "
            f"than prior baseline ({prior_ll:.4f})"
        )

    # -- Per-loss-function comparisons ------------------------------------

    @pytest.mark.parametrize(
        "loss_name", ["conditional_log_loss", "balanced_log_loss", "mise"]
    )
    def test_loss_functions_beat_uniform_on_iris(self, loss_name):
        """Each loss function's probabilities should beat uniform on Iris."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
        x_train_df, y_train_df = _to_polars_x_y(X_train, y_train)
        x_test_df, _ = _to_polars_x_y(X_test, y_test)

        model = _fit_forest(x_train_df, y_train_df, loss=loss_name)
        proba_dists = model.predict_proba(x_test_df)

        proba = np.array(
            [_extract_class_probabilities(d, n_classes) for d in proba_dists]
        )
        forest_ll = _log_loss(y_test, proba)

        uniform_proba = np.full((len(y_test), n_classes), 1.0 / n_classes)
        uniform_ll = _log_loss(y_test, uniform_proba)

        assert forest_ll < uniform_ll, (
            f"Forest with loss={loss_name}: test log-loss ({forest_ll:.4f}) "
            f"should be lower than uniform ({uniform_ll:.4f})"
        )

    # -- Probability calibration sanity checks ----------------------------

    def test_class_probabilities_sum_to_one(self):
        """Extracted class probabilities should sum to 1 for every sample."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba_dists = model.predict_proba(x_df)

        for i, dist in enumerate(proba_dists):
            p = _extract_class_probabilities(dist, n_classes)
            assert (
                abs(p.sum() - 1.0) < 1e-10
            ), f"Sample {i}: class probabilities sum to {p.sum()}, expected 1.0"

    def test_class_probabilities_non_negative(self):
        """All extracted class probabilities should be non-negative."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        proba_dists = model.predict_proba(x_df)

        for i, dist in enumerate(proba_dists):
            p = _extract_class_probabilities(dist, n_classes)
            assert np.all(p >= 0), f"Sample {i}: negative class probability {p}"

    def test_highest_probability_class_matches_prediction(self):
        """The argmax of class probabilities should generally agree with
        the rounded point prediction for most training samples."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model = _fit_forest(x_df, y_df)
        preds = model.predict(x_df).to_series(0).to_numpy()
        proba_dists = model.predict_proba(x_df)

        pred_classes = np.round(preds).astype(int).clip(0, n_classes - 1)
        proba_classes = np.array(
            [np.argmax(_extract_class_probabilities(d, n_classes)) for d in proba_dists]
        )
        agreement = np.mean(pred_classes == proba_classes)
        assert agreement > 0.5, (
            f"Point predictions and argmax(proba) agree only {agreement:.1%} "
            f"of the time — expected majority agreement"
        )

    def test_forest_more_trees_improves_or_matches_log_loss(self):
        """A forest with more trees should achieve equal or better log-loss
        than a small forest (on training data)."""
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
        x_df, y_df = _to_polars_x_y(X, y)

        model_small = _fit_forest(x_df, y_df, n_estimators=3)
        model_large = _fit_forest(x_df, y_df, n_estimators=50)

        proba_small = np.array(
            [
                _extract_class_probabilities(d, n_classes)
                for d in model_small.predict_proba(x_df)
            ]
        )
        proba_large = np.array(
            [
                _extract_class_probabilities(d, n_classes)
                for d in model_large.predict_proba(x_df)
            ]
        )

        ll_small = _log_loss(y, proba_small)
        ll_large = _log_loss(y, proba_large)

        # Large forest should be at least as good (allow 5% tolerance for
        # randomness)
        assert ll_large <= ll_small * 1.05, (
            f"50-tree forest log-loss ({ll_large:.4f}) should not be much "
            f"worse than 3-tree forest ({ll_small:.4f})"
        )
