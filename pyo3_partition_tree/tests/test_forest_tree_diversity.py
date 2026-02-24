"""Tests verifying that individual trees within a PartitionForest are different.

A random forest derives its strength from having diverse, decorrelated trees.
These tests check that the per-tree predictions, distributions, and structure
actually differ from one tree to another, confirming that randomization
(bagging / feature subsampling) is working correctly.

NOTE: Tree diversity requires at least one of ``max_samples`` (bootstrap
sampling) or ``max_features`` (feature subsampling) to be set.  Without
these, every tree receives the exact same data and feature set, so the
per-tree seed has no observable effect and all trees are identical.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import load_diabetes, load_iris, make_regression

from pyo3_partition_tree import PyPartitionForest


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


def _fit_forest(x_df: pl.DataFrame, y_df: pl.DataFrame, **kwargs) -> PyPartitionForest:
    """Fit a forest with sensible defaults for diversity testing.

    By default, ``max_samples=0.8`` and ``max_features=0.8`` are set so
    that the per-tree RNG seed actually produces different bootstrap
    samples and feature subsets.
    """
    defaults = dict(
        n_estimators=10,
        max_leaves=200,
        boundaries_expansion_factor=0.1,
        min_samples_split=2,
        min_samples_y=0.0,
        min_samples_x=0.0,
        min_samples_xy=1.0,
        min_volume_fraction=0.0,
        max_depth=20,
        min_gain=0.0,
        max_samples=0.8,
        max_features=0.8,
        seed=42,
    )
    defaults.update(kwargs)
    model = PyPartitionForest(**defaults)
    model.fit(x_df, y_df, None)
    return model


def _extract_per_tree_means(forest, x_df):
    """Return a list (one per tree) of arrays of mean predictions for every sample."""
    per_tree = forest.predict_trees_proba(x_df)
    means_per_tree = []
    for tree_dists in per_tree:
        tree_means = []
        for dist in tree_dists:
            mean_dict = dist.mean()
            # mean() returns dict[str, list[float]] — take the first target column
            first_col = next(iter(mean_dict.values()))
            tree_means.append(first_col[0])
        means_per_tree.append(np.array(tree_means))
    return means_per_tree


# ---------------------------------------------------------------------------
# Per-tree mean predictions should differ
# ---------------------------------------------------------------------------


class TestForestTreeMeanPredictionsDiffer:
    """Verify that different trees in the forest produce different mean
    predictions, confirming randomization is effective."""

    def test_iris_trees_give_different_mean_predictions(self):
        """On Iris, individual trees should not all produce the same means."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(x_df, y_df, n_estimators=10)
        means = _extract_per_tree_means(forest, x_df)

        # At least two trees must disagree on at least one sample
        all_same = all(np.allclose(means[0], m) for m in means[1:])
        assert (
            not all_same
        ), "All trees produced identical mean predictions — randomization may be broken"

    def test_diabetes_trees_give_different_mean_predictions(self):
        """On Diabetes (regression), individual trees must differ."""
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(x_df, y_df, n_estimators=10)
        means = _extract_per_tree_means(forest, x_df)

        all_same = all(np.allclose(means[0], m) for m in means[1:])
        assert not all_same, "All trees produced identical mean predictions on Diabetes"

    def test_synthetic_regression_trees_differ(self):
        """On synthetic regression with many features, trees should differ."""
        X, y = make_regression(n_samples=300, n_features=10, noise=1.0, random_state=7)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(x_df, y_df, n_estimators=10)
        means = _extract_per_tree_means(forest, x_df)

        all_same = all(np.allclose(means[0], m) for m in means[1:])
        assert (
            not all_same
        ), "All trees produced identical predictions on synthetic regression"


# ---------------------------------------------------------------------------
# Per-tree distribution structure should differ
# ---------------------------------------------------------------------------


class TestForestTreeStructureDiffers:
    """Verify structural diversity: trees differ in the number of cells,
    pdf segments, or total mass of the distributions they produce."""

    def test_trees_have_different_cell_counts(self):
        """Different trees should partition the space differently,
        leading to different numbers of distribution cells for at least
        some samples.  We use a regression dataset where the tree
        structure is less constrained by class boundaries."""
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(x_df, y_df, n_estimators=10)
        per_tree = forest.predict_trees_proba(x_df)

        # Collect n_cells for each (tree, sample) pair
        cell_counts_per_tree = []
        for tree_dists in per_tree:
            cell_counts_per_tree.append(tuple(dist.n_cells() for dist in tree_dists))

        unique_patterns = set(cell_counts_per_tree)
        assert len(unique_patterns) > 1, (
            "All trees have identical cell-count patterns across samples — "
            "trees may not be structurally diverse"
        )

    def test_trees_have_different_pdf_segments(self):
        """On a regression task, the pdf_segments for a given sample should
        differ between at least some trees."""
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(x_df, y_df, n_estimators=10)
        per_tree = forest.predict_trees_proba(x_df)

        # Compare pdf_segments for the first sample across trees
        first_sample_segments = []
        for tree_dists in per_tree:
            segments = tree_dists[0].pdf_segments()
            first_sample_segments.append(tuple(segments))

        unique_segments = set(first_sample_segments)
        assert len(unique_segments) > 1, (
            "All trees produce the same pdf_segments for sample 0 — "
            "trees are not producing diverse density estimates"
        )


# ---------------------------------------------------------------------------
# Pairwise correlation between trees should be less than perfect
# ---------------------------------------------------------------------------


class TestForestTreeDecorrelation:
    """Trees in a forest should be somewhat decorrelated. If every tree
    produced perfectly correlated predictions the ensemble would gain
    nothing over a single tree."""

    def test_pairwise_correlations_below_one(self):
        """Mean pairwise Pearson correlation between trees should be < 1."""
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(x_df, y_df, n_estimators=10)
        means = _extract_per_tree_means(forest, x_df)

        n_trees = len(means)
        correlations = []
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                r = np.corrcoef(means[i], means[j])[0, 1]
                correlations.append(r)

        avg_corr = np.mean(correlations)
        assert avg_corr < 1.0, (
            f"Average pairwise correlation is {avg_corr:.4f} — "
            "trees are perfectly correlated"
        )

    def test_at_least_some_pairs_have_moderate_decorrelation(self):
        """At least one tree pair should have correlation noticeably below 1."""
        X, y = make_regression(n_samples=300, n_features=10, noise=1.0, random_state=7)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(x_df, y_df, n_estimators=10)
        means = _extract_per_tree_means(forest, x_df)

        n_trees = len(means)
        min_corr = 1.0
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                r = np.corrcoef(means[i], means[j])[0, 1]
                min_corr = min(min_corr, r)

        assert min_corr < 0.99, (
            f"Minimum pairwise correlation is {min_corr:.4f} — "
            "all trees are nearly identical"
        )


# ---------------------------------------------------------------------------
# Different seeds should produce different forests
# ---------------------------------------------------------------------------


class TestForestSeedDiversity:
    """Changing the seed should produce a different set of trees."""

    def test_different_seeds_give_different_predictions(self):
        """Forests with different seeds should not produce identical predictions."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest_a = _fit_forest(x_df, y_df, n_estimators=5, seed=1)
        forest_b = _fit_forest(x_df, y_df, n_estimators=5, seed=999)

        preds_a = forest_a.predict(x_df)
        preds_b = forest_b.predict(x_df)

        assert not preds_a.equals(
            preds_b
        ), "Forests with different seeds produced identical predictions"

    def test_same_seed_gives_same_predictions(self):
        """Forests with the same seed should be deterministic."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest_a = _fit_forest(x_df, y_df, n_estimators=5, seed=42)
        forest_b = _fit_forest(x_df, y_df, n_estimators=5, seed=42)

        preds_a = forest_a.predict(x_df)
        preds_b = forest_b.predict(x_df)

        assert preds_a.equals(
            preds_b
        ), "Forests with the same seed should produce identical predictions"


# ---------------------------------------------------------------------------
# Ensemble predictions should differ from any single tree
# ---------------------------------------------------------------------------


class TestEnsembleVsSingleTree:
    """The aggregated forest prediction should differ from any individual
    tree's prediction, confirming that averaging/aggregation is happening."""

    def test_forest_prediction_differs_from_individual_trees(self):
        """The forest's predict output should not match any single tree exactly."""
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(x_df, y_df, n_estimators=10)
        ensemble_preds = forest.predict(x_df).to_series(0).to_numpy()

        per_tree_means = _extract_per_tree_means(forest, x_df)

        matches = sum(
            1 for tree_m in per_tree_means if np.allclose(ensemble_preds, tree_m)
        )
        assert matches == 0, (
            "Ensemble prediction is identical to one of the individual trees — "
            "aggregation may not be working"
        )


# ---------------------------------------------------------------------------
# Without randomization all trees must be identical (sanity check)
# ---------------------------------------------------------------------------


class TestForestTreesIdenticalWithoutRandomization:
    """When max_samples and max_features are both None the per-tree seed
    is never consumed, so every tree should produce the exact same
    predictions.  These tests document the expected behaviour."""

    def test_trees_identical_without_max_samples_or_max_features(self):
        """Without bootstrap or feature subsampling, all trees are clones."""
        X, y = load_iris(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(
            x_df,
            y_df,
            n_estimators=5,
            max_samples=None,
            max_features=None,
        )
        means = _extract_per_tree_means(forest, x_df)

        all_same = all(np.allclose(means[0], m) for m in means[1:])
        assert (
            all_same
        ), "Without max_samples/max_features all trees should be identical"

    def test_bootstrap_alone_produces_diversity(self):
        """Setting only max_samples (bootstrap) should produce diverse trees."""
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(
            x_df,
            y_df,
            n_estimators=10,
            max_samples=0.8,
            max_features=None,
        )
        means = _extract_per_tree_means(forest, x_df)

        all_same = all(np.allclose(means[0], m) for m in means[1:])
        assert not all_same, "Bootstrap sampling alone should produce different trees"

    def test_feature_subsampling_alone_produces_diversity(self):
        """Setting only max_features should produce diverse trees."""
        X, y = load_diabetes(return_X_y=True)
        x_df, y_df = _to_polars_x_y(X, y)

        forest = _fit_forest(
            x_df,
            y_df,
            n_estimators=10,
            max_samples=None,
            max_features=0.7,
        )
        means = _extract_per_tree_means(forest, x_df)

        all_same = all(np.allclose(means[0], m) for m in means[1:])
        assert not all_same, "Feature subsampling alone should produce different trees"
