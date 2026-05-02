"""End-to-end and better-than-random tests for skpro estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from partition_tree.skpro import (
    PartitionForestRegressor,
    PartitionTreeRegressor,
)
from partition_tree.skpro.distribution import IntervalDistribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_data():
    """Simple tabular regression dataset with informative features."""
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=3,
        noise=10.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def regression_data_dataframe(regression_data):
    """Same dataset but as pandas DataFrames/Series."""
    X_train, X_test, y_train, y_test = regression_data
    cols = [f"feat_{i}" for i in range(X_train.shape[1])]
    return (
        pd.DataFrame(X_train, columns=cols),
        pd.DataFrame(X_test, columns=cols),
        pd.Series(y_train, name="target"),
        pd.Series(y_test, name="target"),
    )


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


class TestPartitionTreeEndToEnd:
    """Verify the PartitionTreeRegressorSkpro works end-to-end."""

    def test_fit_predict_numpy(self, regression_data):
        """Fit on numpy arrays and produce point predictions."""
        X_train, X_test, y_train, _ = regression_data
        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    def test_fit_predict_dataframe(self, regression_data_dataframe):
        """Fit on pandas DataFrames and produce point predictions."""
        X_train, X_test, y_train, _ = regression_data_dataframe
        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    def test_fit_predict_multioutput_with_auto_dtype_overrides(self, regression_data):
        """A scalar 'auto' override should be applied to every target column."""
        X_train, X_test, y_train, _ = regression_data
        y_train_multi = pd.DataFrame(
            {
                "target_0": np.round(y_train / 0.25) * 0.25,
                "target_1": np.round((y_train + 1.0) / 0.5) * 0.5,
            }
        )

        model = PartitionTreeRegressor(
            max_leaves=20,
            max_depth=5,
            dtype_overrides="auto",
        )
        model.fit(X_train, y_train_multi)
        preds = model.predict(X_test)

        assert list(preds.columns) == ["target_0", "target_1"]
        assert preds.shape == (X_test.shape[0], y_train_multi.shape[1])

    def test_predict_proba_returns_distribution(self, regression_data):
        """predict_proba should return an IntervalDistribution."""
        X_train, X_test, y_train, _ = regression_data
        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)
        assert isinstance(dist, IntervalDistribution)
        assert len(dist.index) == X_test.shape[0]

    def test_predict_proba_mean_and_var(self, regression_data):
        """Distribution should expose mean and var with correct shapes."""
        X_train, X_test, y_train, _ = regression_data
        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)

        mean = dist.mean()
        var = dist.var()
        assert mean.shape[0] == X_test.shape[0]
        assert var.shape[0] == X_test.shape[0]
        assert (var.values >= 0).all(), "Variance must be non-negative"

    def test_predict_proba_sample(self, regression_data):
        """Distribution should support sampling."""
        X_train, X_test, y_train, _ = regression_data
        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)

        samples = dist.sample(n_samples=10)
        assert samples.shape[0] == 10 * X_test.shape[0]

    def test_predict_proba_cdf_pdf(self, regression_data):
        """Distribution should support CDF and PDF evaluation."""
        X_train, X_test, y_train, _ = regression_data
        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)

        # Evaluate CDF at the predicted mean
        mean_vals = dist.mean()
        cdf_at_mean = dist.cdf(mean_vals)
        # CDF values should be in [0, 1]
        assert (cdf_at_mean.values >= 0).all()
        assert (cdf_at_mean.values <= 1.0 + 1e-6).all()

        pdf_at_mean = dist.pdf(mean_vals)
        assert (pdf_at_mean.values >= 0).all()


class TestPartitionForestEndToEnd:
    """Verify the PartitionForestRegressorSkpro works end-to-end."""

    def test_fit_predict_numpy(self, regression_data):
        X_train, X_test, y_train, _ = regression_data
        model = PartitionForestRegressor(n_estimators=5, max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    def test_predict_proba_returns_distribution(self, regression_data):
        X_train, X_test, y_train, _ = regression_data
        model = PartitionForestRegressor(n_estimators=5, max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)
        assert isinstance(dist, IntervalDistribution)
        assert len(dist.index) == X_test.shape[0]

    def test_predict_proba_sample(self, regression_data):
        X_train, X_test, y_train, _ = regression_data
        model = PartitionForestRegressor(n_estimators=5, max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)
        samples = dist.sample(n_samples=10)
        assert samples.shape[0] == 10 * X_test.shape[0]


def test_skpro_estimators_expose_max_candidate_split_points_parameter():
    tree = PartitionTreeRegressor(
        max_leaves=20, max_depth=5, max_candidate_split_points=7
    )
    forest = PartitionForestRegressor(
        n_estimators=5,
        max_leaves=20,
        max_depth=5,
        max_candidate_split_points=11,
    )

    assert tree.get_params()["max_candidate_split_points"] == 7
    assert forest.get_params()["max_candidate_split_points"] == 11


# ---------------------------------------------------------------------------
# Better-than-random tests
# ---------------------------------------------------------------------------


class TestBetterThanRandom:
    """Estimator predictions must beat trivial baselines.

    A correct probabilistic regression model should:
    1. Achieve lower MSE than always predicting the training-set mean.
    2. Assign higher conditional density (log-likelihood) to training
       targets than a uniform distribution over the observed Y range.
    3. Produce non-degenerate predictions (not all identical).
    """

    def test_tree_beats_mean_baseline(self, regression_data):
        """PartitionTreeRegressorSkpro MSE should be lower than predicting the mean."""
        X_train, X_test, y_train, y_test = regression_data

        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse_model = mean_squared_error(y_test, preds)

        mean_baseline = np.full_like(y_test, y_train.mean())
        mse_baseline = mean_squared_error(y_test, mean_baseline)

        assert mse_model < mse_baseline, (
            f"Model MSE ({mse_model:.2f}) should be lower than "
            f"mean-baseline MSE ({mse_baseline:.2f})"
        )

    def test_forest_beats_mean_baseline(self, regression_data):
        """PartitionForestRegressorSkpro MSE should be lower than predicting the mean."""
        X_train, X_test, y_train, y_test = regression_data

        model = PartitionForestRegressor(n_estimators=10, max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse_model = mean_squared_error(y_test, preds)

        mean_baseline = np.full_like(y_test, y_train.mean())
        mse_baseline = mean_squared_error(y_test, mean_baseline)

        assert mse_model < mse_baseline, (
            f"Model MSE ({mse_model:.2f}) should be lower than "
            f"mean-baseline MSE ({mse_baseline:.2f})"
        )

    def test_tree_distributional_mean_beats_random(self, regression_data):
        """The distributional mean from predict_proba should also beat random."""
        X_train, X_test, y_train, y_test = regression_data

        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)
        dist_mean = dist.mean().values.ravel()
        mse_dist = mean_squared_error(y_test, dist_mean)

        mean_baseline = np.full_like(y_test, y_train.mean())
        mse_baseline = mean_squared_error(y_test, mean_baseline)

        assert mse_dist < mse_baseline, (
            f"Distributional mean MSE ({mse_dist:.2f}) should be lower than "
            f"mean-baseline MSE ({mse_baseline:.2f})"
        )

    def test_tree_log_likelihood_beats_uniform(self, regression_data):
        """Conditional density at y_test should exceed a uniform density."""
        X_train, X_test, y_train, y_test = regression_data

        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)

        y_test_df = pd.DataFrame(y_test, index=dist.index, columns=dist.columns)
        model_pdf = dist.pdf(y_test_df).values.ravel()
        model_log_lik = np.mean(np.log(np.maximum(model_pdf, 1e-15)))

        y_range = y_train.max() - y_train.min()
        uniform_density = 1.0 / y_range if y_range > 0 else 1e-15
        baseline_log_lik = np.log(uniform_density)

        assert model_log_lik > baseline_log_lik, (
            f"Model mean log-pdf ({model_log_lik:.4f}) should exceed "
            f"uniform baseline log-pdf ({baseline_log_lik:.4f})"
        )

    def test_forest_log_likelihood_beats_uniform(self, regression_data):
        """Forest conditional density at y_test should exceed a uniform density."""
        X_train, X_test, y_train, y_test = regression_data

        model = PartitionForestRegressor(n_estimators=10, max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)

        y_test_df = pd.DataFrame(y_test, index=dist.index, columns=dist.columns)
        model_pdf = dist.pdf(y_test_df).values.ravel()
        model_log_lik = np.mean(np.log(np.maximum(model_pdf, 1e-15)))

        y_range = y_train.max() - y_train.min()
        uniform_density = 1.0 / y_range if y_range > 0 else 1e-15
        baseline_log_lik = np.log(uniform_density)

        assert model_log_lik > baseline_log_lik, (
            f"Model mean log-pdf ({model_log_lik:.4f}) should exceed "
            f"uniform baseline log-pdf ({baseline_log_lik:.4f})"
        )

    def test_tree_predictions_not_constant(self, regression_data):
        """Point predictions should vary across different inputs."""
        X_train, X_test, y_train, _ = regression_data

        model = PartitionTreeRegressor(max_leaves=20, max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        assert np.std(preds) > 0, (
            "Predictions are all identical — the model is not using "
            "feature information to differentiate inputs"
        )


# ---------------------------------------------------------------------------
# min_volume_fraction (min target volume) constraint tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide_target_data():
    """Dataset whose target spans [0, 100] — wide enough to trigger Y-splits.

    Using ``boundaries_expansion_factor=0.0`` so the root Y-interval is
    exactly ``[min(y_train), max(y_train)]``, giving a known root volume.
    """
    rng = np.random.default_rng(0)
    n = 100
    X = rng.standard_normal((n, 3))
    y = np.linspace(0.0, 100.0, n)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    return (
        pd.DataFrame(X_train, columns=["f0", "f1", "f2"]),
        pd.DataFrame(X_test, columns=["f0", "f1", "f2"]),
        pd.Series(y_train, name="target"),
        pd.Series(y_test, name="target"),
    )


def _interval_widths(dist):
    """Collect widths of every predicted distribution interval."""
    return [iv.high - iv.low for ivs in dist._intervals for iv in ivs]


def _y_split_leaf_volumes(model, root_y_volume, tol=1e-6):
    """Return target_volumes for leaves narrower than the root Y range.

    Leaves whose ``target_volume < root_y_volume`` were produced by at
    least one Y-split, so they must satisfy ``target_volume >= min_volume_fraction``.
    """
    return [
        leaf["target_volume"]
        for leaf in model.get_leaves_info()
        if leaf["target_volume"] < root_y_volume - tol
    ]


class TestMinVolume:
    """Verify that the ``min_volume_fraction`` (minimum target interval volume) constraint
    is respected by both the tree and the forest estimators."""

    def test_tree_leaf_target_volumes_respect_min_volume_fraction(
        self, wide_target_data
    ):
        """Every leaf produced by a Y-split must have target_volume >= min_volume_fraction.

        Leaves with ``target_volume < root_y_volume`` are narrower than the
        initial Y bounding box, meaning a Y-split created them.  The
        ``min_volume_fraction`` parameter must ensure none of those volumes falls
        below the threshold.
        """
        X_train, _, y_train, _ = wide_target_data
        min_vol = 20.0
        root_y_volume = float(y_train.max() - y_train.min())

        model = PartitionTreeRegressor(
            max_leaves=50,
            max_depth=10,
            min_volume_fraction=min_vol,
            boundaries_expansion_factor=0.0,
        )
        model.fit(X_train, y_train)

        for vol in _y_split_leaf_volumes(model, root_y_volume):
            assert (
                vol >= min_vol
            ), f"Leaf has target_volume {vol:.4f} < min_volume_fraction={min_vol}"

    def test_tree_prohibitively_large_min_volume_fraction_prevents_y_splits(
        self, wide_target_data
    ):
        """When min_volume_fraction exceeds the full target range, no Y-splits can occur.

        Any Y-split of the root interval (width ≈ 100) would produce at
        least one child with width < 100 < min_volume_fraction=200, so all Y-splits
        must be rejected, leaving no leaf narrower than the root interval.
        """
        X_train, _, y_train, _ = wide_target_data
        root_y_volume = float(y_train.max() - y_train.min())

        model = PartitionTreeRegressor(
            max_leaves=50,
            max_depth=10,
            min_volume_fraction=root_y_volume
            * 2,  # impossible to satisfy for any Y-split
            boundaries_expansion_factor=0.0,
        )
        model.fit(X_train, y_train)

        y_split_vols = _y_split_leaf_volumes(model, root_y_volume)
        assert len(y_split_vols) == 0, (
            f"Expected no Y-split leaves but found {len(y_split_vols)} "
            f"with volumes {y_split_vols}"
        )

    def test_tree_predicted_intervals_respect_min_volume_fraction(
        self, wide_target_data
    ):
        """Every predicted distribution interval must have width >= min_volume_fraction.

        ``predict_proba`` returns a piecewise distribution whose segments
        correspond to the Y-intervals of the matched leaf.  Because the
        ``min_volume_fraction`` constraint ensures each Y-split child has
        ``target_volume >= min_volume_fraction``, no segment width should fall below
        the threshold.
        """
        X_train, X_test, y_train, _ = wide_target_data
        min_vol = 20.0

        model = PartitionTreeRegressor(
            max_leaves=50,
            max_depth=10,
            min_volume_fraction=min_vol,
            boundaries_expansion_factor=0.0,
        )
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)

        for width in _interval_widths(dist):
            assert (
                width >= min_vol
            ), f"Predicted interval width {width:.4f} < min_volume_fraction={min_vol}"

    def test_tighter_min_volume_fraction_widens_minimum_interval(
        self, wide_target_data
    ):
        """A stricter min_volume_fraction must raise the minimum predicted interval width.

        With ``min_volume_fraction=0.0`` the tree is free to produce arbitrarily
        narrow Y-intervals.  With ``min_volume_fraction=strict`` every interval must
        be at least ``strict`` wide.  We verify both that the constrained
        model meets the threshold AND that the unconstrained model would
        violate it — proving the constraint is genuinely restrictive.
        """
        X_train, X_test, y_train, _ = wide_target_data
        min_vol_strict = 30.0

        model_free = PartitionTreeRegressor(
            max_leaves=50,
            max_depth=10,
            min_volume_fraction=0.0,
            boundaries_expansion_factor=0.0,
        )
        model_free.fit(X_train, y_train)

        model_constrained = PartitionTreeRegressor(
            max_leaves=50,
            max_depth=10,
            min_volume_fraction=min_vol_strict,
            boundaries_expansion_factor=0.0,
        )
        model_constrained.fit(X_train, y_train)

        widths_free = _interval_widths(model_free.predict_proba(X_test))
        widths_constrained = _interval_widths(model_constrained.predict_proba(X_test))

        min_width_constrained = min(widths_constrained)
        min_width_free = min(widths_free)

        assert min_width_constrained >= min_vol_strict, (
            f"Constrained model (min_volume_fraction={min_vol_strict}) has interval "
            f"width {min_width_constrained:.4f} < min_volume_fraction"
        )
        # The unconstrained model must produce at least one narrower interval,
        # otherwise the constraint test above would be vacuous.
        assert min_width_free < min_vol_strict, (
            f"Unconstrained model already has no intervals below {min_vol_strict}; "
            f"the constraint test is vacuous (min free width = {min_width_free:.4f})"
        )

    def test_forest_predicted_intervals_respect_min_volume_fraction(
        self, wide_target_data
    ):
        """Forest: every predicted interval width must be >= min_volume_fraction.

        Each tree in the forest individually respects ``min_volume_fraction``, so
        every segment in the mixture distribution returned by
        ``predict_proba`` must also satisfy the width constraint.
        """
        X_train, X_test, y_train, _ = wide_target_data
        min_vol = 20.0

        model = PartitionForestRegressor(
            n_estimators=5,
            max_leaves=50,
            max_depth=10,
            min_volume_fraction=min_vol,
            boundaries_expansion_factor=0.0,
        )
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)

        for width in _interval_widths(dist):
            assert (
                width >= min_vol
            ), f"Forest predicted interval width {width:.4f} < min_volume_fraction={min_vol}"


# ---------------------------------------------------------------------------
# Forest diversity tests
# ---------------------------------------------------------------------------


class TestForestDiversity:
    """Verify that each tree in the forest produces a distinct probability distribution."""

    def test_trees_produce_different_probas(self, regression_data):
        """Each tree in the forest must generate different predictive distributions.

        If every tree produced the same distribution the ensemble would provide
        no value over a single tree.  We verify this by checking that the
        per-tree mean predictions are not all identical across trees.
        """
        X_train, X_test, y_train, _ = regression_data
        n_estimators = 5

        model = PartitionForestRegressor(
            n_estimators=n_estimators,
            max_leaves=30,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

        per_tree_dists = model.predict_proba_per_tree(X_test)

        assert len(per_tree_dists) == n_estimators, (
            f"Expected {n_estimators} per-tree distributions, "
            f"got {len(per_tree_dists)}"
        )

        # Collect the posterior mean from each tree's distribution.
        # shape: (n_estimators, n_test_samples)
        tree_means = np.array([d.mean().values.ravel() for d in per_tree_dists])

        # The std of means *across trees* must be > 0 for at least one test
        # sample, proving that trees do not all predict identically.
        std_across_trees = np.std(tree_means, axis=0)
        assert std_across_trees.max() > 0, (
            "All trees in the forest produce identical mean predictions — "
            "the ensemble is not generating diverse distributions."
        )

    def test_per_tree_count_matches_n_estimators(self, regression_data):
        """predict_proba_per_tree must return exactly n_estimators distributions."""
        X_train, X_test, y_train, _ = regression_data

        for n in [3, 7]:
            model = PartitionForestRegressor(
                n_estimators=n, max_leaves=20, max_depth=4, random_state=42
            )
            model.fit(X_train, y_train)
            dists = model.predict_proba_per_tree(X_test)
            assert (
                len(dists) == n
            ), f"With n_estimators={n}, expected {n} distributions, got {len(dists)}"

    def test_per_tree_distributions_are_interval_distributions(self, regression_data):
        """Each element returned by predict_proba_per_tree must be an IntervalDistribution."""
        X_train, X_test, y_train, _ = regression_data

        model = PartitionForestRegressor(n_estimators=3, max_leaves=20, max_depth=4)
        model.fit(X_train, y_train)
        dists = model.predict_proba_per_tree(X_test)

        for i, d in enumerate(dists):
            assert isinstance(
                d, IntervalDistribution
            ), f"Tree {i} returned {type(d).__name__}, expected IntervalDistribution"
            assert len(d.index) == X_test.shape[0], (
                f"Tree {i} distribution has {len(d.index)} rows, "
                f"expected {X_test.shape[0]}"
            )
