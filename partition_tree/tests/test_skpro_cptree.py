"""skpro compatibility tests for CPTreeRegressorSkpro."""

import numpy as np
import pandas as pd
from skpro.utils.estimator_checks import check_estimator
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split
from skpro.metrics import CRPS

from partition_tree.estimators import PartitionTreeRegressorSkpro
from partition_tree.skpro.distribution import IntervalDistribution


def test_cptree_skpro_predict_proba_matches_mean():
    X, y = make_regression(n_samples=30, n_features=3, noise=0.1, random_state=0)
    est = PartitionTreeRegressorSkpro(max_depth=2, max_iter=20, min_samples_split=2)

    est.fit(X, y)

    point_pred = est.predict(X)
    dist = est.predict_proba(X)

    means = np.asarray(dist.mean())

    assert point_pred.shape[0] == 30
    assert means.shape[0] == 30
    np.testing.assert_allclose(
        np.asarray(point_pred).reshape(-1), means.reshape(-1), rtol=1e-3, atol=1e-3
    )


def test_cptree_skpro_check_estimator():
    estimator = PartitionTreeRegressorSkpro(
        max_depth=2, max_iter=20, min_samples_split=2
    )
    # skpro's check_estimator exercises fit/predict/proba contracts
    check_estimator(estimator)


def test_cptree_skpro_crps_runs():
    """CRPS evaluation should run on predicted piecewise distributions."""

    X, y = make_regression(n_samples=120, n_features=4, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    y_train_df = pd.Series(y_train).to_frame("target")
    y_test_df = pd.Series(y_test).to_frame("target")

    est = PartitionTreeRegressorSkpro(max_depth=2, max_iter=30, min_samples_split=2)
    est.fit(X_train, y_train_df)

    y_pred = est.predict_proba(X_test)

    crps_value = CRPS()(y_test_df, y_pred)

    assert np.isfinite(crps_value)


def test_predict_proba_pdfs_vary_across_samples():
    """Predictive pdfs should differ for inputs landing in different leaves."""

    # Simple dataset with clearly separated targets so the tree can split.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.5, 2.0, 3.0])

    est = PartitionTreeRegressorSkpro(
        max_depth=2,
        max_iter=20,
        min_samples_split=1,
        min_samples_leaf_x=1,
        min_samples_leaf_y=1,
    )
    est.fit(X, y)

    X_test = np.array([[0.1], [2.9]])
    dist = est.predict_proba(X_test)

    means = dist.mean()

    # Assert means differ
    assert means.std().item() > 0


def test_no_zero_pdf_for_training_samples():
    """Training samples should have non-zero PDF when at interval boundaries.

    This tests that interval boundaries are correctly handled (open vs closed)
    so that training points at split boundaries are properly matched.
    We use a simple dataset where all training samples should be covered.
    """
    # Simple synthetic dataset where we can guarantee coverage
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = X[:, 0] * 10 + 50 + np.random.randn(n_samples) * 5  # Linear relationship
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # Use a simple tree that should cover all training points
    tree = PartitionTreeRegressorSkpro(
        max_iter=20,
        max_depth=5,
        min_samples_leaf_x=5,
        min_samples_leaf_y=5,
        min_target_volume=0.0,
        min_samples_leaf=5,
        boundaries_expansion_factor=0.5,
        min_split_gain=0,
        seed=42,
    )
    tree.fit(X, y)

    y_pred_proba = tree.predict_proba(X)
    pdfs = y_pred_proba.pdf(y)

    # No training sample should have a PDF <= 1e-8 (effectively zero)
    zero_pdf_samples = y[pdfs.values.flatten() <= 1e-8].dropna()

    assert len(zero_pdf_samples) == 0, (
        f"Found {len(zero_pdf_samples)} training samples with zero PDF. "
        "This indicates interval boundary handling issues."
    )


def test_interval_boundary_open_closed():
    """Test that interval boundaries correctly handle open vs closed.

    Specifically tests that:
    - Left interval [a, split) excludes split point
    - Right interval [split, b] includes split point
    - No gaps exist at split boundaries
    """
    from partition_tree.skpro.distribution import Interval

    # Test half-open interval [0, 1)
    left_interval = Interval(0.0, 1.0, lower_closed=True, upper_closed=False)
    assert (
        left_interval.contains(0.0) == True
    ), "Left interval should include lower bound"
    assert left_interval.contains(0.5) == True, "Left interval should include interior"
    assert (
        left_interval.contains(1.0) == False
    ), "Left interval should exclude upper bound"

    # Test right interval [1, 2]
    right_interval = Interval(1.0, 2.0, lower_closed=True, upper_closed=True)
    assert (
        right_interval.contains(1.0) == True
    ), "Right interval should include lower bound"
    assert (
        right_interval.contains(1.5) == True
    ), "Right interval should include interior"
    assert (
        right_interval.contains(2.0) == True
    ), "Right interval should include upper bound"

    # Verify no gap: value at split point should be covered by exactly one interval
    split_point = 1.0
    left_matches = left_interval.contains(split_point)
    right_matches = right_interval.contains(split_point)
    assert left_matches != right_matches, (
        f"Split point {split_point} should match exactly one interval, "
        f"but left={left_matches}, right={right_matches}"
    )
