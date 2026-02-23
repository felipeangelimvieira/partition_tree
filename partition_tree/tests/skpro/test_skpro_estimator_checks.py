"""Run skpro's check_estimator suite on all custom estimators and distributions."""

import pytest
from skpro.utils.estimator_checks import check_estimator

from partition_tree.skpro import (
    PartitionForestRegressor,
    PartitionTreeRegressor,
)
from partition_tree.skpro.distribution import IntervalDistribution


class TestCheckEstimatorTree:
    """Run skpro estimator contract checks on PartitionTreeRegressorSkpro."""

    def test_check_estimator(self):
        results = check_estimator(
            PartitionTreeRegressor,
            raise_exceptions=True,
        )
        failed = {k: v for k, v in results.items() if v != "PASSED"}
        assert not failed, f"check_estimator failures: {failed}"


class TestCheckEstimatorForest:
    """Run skpro estimator contract checks on PartitionForestRegressorSkpro."""

    def test_check_estimator(self):
        results = check_estimator(
            PartitionForestRegressor,
            raise_exceptions=True,
        )
        failed = {k: v for k, v in results.items() if v != "PASSED"}
        assert not failed, f"check_estimator failures: {failed}"


class TestCheckDistribution:
    """Run skpro estimator contract checks on IntervalDistribution."""

    def test_check_estimator(self):
        results = check_estimator(
            IntervalDistribution,
            raise_exceptions=True,
        )
        failed = {k: v for k, v in results.items() if v != "PASSED"}
        assert not failed, f"check_estimator failures: {failed}"
