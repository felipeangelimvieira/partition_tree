"""Run skpro's check_estimator suite on all custom estimators and distributions."""

import pytest
from skpro.utils.estimator_checks import check_estimator

from partition_tree.skpro import (
    PartitionForestRegressor,
    PartitionTreeRegressor,
)
from partition_tree.skpro.distribution import IntervalDistribution


def test_check_estimator_tree():
    results = check_estimator(
        PartitionTreeRegressor,
        raise_exceptions=True,
    )
    failed = {k: v for k, v in results.items() if v != "PASSED"}
    assert not failed, f"check_estimator failures: {failed}"


def test_check_estimator_forest():
    results = check_estimator(
        PartitionForestRegressor,
        raise_exceptions=True,
    )
    failed = {k: v for k, v in results.items() if v != "PASSED"}
    assert not failed, f"check_estimator failures: {failed}"


def test_check_interval_distribution():
    results = check_estimator(
        IntervalDistribution,
        raise_exceptions=True,
    )
    failed = {k: v for k, v in results.items() if v != "PASSED"}
    assert not failed, f"check_estimator failures: {failed}"
