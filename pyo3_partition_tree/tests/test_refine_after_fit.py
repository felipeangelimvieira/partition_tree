"""Tests for the ``refine_after_fit`` flag exposed on both
:class:`PyPartitionTree` and :class:`PyPartitionForest`.

Refinement subdivides every leaf cell by every unique split coordinate
that appears anywhere in the (forest of) tree(s), so that no split line
crosses the interior of any leaf. The key invariant exercised here is
that refinement does **not** change predictions on the training data
(or any other ``X``): mass rescaling at the conditioned-cell boundary
keeps :meth:`predict` and :meth:`predict_proba` identical.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from pyo3_partition_tree import PyPartitionForest, PyPartitionTree


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def make_xy(n: int = 100) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two-feature dataset whose response is a thresholded version of x1.

    The signal is strong enough to produce several X splits, which in
    turn create unique split coordinates that the refinement pass must
    propagate to other leaves.
    """
    x1 = [float(i % 4) for i in range(n)]
    x2 = [float(i % 3) for i in range(n)]
    y = [2.0 if i % 4 < 2 else 4.0 for i in range(n)]
    X = pl.DataFrame({"x1": x1, "x2": x2}).cast(pl.Float64)
    y_df = pl.DataFrame({"y": y}).cast(pl.Float64)
    return X, y_df


def _tree_kwargs() -> dict:
    """Keyword arguments shared by both ``refine_after_fit`` variants."""
    return dict(
        max_leaves=13,
        boundaries_expansion_factor=0.0,
        min_samples_xy=0.0,
        min_samples_x=0.0,
        min_samples_y=0.0,
        min_gain=1e-8,
        min_volume_fraction=0.0,
        max_depth=6,
        min_samples_split=2.0,
        seed=42,
    )


def _forest_kwargs() -> dict:
    """Keyword arguments shared by both ``refine_after_fit`` variants."""
    return dict(
        n_estimators=5,
        max_leaves=13,
        boundaries_expansion_factor=0.0,
        min_samples_xy=0.0,
        min_samples_x=0.0,
        min_samples_y=0.0,
        min_gain=1e-8,
        min_volume_fraction=0.0,
        max_depth=6,
        min_samples_split=2.0,
        max_samples=0.8,
        replace=True,
        seed=42,
    )


# ----------------------------------------------------------------------
# PyPartitionTree
# ----------------------------------------------------------------------


def test_tree_refine_after_fit_defaults_to_false():
    """Backwards-compat: the new flag must default to ``False``."""
    model = PyPartitionTree(**_tree_kwargs())
    assert model.refine_after_fit is False


def test_tree_refine_after_fit_setter_roundtrips():
    model = PyPartitionTree(**_tree_kwargs())
    model.refine_after_fit = True
    assert model.refine_after_fit is True
    model.refine_after_fit = False
    assert model.refine_after_fit is False


def test_tree_refine_after_fit_preserves_predictions():
    """``predict`` must agree on every row between the plain and refined fits."""
    X, y = make_xy()

    plain = PyPartitionTree(**_tree_kwargs())
    plain.fit(X, y, None)
    preds_plain = plain.predict(X)

    refined = PyPartitionTree(**_tree_kwargs(), refine_after_fit=True)
    refined.fit(X, y, None)
    preds_refined = refined.predict(X)

    assert preds_plain.shape == preds_refined.shape
    plain_vals = preds_plain["y"].to_list()
    refined_vals = preds_refined["y"].to_list()
    assert len(plain_vals) == len(refined_vals)
    for i, (a, b) in enumerate(zip(plain_vals, refined_vals)):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (
            f"row {i}: plain {a} vs refined {b}"
        )


def test_tree_refine_after_fit_preserves_predict_proba():
    """``predict_proba`` mean and total mass must match per row."""
    X, y = make_xy()

    plain = PyPartitionTree(**_tree_kwargs())
    plain.fit(X, y, None)
    probs_plain = plain.predict_proba(X)

    refined = PyPartitionTree(**_tree_kwargs(), refine_after_fit=True)
    refined.fit(X, y, None)
    probs_refined = refined.predict_proba(X)

    assert len(probs_plain) == len(probs_refined)
    for i, (dp, dr) in enumerate(zip(probs_plain, probs_refined)):
        # Total mass is invariant: rescaled refined cells sum back to
        # the source leaf's mass.
        assert math.isclose(dp.total_mass(), dr.total_mass(), abs_tol=1e-10), (
            f"row {i}: total_mass differs: plain={dp.total_mass()} "
            f"refined={dr.total_mass()}"
        )

        mp = dp.mean()
        mr = dr.mean()
        assert set(mp.keys()) == set(mr.keys()), (
            f"row {i}: mean columns differ"
        )
        for col, vp in mp.items():
            vr = mr[col]
            assert len(vp) == len(vr), (
                f"row {i} col {col}: mean length differs"
            )
            for j, (a, b) in enumerate(zip(vp, vr)):
                assert math.isclose(a, b, abs_tol=1e-10), (
                    f"row {i} col {col}[{j}]: plain {a} vs refined {b}"
                )


def test_tree_refine_after_fit_grows_leaves_monotonically():
    """The refined tree should have at least as many leaves as the plain one."""
    X, y = make_xy()

    plain = PyPartitionTree(**_tree_kwargs())
    plain.fit(X, y, None)
    n_plain = sum(1 for n in plain.get_nodes_info() if n["is_leaf"])

    refined = PyPartitionTree(**_tree_kwargs(), refine_after_fit=True)
    refined.fit(X, y, None)
    n_refined = sum(1 for n in refined.get_nodes_info() if n["is_leaf"])

    assert n_refined >= n_plain, (
        f"refined leaves ({n_refined}) should be >= plain leaves ({n_plain})"
    )


# ----------------------------------------------------------------------
# PyPartitionForest
# ----------------------------------------------------------------------


def test_forest_refine_after_fit_defaults_to_false():
    model = PyPartitionForest(**_forest_kwargs())
    assert model.refine_after_fit is False


def test_forest_refine_after_fit_setter_roundtrips():
    model = PyPartitionForest(**_forest_kwargs())
    model.refine_after_fit = True
    assert model.refine_after_fit is True
    model.refine_after_fit = False
    assert model.refine_after_fit is False


def test_forest_refine_after_fit_preserves_predictions():
    """``predict`` must agree on every row between the plain and refined fits."""
    X, y = make_xy()

    plain = PyPartitionForest(**_forest_kwargs())
    plain.fit(X, y, None)
    preds_plain = plain.predict(X)

    refined = PyPartitionForest(**_forest_kwargs(), refine_after_fit=True)
    refined.fit(X, y, None)
    preds_refined = refined.predict(X)

    plain_vals = preds_plain["y"].to_list()
    refined_vals = preds_refined["y"].to_list()
    assert len(plain_vals) == len(refined_vals)
    for i, (a, b) in enumerate(zip(plain_vals, refined_vals)):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-10), (
            f"row {i}: plain {a} vs refined {b}"
        )


def test_forest_refine_after_fit_preserves_predict_proba_mean():
    """Forest ``predict_proba`` (ensembled) means and total mass must match."""
    X, y = make_xy()

    plain = PyPartitionForest(**_forest_kwargs())
    plain.fit(X, y, None)
    probs_plain = plain.predict_proba(X)

    refined = PyPartitionForest(**_forest_kwargs(), refine_after_fit=True)
    refined.fit(X, y, None)
    probs_refined = refined.predict_proba(X)

    assert len(probs_plain) == len(probs_refined)
    for i, (dp, dr) in enumerate(zip(probs_plain, probs_refined)):
        # Forest ensembling divides masses by n_trees; the per-tree
        # rescaling preserves their sum, so total_mass is unchanged.
        assert math.isclose(dp.total_mass(), dr.total_mass(), abs_tol=1e-10), (
            f"row {i}: total_mass differs: plain={dp.total_mass()} "
            f"refined={dr.total_mass()}"
        )

        mp = dp.mean()
        mr = dr.mean()
        assert set(mp.keys()) == set(mr.keys())
        for col, vp in mp.items():
            vr = mr[col]
            assert len(vp) == len(vr)
            for j, (a, b) in enumerate(zip(vp, vr)):
                assert math.isclose(a, b, abs_tol=1e-10), (
                    f"row {i} col {col}[{j}]: plain {a} vs refined {b}"
                )


def test_forest_refine_after_fit_grows_or_keeps_leaves():
    """Every tree in the refined forest has >= leaves than its plain twin."""
    X, y = make_xy()

    plain = PyPartitionForest(**_forest_kwargs())
    plain.fit(X, y, None)
    plain_trees = plain.get_nodes_info()

    refined = PyPartitionForest(**_forest_kwargs(), refine_after_fit=True)
    refined.fit(X, y, None)
    refined_trees = refined.get_nodes_info()

    assert len(plain_trees) == len(refined_trees)
    for t_idx, (pt, rt) in enumerate(zip(plain_trees, refined_trees)):
        n_plain = sum(1 for n in pt if n["is_leaf"])
        n_refined = sum(1 for n in rt if n["is_leaf"])
        assert n_refined >= n_plain, (
            f"tree {t_idx}: refined leaves ({n_refined}) "
            f"should be >= plain leaves ({n_plain})"
        )


# ----------------------------------------------------------------------
# Pickle roundtrip — the new flag must survive serde
# ----------------------------------------------------------------------


def test_tree_refine_after_fit_survives_pickle():
    import pickle

    X, y = make_xy()
    refined = PyPartitionTree(**_tree_kwargs(), refine_after_fit=True)
    refined.fit(X, y, None)
    preds_before = refined.predict(X)

    restored = pickle.loads(pickle.dumps(refined))
    preds_after = restored.predict(X)
    assert preds_before.equals(preds_after)


def test_forest_refine_after_fit_survives_pickle():
    import pickle

    X, y = make_xy()
    refined = PyPartitionForest(**_forest_kwargs(), refine_after_fit=True)
    refined.fit(X, y, None)
    preds_before = refined.predict(X)

    restored = pickle.loads(pickle.dumps(refined))
    preds_after = restored.predict(X)
    assert preds_before.equals(preds_after)
