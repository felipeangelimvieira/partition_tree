import numpy as np
import pandas as pd
import polars as pl
from pyo3_partition_tree import PyPartitionForest, PyPartitionTree
from sklearn.utils.validation import check_is_fitted
from skpro.regression.base import BaseProbaRegressor

from partition_tree.utils import (
    _convert_string_columns_to_categorical,
    _ensure_numeric_float64,
    _prepare_regression_training_data,
    _preprocess_X,
)
from partition_tree.skpro.distribution import (
    IntervalDistribution,
    MixtureIntervalDistribution,
)


class PartitionTreeRegressor(BaseProbaRegressor):

    _tags = {
        "authors": ["felipeangelimvieira"],
    }

    def __init__(
        self,
        max_leaves=None,
        boundaries_expansion_factor=0.1,
        min_samples_xy=1.0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0.1,
        max_depth=None,
        min_samples_split=2.0,
        max_candidate_split_points=None,
        loss=None,
        random_state=42,
        dtype_overrides="auto",
    ):
        self.max_leaves = max_leaves
        self.boundaries_expansion_factor = boundaries_expansion_factor
        self.min_samples_xy = min_samples_xy
        self.min_samples_x = min_samples_x
        self.min_samples_y = min_samples_y
        self.min_gain = min_gain
        self.min_volume_fraction = min_volume_fraction
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_candidate_split_points = max_candidate_split_points
        self.loss = loss
        self.random_state = random_state
        self.dtype_overrides = dtype_overrides
        super().__init__()

    @property
    def _max_leaves(self):
        return self.max_leaves if self.max_leaves is not None else int(1e6)

    @property
    def _max_depth(self):
        return self.max_depth if self.max_depth is not None else int(1e6)

    def _fit(self, X, y):
        (
            X_pol,
            y_pol,
            self._y_columns,
            self._categorical_metadata,
            resolved_dtype_overrides,
        ) = _prepare_regression_training_data(X, y, self.dtype_overrides)

        self.partition_tree_ = PyPartitionTree(
            max_leaves=self._max_leaves,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_samples_xy=self.min_samples_xy,
            min_samples_x=self.min_samples_x,
            min_samples_y=self.min_samples_y,
            min_gain=self.min_gain,
            min_volume_fraction=self.min_volume_fraction,
            max_depth=self._max_depth,
            min_samples_split=self.min_samples_split,
            max_candidate_split_points=self.max_candidate_split_points,
            loss=self.loss,
            seed=self.random_state,
            dtype_overrides=resolved_dtype_overrides,
        )

        try:
            self.partition_tree_.fit(X_pol, y_pol, None)
        except Exception as e:
            raise ValueError(f"Error fitting PartitionTreeRegressorSkpro: {e}")
        return self

    def _predict(self, X):
        check_is_fitted(self)
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)
        preds = self.partition_tree_.predict(X_pol)
        return pd.DataFrame(preds, columns=self._y_columns, index=X_proc.index)

    def _predict_proba(self, X):
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)

        piecewise_proba = self.partition_tree_.predict_proba(X_pol)

        intervals_per_row = []
        pdf_values_per_row = []
        for dist in piecewise_proba:
            row_intervals = []
            row_pdfs = []
            for density, low, high in dist.pdf_segments():
                row_intervals.append((float(low), float(high)))
                row_pdfs.append(float(density))

            sorted_indices = np.argsort([iv[0] for iv in row_intervals])
            row_intervals = [row_intervals[i] for i in sorted_indices]
            row_pdfs = [row_pdfs[i] for i in sorted_indices]

            intervals_per_row.append(row_intervals)
            pdf_values_per_row.append(np.asarray(row_pdfs, dtype=float))

        return IntervalDistribution(
            intervals_per_row,
            pdf_values=pdf_values_per_row,
            index=X_proc.index,
            columns=self._y_columns,
        )

    def apply(self, X):
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)
        return self.partition_tree_.apply(X_pol)

    def get_leaves_info(self):
        return self.partition_tree_.get_leaves_info()

    def get_feature_importances(self, normalize: bool = True) -> dict:
        importances = self.partition_tree_.get_feature_importances(normalize)
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "max_leaves": 10,
            "max_depth": 3,
        }
        params2 = {
            "max_leaves": 20,
            "max_depth": 5,
            "min_samples_split": 5.0,
        }
        return [params1, params2]


class PartitionForestRegressor(BaseProbaRegressor):

    _tags = {
        "authors": ["felipeangelimvieira"],
    }

    def __init__(
        self,
        n_estimators=100,
        max_leaves=None,
        boundaries_expansion_factor=0.1,
        min_samples_xy=0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0.1,
        max_depth=None,
        min_samples_split=2.0,
        max_samples=1.0,
        replace=True,
        max_features=1.0,
        max_candidate_split_points=None,
        loss=None,
        output_distribution="merged",
        random_state=42,
        dtype_overrides="auto",
    ):
        """
        Parameters
        ----------
        output_distribution : {"merged", "mixture"}, default "merged"
            Controls how the per-tree predictive distributions are combined
            when calling ``predict_proba``.

            * ``"merged"`` (default) — builds a single :class:`IntervalDistribution`
              on the union of all tree breakpoints via
              ``IntervalDistribution.from_mixture``.  The resulting object is a
              standard piecewise-uniform distribution and supports all skpro methods.
            * ``"mixture"`` — returns a :class:`MixtureIntervalDistribution` that
              stores the per-tree distributions and weights without merging.  PDF,
              CDF, mean, variance, energy and PPF are all computed on-the-fly from
              the mixture identity, avoiding the up-front merge cost.
        """
        self.n_estimators = n_estimators
        self.max_leaves = max_leaves
        self.boundaries_expansion_factor = boundaries_expansion_factor
        self.min_samples_xy = min_samples_xy
        self.min_samples_x = min_samples_x
        self.min_samples_y = min_samples_y
        self.min_gain = min_gain
        self.min_volume_fraction = min_volume_fraction
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_samples = max_samples
        self.replace = replace
        self.max_features = max_features
        self.max_candidate_split_points = max_candidate_split_points
        self.loss = loss
        self.random_state = random_state
        self.output_distribution = output_distribution
        self.dtype_overrides = dtype_overrides
        super().__init__()

    @property
    def _max_leaves(self):
        return self.max_leaves if self.max_leaves is not None else int(1e6)

    @property
    def _max_depth(self):
        return self.max_depth if self.max_depth is not None else int(1e6)

    def _fit(self, X, y):
        (
            X_pol,
            y_pol,
            self._y_columns,
            self._categorical_metadata,
            resolved_dtype_overrides,
        ) = _prepare_regression_training_data(X, y, self.dtype_overrides)

        self.partition_forest_ = PyPartitionForest(
            n_estimators=self.n_estimators,
            max_leaves=self._max_leaves,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_samples_xy=self.min_samples_xy,
            min_samples_x=self.min_samples_x,
            min_samples_y=self.min_samples_y,
            min_gain=self.min_gain,
            min_volume_fraction=self.min_volume_fraction,
            max_depth=self._max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            max_samples=self.max_samples,
            replace=self.replace,
            max_candidate_split_points=self.max_candidate_split_points,
            loss=self.loss,
            seed=self.random_state,
            dtype_overrides=resolved_dtype_overrides,
        )

        self.partition_forest_.fit(X_pol, y_pol, None)
        return self

    def _predict(self, X):

        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)
        preds = self.partition_forest_.predict(X_pol)
        return pd.DataFrame(preds, columns=self._y_columns, index=X_proc.index)

    def predict_proba_per_tree(self, X) -> list:
        """Return per-tree predictive distributions without mixing them.

        Parameters
        ----------
        X : array-like
            Features to predict for.

        Returns
        -------
        list of IntervalDistribution
            One ``IntervalDistribution`` per tree in the forest, in tree order.
        """

        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)

        piecewise_probas = self.partition_forest_.predict_trees_proba(X_pol)

        interval_dists = []
        for piecewise_proba in piecewise_probas:
            intervals_per_row = []
            pdf_values_per_row = []
            for dist in piecewise_proba:
                row_intervals = []
                row_pdfs = []
                for density, low, high in dist.pdf_segments():
                    row_intervals.append((float(low), float(high)))
                    row_pdfs.append(float(density))

                sorted_indices = np.argsort([iv[0] for iv in row_intervals])
                row_intervals = [row_intervals[i] for i in sorted_indices]
                row_pdfs = [row_pdfs[i] for i in sorted_indices]

                intervals_per_row.append(row_intervals)
                pdf_values_per_row.append(np.asarray(row_pdfs, dtype=float))

            interval_dists.append(
                IntervalDistribution(
                    intervals_per_row,
                    pdf_values=pdf_values_per_row,
                    index=X_proc.index,
                    columns=self._y_columns,
                )
            )

        return interval_dists

    def _predict_proba(self, X):
        check_is_fitted(self)
        if self.output_distribution not in ("merged", "mixture"):
            raise ValueError(
                f"output_distribution must be 'merged' or 'mixture', "
                f"got {self.output_distribution!r}"
            )
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        interval_dists = self.predict_proba_per_tree(X)
        weights = [1.0 / len(interval_dists)] * len(interval_dists)
        if self.output_distribution == "mixture":
            return MixtureIntervalDistribution(
                distributions=interval_dists,
                weights=weights,
                index=X_proc.index,
                columns=self._y_columns,
            )

        return IntervalDistribution.from_mixture(
            distributions=interval_dists,
            weights=weights,
            index=X_proc.index,
            columns=self._y_columns,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "n_estimators": 5,
            "max_leaves": 10,
            "max_depth": 3,
        }
        params2 = {
            "n_estimators": 3,
            "max_leaves": 20,
            "max_depth": 5,
            "min_samples_split": 5.0,
        }
        return [params1, params2]
