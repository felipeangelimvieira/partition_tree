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
from partition_tree.skpro.distribution import IntervalDistribution


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
        random_state=42,
        dtype_overrides="auto",
    ):
        """Partition forest probabilistic regressor.

        ``predict_proba`` returns a single :class:`IntervalDistribution` built on
        the union of all per-tree breakpoints. The breakpoint merge is performed
        in Rust and returned as flat segment arrays.
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

    def _predict_proba(self, X):
        """Merged predictive distribution via Rust segment merge.

        Calls the Rust forest, which merges each sample's per-tree
        piecewise-constant segments in parallel and returns flat CSR arrays
        ``(densities, lows, highs, offsets)``. A single
        :class:`IntervalDistribution` is constructed directly from those arrays.
        """
        check_is_fitted(self)
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)

        densities_flat, lows_flat, highs_flat, offsets = (
            self.partition_forest_.predict_proba_merged_segments(X_pol)
        )

        densities_arr = np.asarray(densities_flat, dtype=float)
        lows_arr = np.asarray(lows_flat, dtype=float)
        highs_arr = np.asarray(highs_flat, dtype=float)

        n_samples = len(offsets) - 1
        intervals_per_row = []
        pdf_values_per_row = []

        for i in range(n_samples):
            start, end = offsets[i], offsets[i + 1]
            row_lows = lows_arr[start:end]
            row_highs = highs_arr[start:end]
            row_densities = densities_arr[start:end]
            intervals_per_row.append(list(zip(row_lows.tolist(), row_highs.tolist())))
            pdf_values_per_row.append(row_densities)

        return IntervalDistribution(
            intervals=intervals_per_row,
            pdf_values=pdf_values_per_row,
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
