import numpy as np
import pandas as pd
import polars as pl
from pyo3_partition_tree import PyPartitionForest, PyPartitionTree
from sklearn.utils.validation import check_is_fitted
from skpro.regression.base import BaseProbaRegressor

from partition_tree.utils import _preprocess, _preprocess_X
from partition_tree.sklearn.partition_tree import (
    _convert_string_columns_to_categorical,
    _ensure_numeric_float64,
)
from partition_tree.skpro.distribution import IntervalDistribution


class PartitionTreeRegressor(BaseProbaRegressor):
    _task = "regression"

    _tags = {
        "authors": ["felipeangelimvieira"],
    }

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

    def __init__(
        self,
        max_leaves=101,
        boundaries_expansion_factor=0.1,
        min_samples_xy=1.0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0.0,
        max_depth=5,
        min_samples_split=2.0,
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
        super().__init__()

    def _fit(self, X, y):
        self.partition_tree_ = PyPartitionTree(
            max_leaves=self.max_leaves,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_samples_xy=self.min_samples_xy,
            min_samples_x=self.min_samples_x,
            min_samples_y=self.min_samples_y,
            min_gain=self.min_gain,
            min_volume_fraction=self.min_volume_fraction,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )

        if isinstance(y, pd.Series):
            y = y.to_frame("target")
        elif isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=["target"])

        self._y_columns = y.columns
        X_proc, y_proc = _preprocess(X, y)
        X_proc = _ensure_numeric_float64(X_proc)
        y_proc = _ensure_numeric_float64(y_proc)

        X_pol = pl.DataFrame(X_proc)
        X_pol, self._categorical_metadata = _convert_string_columns_to_categorical(
            X_pol, return_categories=True
        )
        X_pol = _ensure_numeric_float64(X_pol)
        y_pol = pl.DataFrame(y_proc).cast(pl.Float64)

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
        check_is_fitted(self)
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)
        return self.partition_tree_.apply(X_pol)

    def get_leaves_info(self):
        check_is_fitted(self)
        return self.partition_tree_.get_leaves_info()

    def get_feature_importances(self, normalize: bool = True) -> dict:
        check_is_fitted(self)
        importances = self.partition_tree_.get_feature_importances(normalize)
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionForestRegressor(BaseProbaRegressor):
    _task = "regression"

    _tags = {
        "authors": ["felipeangelimvieira"],
    }

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

    def __init__(
        self,
        n_estimators=100,
        max_leaves=101,
        boundaries_expansion_factor=0.1,
        min_samples_xy=1.0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0.0,
        max_depth=5,
        min_samples_split=2.0,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
    ):
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
        self.max_features = max_features
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        self.partition_forest_ = PyPartitionForest(
            n_estimators=self.n_estimators,
            max_leaves=self.max_leaves,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_samples_xy=self.min_samples_xy,
            min_samples_x=self.min_samples_x,
            min_samples_y=self.min_samples_y,
            min_gain=self.min_gain,
            min_volume_fraction=self.min_volume_fraction,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            max_samples=self.max_samples,
            seed=self.random_state,
        )

        if isinstance(y, pd.Series):
            self._y_columns = [y.name]
            y = y.to_frame("target")
        elif isinstance(y, pd.DataFrame):
            self._y_columns = y.columns.to_list()
        elif isinstance(y, np.ndarray):
            self._y_columns = ["target"]
            y = pd.DataFrame(y, columns=["target"])

        X_proc, y_proc = _preprocess(X, y)
        X_proc = _ensure_numeric_float64(X_proc)
        y_proc = _ensure_numeric_float64(y_proc)

        X_pol = pl.DataFrame(X_proc)
        X_pol, self._categorical_metadata = _convert_string_columns_to_categorical(
            X_pol, return_categories=True
        )
        X_pol = _ensure_numeric_float64(X_pol)
        y_pol = pl.DataFrame(y_proc).cast(pl.Float64)

        self.partition_forest_.fit(X_pol, y_pol, None)
        return self

    def _predict(self, X):
        check_is_fitted(self)
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
        check_is_fitted(self)
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
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        interval_dists = self.predict_proba_per_tree(X)
        return IntervalDistribution.from_mixture(
            distributions=interval_dists,
            weights=[1.0 / self.n_estimators] * self.n_estimators,
            index=X_proc.index,
            columns=self._y_columns,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
