from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from pyo3_partition_tree import PyPartitionForest, PyPartitionTree
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, validate_data

from partition_tree.utils import _preprocess, _preprocess_X


def _convert_string_columns_to_categorical(
    df: pl.DataFrame,
    categories_map: Optional[dict] = None,
    return_categories: bool = False,
):
    if categories_map is not None:
        missing_cols = [col for col in categories_map.keys() if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "Missing expected categorical columns: " + ", ".join(missing_cols)
            )
        string_columns = list(categories_map.keys())
    else:
        string_columns = [
            col
            for col in df.columns
            if df[col].dtype == pl.Utf8 or df[col].dtype == pl.Categorical
        ]

    if not string_columns:
        if return_categories:
            return df, categories_map or {}
        return df

    if categories_map is None:
        categories_map = {
            col: sorted(map(str, df[col].unique().to_list())) for col in string_columns
        }

    df_converted = df.with_columns(
        [pl.col(col).cast(pl.Categorical()) for col in string_columns]
    )

    if return_categories:
        return df_converted, categories_map
    return df_converted


def _convert_int_to_float64(df):
    if isinstance(df, pl.DataFrame):
        int_cols = [
            col for col, dtype in df.schema.items() if dtype in pl.INTEGER_DTYPES
        ]
        if not int_cols:
            return df
        return df.with_columns([pl.col(col).cast(pl.Float64) for col in int_cols])

    if isinstance(df, pd.DataFrame):
        df_converted = df.copy()
        int_cols = df_converted.select_dtypes(
            include=["int", "Int64", "Int32", "int64", "int32"]
        ).columns
        if len(int_cols) == 0:
            return df_converted
        df_converted.loc[:, int_cols] = df_converted.loc[:, int_cols].astype("float64")
        return df_converted

    return df


def _coerce_target_dataframe(y, default_column: str = "target") -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        return y.copy()
    if isinstance(y, pd.Series):
        return y.to_frame(default_column)
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            return pd.DataFrame(y, columns=[default_column])
        return pd.DataFrame(
            y, columns=[f"{default_column}_{i}" for i in range(y.shape[1])]
        )
    return pd.DataFrame(y, columns=[default_column])


def _prepare_classification_training_data(
    X, y
) -> Tuple[pl.DataFrame, pl.DataFrame, np.dtype, List[str], dict]:
    y_df = _coerce_target_dataframe(y)
    y_dtype = y_df.dtypes[0]
    y_columns = y_df.columns.to_list()

    categorical_targets = y_columns if isinstance(y, pd.DataFrame) else None
    X_processed, y_processed = _preprocess(
        X, y_df, categorical_targets=categorical_targets
    )

    target_column = y_processed.columns[0]
    target_values = y_processed[target_column].to_numpy()

    if hasattr(target_values.dtype, "categories"):
        pass
    elif (
        np.issubdtype(target_values.dtype, np.floating)
        and len(np.unique(target_values)) > 20
    ):
        raise ValueError(
            "Unknown label type: 'continuous'. Classifiers require discrete targets."
        )

    y_processed = y_processed.astype(str)
    y_processed = y_processed.copy()
    y_processed[target_column] = pd.Categorical(y_processed[target_column])

    X_pol = pl.DataFrame(_convert_int_to_float64(X_processed))
    X_pol = _convert_int_to_float64(X_pol)
    X_pol, categorical_metadata = _convert_string_columns_to_categorical(
        X_pol, return_categories=True
    )
    y_pol = pl.DataFrame(y_processed).cast(
        pl.Enum(categories=list(y_processed[target_column].cat.categories))
    )

    return X_pol, y_pol, y_dtype, y_columns, categorical_metadata


def _prepare_classification_features(
    X, categorical_metadata: Optional[dict] = None
) -> pl.DataFrame:
    X_processed = _preprocess_X(X)
    X_pol = pl.DataFrame(_convert_int_to_float64(X_processed))
    X_pol = _convert_string_columns_to_categorical(
        X_pol, categories_map=categorical_metadata
    )
    return _convert_int_to_float64(X_pol)


def _probabilities_from_piecewise(
    piecewise_proba, classes: List[str], target_column: str
) -> np.ndarray:
    n_samples = len(piecewise_proba)
    n_classes = len(classes)
    probabilities = np.empty((n_samples, n_classes))

    for i, dist in enumerate(piecewise_proba):
        cat_probs = dist.category_probabilities()
        domain_names = dist.categorical_domain_names(target_column)

        if target_column in cat_probs and domain_names is not None:
            probs = cat_probs[target_column]
            name_to_prob = dict(zip(domain_names, probs))
            probabilities[i, :] = [name_to_prob.get(str(c), 0.0) for c in classes]
        else:
            probabilities[i, :] = 0.0

    probabilities = probabilities + 1e-8
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _ensure_numeric_float64(df):
    if isinstance(df, pl.DataFrame):
        numeric_cols = [
            col for col, dtype in df.schema.items() if dtype in pl.NUMERIC_DTYPES
        ]
        if not numeric_cols:
            return df
        return df.with_columns([pl.col(col).cast(pl.Float64) for col in numeric_cols])

    if isinstance(df, pd.DataFrame):
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            return df
        df_float = df.copy()
        df_float.loc[:, numeric_cols] = df_float.loc[:, numeric_cols].astype(np.float64)
        return df_float

    return df


def _prepare_regression_training_data(
    X, y
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str], dict]:
    y_df = _coerce_target_dataframe(y)
    y_columns = y_df.columns.to_list()

    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    X_processed, y_processed = _preprocess(X_df, y_df)

    X_processed = _convert_int_to_float64(X_processed)
    X_pol = pl.DataFrame(X_processed)
    X_pol = _convert_int_to_float64(X_pol)
    X_pol, categorical_metadata = _convert_string_columns_to_categorical(
        X_pol, return_categories=True
    )
    y_pol = pl.DataFrame(y_processed).cast(pl.Float64)

    return X_pol, y_pol, y_columns, categorical_metadata


def _prepare_regression_features(
    X, categorical_metadata: Optional[dict] = None
) -> pl.DataFrame:
    if isinstance(X, pd.DataFrame):
        X_df = _convert_int_to_float64(X)
    else:
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_df = _convert_int_to_float64(X_df)
    X_processed = _preprocess_X(X_df)
    X_pol = pl.DataFrame(_convert_int_to_float64(X_processed))
    X_pol = _convert_string_columns_to_categorical(
        X_pol, categories_map=categorical_metadata
    )
    return _convert_int_to_float64(X_pol)


def _wrap_sample_weights(sample_weights):
    if sample_weights is None:
        return None
    if isinstance(sample_weights, pl.DataFrame):
        return sample_weights
    return pl.DataFrame({"sample_weights": sample_weights})


class PartitionTreeClassifier(ClassifierMixin, BaseEstimator):
    _task = "classification"

    def __init__(
        self,
        max_leaves=101,
        boundaries_expansion_factor=0.1,
        min_samples_xy=1.0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0.0,
        max_depth=1000,
        min_samples_split=2.0,
        loss=None,
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
        self.loss = loss
        super().__init__()

    def fit(self, X, y, sample_weights=None):
        backend = PyPartitionTree(
            max_leaves=self.max_leaves,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_samples_xy=self.min_samples_xy,
            min_samples_x=self.min_samples_x,
            min_samples_y=self.min_samples_y,
            min_gain=self.min_gain,
            min_volume_fraction=self.min_volume_fraction,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            loss=self.loss,
        )

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        (
            X_pol,
            y_pol,
            self._y_dtype,
            self._y_columns,
            self._categorical_metadata,
        ) = _prepare_classification_training_data(X, y)

        self.classes_ = self.label_encoder_.classes_

        backend.fit(X_pol, y_pol, _wrap_sample_weights(sample_weights))
        self.partition_tree_ = backend
        return self

    def predict(self, X):
        check_is_fitted(self)
        probas = self.predict_proba(X)
        best_class_idx = np.argmax(probas, axis=1)
        return self.classes_[best_class_idx]

    def predict_proba(self, X):
        check_is_fitted(self)
        X_pol = _prepare_classification_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )
        piecewise_proba = self.partition_tree_.predict_proba(X_pol)
        return _probabilities_from_piecewise(
            piecewise_proba, self.classes_, self._y_columns[0]
        )

    def apply(self, X):
        check_is_fitted(self)
        X_pol = _prepare_classification_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )
        return self.partition_tree_.apply(X_pol)

    def get_leaves_info(self):
        check_is_fitted(self)
        return self.partition_tree_.get_leaves_info()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionForestClassifier(ClassifierMixin, BaseEstimator):
    _task = "classification"

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
        loss=None,
        random_state=None,
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
        self.loss = loss
        self.random_state = random_state
        super().__init__()

    def fit(self, X, y, sample_weights=None):
        backend = PyPartitionForest(
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
            max_samples=self.max_samples,
            max_features=self.max_features,
            loss=self.loss,
            seed=self.random_state,
        )

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        (
            X_pol,
            y_pol,
            self._y_dtype,
            self._y_columns,
            self._categorical_metadata,
        ) = _prepare_classification_training_data(X, y)

        self.classes_ = self.label_encoder_.classes_

        backend.fit(X_pol, y_pol, _wrap_sample_weights(sample_weights))
        self.partition_forest_ = backend
        return self

    def predict(self, X):
        check_is_fitted(self)
        X_pol = _prepare_classification_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )
        output_classes: pl.DataFrame = self.partition_forest_.predict(X_pol)
        return output_classes.to_pandas().astype(self._y_dtype)

    def predict_proba(self, X):
        check_is_fitted(self)
        X_pol = _prepare_classification_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )
        piecewise_proba = self.partition_forest_.predict_proba(X_pol)
        return _probabilities_from_piecewise(
            piecewise_proba, self.classes_, self._y_columns[0]
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionTreeRegressor(RegressorMixin, BaseEstimator):
    _task = "regression"

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
        loss=None,
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
        self.loss = loss
        super().__init__()

    def fit(self, X, y, sample_weights=None):
        X, y = validate_data(
            self, X, y, accept_sparse=False, multi_output=True, skip_check_array=True
        )

        backend = PyPartitionTree(
            max_leaves=self.max_leaves,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_samples_xy=self.min_samples_xy,
            min_samples_x=self.min_samples_x,
            min_samples_y=self.min_samples_y,
            min_gain=self.min_gain,
            min_volume_fraction=self.min_volume_fraction,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            loss=self.loss,
        )

        X_pol, y_pol, self._y_columns, self._categorical_metadata = (
            _prepare_regression_training_data(X, y)
        )

        backend.fit(X_pol, y_pol, _wrap_sample_weights(sample_weights))
        self.partition_tree_ = backend
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )
        X_pol = _prepare_regression_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )

        predictions = self.partition_tree_.predict(X_pol)
        predictions_df = pd.DataFrame(predictions, columns=self._y_columns)
        if len(self._y_columns) == 1:
            return predictions_df.iloc[:, 0].values
        return predictions_df.values

    def _predict_piecewise_proba(self, X):
        check_is_fitted(self)
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )
        X_pol = _prepare_regression_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )
        return self.partition_tree_.predict_proba(X_pol)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionForestRegressor(RegressorMixin, BaseEstimator):
    _task = "regression"

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
        loss=None,
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
        self.loss = loss
        self.random_state = random_state
        super().__init__()

    def fit(self, X, y, sample_weights=None):
        X, y = validate_data(
            self, X, y, accept_sparse=False, multi_output=True, skip_check_array=True
        )

        backend = PyPartitionForest(
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
            max_samples=self.max_samples,
            max_features=self.max_features,
            loss=self.loss,
            seed=self.random_state,
        )

        X_pol, y_pol, self._y_columns, self._categorical_metadata = (
            _prepare_regression_training_data(X, y)
        )

        backend.fit(X_pol, y_pol, _wrap_sample_weights(sample_weights))
        self.partition_tree_ = backend
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )
        X_pol = _prepare_regression_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )

        predictions = self.partition_tree_.predict(X_pol)
        predictions_df = pd.DataFrame(predictions, columns=self._y_columns)
        if len(self._y_columns) == 1:
            return predictions_df.iloc[:, 0].values
        return predictions_df.values

    def _predict_piecewise_proba(self, X):
        check_is_fitted(self)
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )
        X_pol = _prepare_regression_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )
        return self.partition_tree_.predict_proba(X_pol)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
