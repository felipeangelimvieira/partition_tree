from collections import Counter, defaultdict
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
) -> Tuple[pl.DataFrame, pl.DataFrame, np.dtype, List[str], dict, List[str]]:
    y_df = _coerce_target_dataframe(y)
    y_dtype = y_df.dtypes[0]
    original_y_columns = y_df.columns.to_list()

    categorical_targets = original_y_columns if isinstance(y, pd.DataFrame) else None
    X_processed, y_processed = _preprocess(
        X, y_df, categorical_targets=categorical_targets
    )

    # Use the processed column names (after _preprocess_y renaming) because
    # this is what the Rust backend will see.  When y arrives as a DataFrame
    # with a column not called "target" (e.g. "lettr"), _preprocess_y renames
    # it to "target_lettr".  We must use that name during prediction so the
    # lookup into the Rust-returned distributions succeeds.
    y_columns = y_processed.columns.to_list()

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

    enum_categories = list(y_processed[target_column].cat.categories)

    X_pol = pl.DataFrame(_convert_int_to_float64(X_processed))
    X_pol = _convert_int_to_float64(X_pol)
    X_pol, categorical_metadata = _convert_string_columns_to_categorical(
        X_pol, return_categories=True
    )

    # Build the y polars DataFrame as an Enum column.
    #
    # The Rust backend receives the target as an Enum column (via pyo3-polars).
    # Enum physical codes are deterministic: they equal the position of each
    # category in the ``enum_categories`` list (which is lexicographically
    # sorted).  The Rust tree builder sees these codes and creates domain
    # names ``cat_0``, ``cat_1``, … corresponding to positions 0, 1, … in
    # the Enum category list.
    #
    # Therefore the correct mapping is simply ``cat_N → enum_categories[N]``.
    # A previous attempt tried to reconstruct the mapping from a pl.Categorical
    # Series (using the Polars global string cache), but Categorical physical
    # codes depend on insertion order / cache state and do NOT match the Enum
    # ordering that Rust actually observes.
    cat_to_class = list(enum_categories)  # cat_N → enum_categories[N]

    y_pol_cat = pl.DataFrame(y_processed)
    y_pol = y_pol_cat.cast(pl.Enum(categories=enum_categories))

    return X_pol, y_pol, y_dtype, y_columns, categorical_metadata, cat_to_class


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
    piecewise_proba,
    classes: List[str],
    target_column: str,
    enum_categories: Optional[List[str]] = None,
) -> np.ndarray:
    n_samples = len(piecewise_proba)
    n_classes = len(classes)
    probabilities = np.empty((n_samples, n_classes))

    # Build a mapping from cat_N domain names to original class strings
    # The Rust backend returns domain names like ['cat_0', 'cat_1', ...]
    # which correspond positionally to enum_categories ['0', '1', ...]
    cat_to_class: Optional[dict] = None
    if enum_categories is not None:
        cat_to_class = {f"cat_{i}": cat for i, cat in enumerate(enum_categories)}

    for i, dist in enumerate(piecewise_proba):
        cat_probs = dist.category_probabilities()
        domain_names = dist.categorical_domain_names(target_column)

        if target_column in cat_probs and domain_names is not None:
            probs = cat_probs[target_column]
            # Remap cat_N domain names to actual category strings
            if cat_to_class is not None:
                resolved_names = [cat_to_class.get(n, n) for n in domain_names]
            else:
                resolved_names = domain_names
            name_to_prob = dict(zip(resolved_names, probs))
            probabilities[i, :] = [name_to_prob.get(str(c), 0.0) for c in classes]
        else:
            probabilities[i, :] = 0.0

    probabilities = probabilities + 1e-8
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _build_leaf_class_probs(backend, X_pol, y, classes):
    """Build per-leaf class frequency distributions from training data.

    The partition tree groups categorical target values into partitions and
    assigns uniform density within each group.  This function computes the
    actual class frequencies within each x-partition leaf so that
    ``_refine_probabilities_with_leaf_frequencies`` can break those ties.
    """
    if not hasattr(backend, "apply"):
        return None

    leaf_ids = backend.apply(X_pol)
    y_flat = np.asarray(y).ravel()

    leaf_class_counts = defaultdict(Counter)
    for leaf_id, label in zip(leaf_ids, y_flat):
        leaf_class_counts[leaf_id][label] += 1

    leaf_class_probs = {}
    for leaf_id, counts in leaf_class_counts.items():
        total_in_leaf = sum(counts.values())
        leaf_class_probs[leaf_id] = {c: n / total_in_leaf for c, n in counts.items()}
    return leaf_class_probs


def _refine_probabilities_with_leaf_frequencies(
    probabilities, leaf_ids, leaf_class_probs, classes
):
    """Refine density-based probabilities using per-leaf class frequencies.

    The density estimator assigns uniform probability to all categories
    within the same target-space partition group.  This function preserves
    the inter-group probability ratios but redistributes probability
    *within* each tied group according to the empirical class frequencies
    observed in the corresponding x-partition leaf.
    """
    if leaf_class_probs is None or leaf_ids is None:
        return probabilities

    refined = probabilities.copy()

    for i, leaf_id in enumerate(leaf_ids):
        freqs = leaf_class_probs.get(leaf_id)
        if freqs is None:
            continue

        row = refined[i]
        # Identify groups of classes sharing the same probability (tied)
        unique_probs = np.unique(np.round(row, 12))
        for p in unique_probs:
            tied_mask = np.isclose(row, p, rtol=1e-10)
            n_tied = tied_mask.sum()
            if n_tied <= 1:
                continue

            # Redistribute this group's total probability by class frequency
            tied_indices = np.where(tied_mask)[0]
            group_total = p * n_tied
            freq_weights = np.array([freqs.get(classes[j], 1e-8) for j in tied_indices])
            freq_sum = freq_weights.sum()
            if freq_sum > 0:
                for k, j in enumerate(tied_indices):
                    refined[i, j] = group_total * freq_weights[k] / freq_sum

    return refined


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
        max_leaves=None,
        boundaries_expansion_factor=0.0,
        min_samples_xy=1.0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0.0,
        max_depth=None,
        min_samples_split=2.0,
        loss=None,
        random_state=42,
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
        self.random_state = random_state
        super().__init__()

    @property
    def _max_leaves(self):
        return self.max_leaves if self.max_leaves is not None else int(1e6)

    @property
    def _max_depth(self):
        return self.max_depth if self.max_depth is not None else int(1e6)

    def fit(self, X, y, sample_weights=None):
        backend = PyPartitionTree(
            max_leaves=self._max_leaves,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_samples_xy=self.min_samples_xy,
            min_samples_x=self.min_samples_x,
            min_samples_y=self.min_samples_y,
            min_gain=self.min_gain,
            min_volume_fraction=self.min_volume_fraction,
            max_depth=self._max_depth,
            min_samples_split=self.min_samples_split,
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
            self._enum_categories,
        ) = _prepare_classification_training_data(X, y)

        self.classes_ = self.label_encoder_.classes_

        backend.fit(X_pol, y_pol, _wrap_sample_weights(sample_weights))
        self.partition_tree_ = backend

        # Build per-leaf class frequency table for refining within-group probs
        self._leaf_class_probs = _build_leaf_class_probs(
            self.partition_tree_, X_pol, y, self.classes_
        )

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
        probas = _probabilities_from_piecewise(
            piecewise_proba,
            self.classes_,
            self._y_columns[0],
            enum_categories=self._enum_categories,
        )
        leaf_ids = self.partition_tree_.apply(X_pol)
        return _refine_probabilities_with_leaf_frequencies(
            probas, leaf_ids, self._leaf_class_probs, self.classes_
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
        max_leaves=None,
        boundaries_expansion_factor=0.0,
        min_samples_xy=1.0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0,
        max_depth=None,
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

    @property
    def _max_leaves(self):
        return self.max_leaves if self.max_leaves is not None else int(1e6)

    @property
    def _max_depth(self):
        return self.max_depth if self.max_depth is not None else int(1e6)

    def fit(self, X, y, sample_weights=None):
        backend = PyPartitionForest(
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
            self._enum_categories,
        ) = _prepare_classification_training_data(X, y)

        self.classes_ = self.label_encoder_.classes_

        backend.fit(X_pol, y_pol, _wrap_sample_weights(sample_weights))
        self.partition_forest_ = backend

        # Build per-leaf class frequency table for refining within-group probs
        self._leaf_class_probs = _build_leaf_class_probs(
            self.partition_forest_, X_pol, y, self.classes_
        )

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
        piecewise_proba = self.partition_forest_.predict_proba(X_pol)
        probas = _probabilities_from_piecewise(
            piecewise_proba,
            self.classes_,
            self._y_columns[0],
            enum_categories=self._enum_categories,
        )
        if hasattr(self.partition_forest_, "apply"):
            leaf_ids = self.partition_forest_.apply(X_pol)
        else:
            leaf_ids = None
        return _refine_probabilities_with_leaf_frequencies(
            probas, leaf_ids, self._leaf_class_probs, self.classes_
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionTreeRegressor(RegressorMixin, BaseEstimator):
    _task = "regression"

    def __init__(
        self,
        max_leaves=None,
        boundaries_expansion_factor=0.0,
        min_samples_xy=1.0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0.1,
        max_depth=None,
        min_samples_split=2.0,
        loss=None,
        random_state=42,
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
        self.random_state = random_state
        super().__init__()

    @property
    def _max_leaves(self):
        return self.max_leaves if self.max_leaves is not None else int(1e6)

    @property
    def _max_depth(self):
        return self.max_depth if self.max_depth is not None else int(1e6)

    def fit(self, X, y, sample_weights=None):
        X, y = validate_data(
            self, X, y, accept_sparse=False, multi_output=True, skip_check_array=True
        )

        backend = PyPartitionTree(
            max_leaves=self._max_leaves,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_samples_xy=self.min_samples_xy,
            min_samples_x=self.min_samples_x,
            min_samples_y=self.min_samples_y,
            min_gain=self.min_gain,
            min_volume_fraction=self.min_volume_fraction,
            max_depth=self._max_depth,
            min_samples_split=self.min_samples_split,
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


class PartitionForestRegressor(RegressorMixin, BaseEstimator):
    _task = "regression"

    def __init__(
        self,
        n_estimators=100,
        max_leaves=None,
        boundaries_expansion_factor=0.0,
        min_samples_xy=1.0,
        min_samples_x=1.0,
        min_samples_y=1.0,
        min_gain=0.0,
        min_volume_fraction=0.1,
        max_depth=None,
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

    @property
    def _max_leaves(self):
        return self.max_leaves if self.max_leaves is not None else int(1e6)

    @property
    def _max_depth(self):
        return self.max_depth if self.max_depth is not None else int(1e6)

    def fit(self, X, y, sample_weights=None):
        X, y = validate_data(
            self, X, y, accept_sparse=False, multi_output=True, skip_check_array=True
        )

        backend = PyPartitionForest(
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
