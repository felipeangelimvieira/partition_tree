from typing import List, Optional, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from partition_tree_python import PyPartitionForest, PyPartitionTree
import numpy as np
import pandas as pd
import polars as pl
import functools

from partition_tree.estimators.base import _preprocess, _preprocess_X
from skpro.regression.base import BaseProbaRegressor
from partition_tree.skpro.distribution import IntervalDistribution
from sklearn.preprocessing import LabelEncoder


def _convert_string_columns_to_categorical(
    df: pl.DataFrame,
    categories_map: Optional[dict] = None,
    return_categories: bool = False,
):
    """
    Convert string columns in a Polars DataFrame to categorical/enum type and optionally
    capture the category metadata for reuse (e.g., during predict).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame that may contain string columns.
    categories_map : dict, optional
        Mapping from column name to an ordered list of category values captured during
        ``fit``. When provided, the function will enforce these categories; missing
        columns will raise a ValueError.
    return_categories : bool, default False
        When True, also returns the categories_map used for conversion.

    Returns
    -------
    pl.DataFrame or Tuple[pl.DataFrame, dict]
        DataFrame with string columns converted to ``pl.Enum`` using the provided or
        inferred categories. If ``return_categories`` is True, returns a tuple of the
        converted DataFrame and the categories map.
    """

    if categories_map is not None:
        missing_cols = [col for col in categories_map.keys() if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "Missing expected categorical columns: " + ", ".join(missing_cols)
            )
        string_columns = list(categories_map.keys())
    else:
        # Capture both String (Utf8) and Categorical columns
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
    """Cast all integer columns to Float64 for both Polars and pandas DataFrames."""

    # Polars DataFrame handling
    if isinstance(df, pl.DataFrame):
        int_cols = [
            col for col, dtype in df.schema.items() if dtype in pl.INTEGER_DTYPES
        ]
        if not int_cols:
            return df
        return df.with_columns([pl.col(col).cast(pl.Float64) for col in int_cols])

    # Pandas DataFrame handling
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
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str], np.dtype, List[str], dict]:
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
    probabilities = np.empty((len(piecewise_proba), len(classes)))
    for i, piecewise_proba_instance in enumerate(piecewise_proba):
        probabilities[i, :] = [
            piecewise_proba_instance.pdf_single({target_column: str(class_)})
            for class_ in classes
        ]
    return probabilities


def _ensure_numeric_float64(df):
    """Coerce all numeric columns to float64 for pandas or Polars DataFrames."""

    # Polars DataFrame
    if isinstance(df, pl.DataFrame):
        numeric_cols = [
            col for col, dtype in df.schema.items() if dtype in pl.NUMERIC_DTYPES
        ]
        if not numeric_cols:
            return df
        return df.with_columns([pl.col(col).cast(pl.Float64) for col in numeric_cols])

    # Pandas DataFrame fallback
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


class PartitionTreeClassifier(ClassifierMixin, BaseEstimator):
    """
    Partition Tree classifier

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    max_depth_x : int, default=1000
        Maximum depth for feature splits.
    max_depth_t : int, default=1000
        Maximum depth for target splits.
    min_samples_leaf : int, default=1
        Minimum number of samples required in a leaf node.
    min_samples_split : int, default=1
        Minimum number of samples required to split an internal node.
    random_state : int, default=20
        Random state for reproducibility.
    objective : str, default="loglikelihood"
        Objective function for splitting criterion.
    max_splits_to_search : float, default=1e6
        Maximum number of splits to consider during tree building.
    min_split_volume : float, default=1e-6
        Minimum volume required for a split.
    tree_builder : {"depth-first", "best-first"}, default="depth-first"
        Tree building strategy:
        - "depth-first": Standard depth-first search (complete exploration)
        - "best-first": Priority-based search (focuses on promising nodes)
    max_nodes_to_expand : int, optional
        Maximum number of nodes to expand (only used with tree_builder="best-first").
        If None, no limit is imposed.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen at fit.
    """

    _task = "classification"

    def __init__(
        self,
        max_iter=1000,
        min_samples_split=1,
        min_samples_leaf_y=0,
        min_samples_leaf_x=0,
        min_samples_leaf=0,
        min_target_volume=0.0,
        max_depth=1000,
        min_split_gain=0.0,
        min_density_value=0.0,
        max_density_value=float("inf"),
        max_measure_value=float("inf"),
        boundaries_expansion_factor=0.0,
        exploration_split_budget: int = 0,
        feature_split_fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.max_iter = max_iter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf_y = min_samples_leaf_y
        self.min_samples_leaf_x = min_samples_leaf_x
        self.min_samples_leaf = min_samples_leaf
        self.min_target_volume = min_target_volume
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_density_value = min_density_value
        self.max_density_value = max_density_value
        self.max_measure_value = max_measure_value
        self.boundaries_expansion_factor = boundaries_expansion_factor
        self.exploration_split_budget = exploration_split_budget
        self.feature_split_fraction = feature_split_fraction
        self.seed = seed
        super().__init__()

    def fit(self, X, y, sample_weights=None):
        backend = PyPartitionTree(
            max_iter=self.max_iter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf_y=self.min_samples_leaf_y,
            min_samples_leaf_x=self.min_samples_leaf_x,
            min_samples_leaf=self.min_samples_leaf,
            min_target_volume=self.min_target_volume,
            max_depth=self.max_depth,
            min_split_gain=self.min_split_gain,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_density_value=self.min_density_value,
            max_density_value=self.max_density_value,
            max_measure_value=self.max_measure_value,
            exploration_split_budget=self.exploration_split_budget,
            feature_split_fraction=self.feature_split_fraction,
            seed=self.seed,
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
        self._X_pol = X_pol  # For debugging purposes
        backend.fit(X_pol, y_pol, sample_weights=sample_weights)
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

        masses_and_dists = self.partition_tree_.predict_categorical_masses(X_pol)

        # Create a mapping from string labels (as returned by Rust) to class indices
        # The labels from Rust are always strings, so we need to map them properly
        str_to_idx = {str(lbl): idx for idx, lbl in enumerate(self.classes_)}

        def flatten_to_vector(mass_classes, str_to_idx):
            class_dict = {}
            for mass, col_and_labels in mass_classes:
                cell_mass = mass

                for col, labels in col_and_labels.items():
                    if col not in class_dict:
                        class_dict[col] = np.zeros((len(str_to_idx),), dtype=float)
                    for lbl in labels:
                        # lbl is a string from Rust, look it up in str_to_idx
                        idx = str_to_idx[lbl]
                        class_dict[col][idx] = class_dict[col][idx] + cell_mass
            return {col: d for col, d in class_dict.items()}

        proba_dicts = list(
            map(
                functools.partial(flatten_to_vector, str_to_idx=str_to_idx),
                masses_and_dists,
            )
        )

        # assume a single column for the moment
        col_name = next(iter(proba_dicts[0].keys()))
        proba_array = (
            np.array([proba_dict[col_name] for proba_dict in proba_dicts]) + 1e-8
        )
        proba_array = proba_array / proba_array.sum(axis=1, keepdims=True)
        return proba_array

    def apply(self, X):
        """
        Apply the tree to the input data, returning the leaf index for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        leaf_indices : list of list of int
            For each sample, a list of leaf indices (positions in the leaves array)
            that the sample belongs to. Use get_leaves_info() to get details about
            each leaf by position.
        """
        check_is_fitted(self)
        X_pol = _prepare_classification_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )
        return self.partition_tree_.apply(X_pol)

    def get_leaves_info(self):
        """
        Get detailed information about all leaves in the tree.

        Returns
        -------
        leaves_info : list of dict
            A list of dictionaries, one per leaf, containing:
            - leaf_index: index of the leaf node in the tree
            - depth: depth of the leaf in the tree
            - n_samples: number of training samples in this leaf
            - partitions: dict mapping column names to their partition info
              - For continuous: {"type": "continuous", "low": float, "high": float,
                "lower_closed": bool, "upper_closed": bool}
              - For categorical: {"type": "categorical", "categories": [str, ...]}
            - indices_xy: list of sample indices matching both X and Y constraints
            - indices_x: list of sample indices matching X constraints only
            - indices_y: list of sample indices matching Y constraints only
        """
        check_is_fitted(self)
        return self.partition_tree_.get_leaves_info()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionForestClassifier(ClassifierMixin, BaseEstimator):
    """Partition Forest classifier."""

    _task = "classification"

    def __init__(
        self,
        n_estimators=100,
        max_iter=1000,
        min_samples_split=1,
        min_samples_leaf_y=0,
        min_samples_leaf_x=0,
        min_samples_leaf=0,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        min_density_value=0.0,
        max_density_value=float("inf"),
        max_measure_value=float("inf"),
        boundaries_expansion_factor=0.0,
        max_features: Optional[float] = 0.8,
        max_samples: Optional[float] = 0.8,
        exploration_split_budget: int = 0,
        feature_split_fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf_y = min_samples_leaf_y
        self.min_samples_leaf_x = min_samples_leaf_x
        self.min_samples_leaf = min_samples_leaf
        self.min_target_volume = min_target_volume
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_density_value = min_density_value
        self.max_density_value = max_density_value
        self.max_measure_value = max_measure_value
        self.boundaries_expansion_factor = boundaries_expansion_factor
        self.max_features = max_features
        self.max_samples = max_samples
        self.exploration_split_budget = exploration_split_budget
        self.feature_split_fraction = feature_split_fraction
        self.seed = seed

        super().__init__()

    def fit(self, X, y, sample_weights=None):
        backend = PyPartitionForest(
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf_y=self.min_samples_leaf_y,
            min_samples_leaf_x=self.min_samples_leaf_x,
            min_samples_leaf=self.min_samples_leaf,
            min_target_volume=self.min_target_volume,
            max_depth=self.max_depth,
            min_split_gain=self.min_split_gain,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            max_features=self.max_features,
            max_samples=self.max_samples,
            min_density_value=self.min_density_value,
            max_density_value=self.max_density_value,
            max_measure_value=self.max_measure_value,
            exploration_split_budget=self.exploration_split_budget,
            feature_split_fraction=self.feature_split_fraction,
            seed=self.seed,
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

        backend.fit(X_pol, y_pol, sample_weights=sample_weights)
        self.partition_forest_ = backend

        return self

    def predict(self, X):
        check_is_fitted(self)
        X_pol = _prepare_classification_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )
        output_classes: pl.DataFrame = self.partition_forest_.predict(X_pol)
        out = output_classes.to_pandas().astype(self._y_dtype)
        return out

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
    """
    Partition Tree regressor

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of iterations for tree building.
    min_samples_split : int, default=1
        Minimum number of samples required to split an internal node.
    min_samples_leaf_y : int, default=0
        Minimum number of samples required in a leaf node for target space.
    min_samples_leaf_x : int, default=0
        Minimum number of samples required in a leaf node for feature space.
    min_samples_leaf : int, default=0
        Minimum number of samples required in a leaf node (global constraint).
    min_target_volume : float, default=0.0
        Minimum volume required in the target space for a leaf.
    max_depth : int, default=5
        Maximum depth of the tree.
    min_split_gain : float, default=0.0
        Minimum gain required for a split.
    boundaries_expansion_factor : float, default=0.0
        Factor for expanding decision boundaries.

    Attributes
    ----------
    partition_tree_ : PyPartitionTree
        The fitted partition tree instance.
    """

    _task = "regression"

    def __init__(
        self,
        max_iter=1000,
        min_samples_split=1,
        min_samples_leaf_y=0,
        min_samples_leaf_x=0,
        min_samples_leaf=0,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        min_density_value=0.0,
        max_density_value=float("inf"),
        max_measure_value=float("inf"),
        boundaries_expansion_factor=0.0,
        exploration_split_budget: int = 0,
        feature_split_fraction: Optional[float] = None,
        seed=42,
    ):
        self.max_iter = max_iter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf_y = min_samples_leaf_y
        self.min_samples_leaf_x = min_samples_leaf_x
        self.min_samples_leaf = min_samples_leaf
        self.min_target_volume = min_target_volume
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_density_value = min_density_value
        self.max_density_value = max_density_value
        self.max_measure_value = max_measure_value
        self.boundaries_expansion_factor = boundaries_expansion_factor
        self.exploration_split_budget = exploration_split_budget
        self.feature_split_fraction = feature_split_fraction
        self.seed = seed

        super().__init__()

    def fit(self, X, y, sample_weights=None):
        """
        Fit the partition tree regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input using sklearn's validate_data (multi_output=True for multi-target regression)
        X, y = validate_data(
            self, X, y, accept_sparse=False, multi_output=True, skip_check_array=True
        )

        self._min_target_volume = self.min_target_volume * (y.max() - y.min())
        backend = PyPartitionTree(
            max_iter=self.max_iter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf_y=self.min_samples_leaf_y,
            min_samples_leaf_x=self.min_samples_leaf_x,
            min_samples_leaf=self.min_samples_leaf,
            min_target_volume=self._min_target_volume,
            max_depth=self.max_depth,
            min_split_gain=self.min_split_gain,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_density_value=self.min_density_value,
            max_density_value=self.max_density_value,
            max_measure_value=self.max_measure_value,
            exploration_split_budget=self.exploration_split_budget,
            feature_split_fraction=self.feature_split_fraction,
            seed=self.seed,
        )

        X_pol, y_pol, self._y_columns, self._categorical_metadata = (
            _prepare_regression_training_data(X, y)
        )

        if sample_weights is not None:
            sample_weights = pl.DataFrame({"sample_weights": sample_weights})

        backend.fit(X_pol, y_pol, sample_weights)
        self.partition_tree_ = backend

        return self

    def predict(self, X):
        """
        Predict regression targets for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted target values.
        """
        check_is_fitted(self)

        # Validate input using sklearn's validate_data (reset=False to check consistency)
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )

        # Convert to DataFrame for preprocessing
        # Convert X to DataFrame for preprocessing
        X_pol = _prepare_regression_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )

        # Get predictions from the partition tree
        predictions = self.partition_tree_.predict(X_pol)
        predictions_df = pd.DataFrame(predictions, columns=self._y_columns)

        # Return as numpy array with appropriate shape
        if len(self._y_columns) == 1:
            return predictions_df.iloc[:, 0].values
        else:
            return predictions_df.values

    def _predict_piecewise_proba(self, X):
        """
        Predict piecewise probability distributions for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        distributions : list of IntervalDistribution
            List of predicted piecewise probability distributions.
        """
        check_is_fitted(self)

        # Validate input using sklearn's validate_data (reset=False to check consistency)
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )

        # Convert to DataFrame for preprocessing
        X_pol = _prepare_regression_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )

        # Get piecewise probability distributions from the partition tree
        piecewise_proba = self.partition_tree_.predict_proba(X_pol)

        return piecewise_proba

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionForestRegressor(RegressorMixin, BaseEstimator):
    """
    Partition Tree regressor

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of iterations for tree building.
    min_samples_split : int, default=1
        Minimum number of samples required to split an internal node.
    min_samples_leaf_y : int, default=0
        Minimum number of samples required in a leaf node for target space.
    min_samples_leaf_x : int, default=0
        Minimum number of samples required in a leaf node for feature space.
    min_samples_leaf : int, default=0
        Minimum number of samples required in a leaf node (global constraint).
    min_target_volume : float, default=0.0
        Minimum volume required in the target space for a leaf.
    max_depth : int, default=5
        Maximum depth of the tree.
    min_split_gain : float, default=0.0
        Minimum gain required for a split.
    boundaries_expansion_factor : float, default=0.0
        Factor for expanding decision boundaries.

    Attributes
    ----------
    partition_tree_ : PyPartitionTree
        The fitted partition tree instance.
    """

    _task = "regression"

    def __init__(
        self,
        n_estimators=100,
        max_iter=1000,
        min_samples_split=1,
        min_samples_leaf_y=0,
        min_samples_leaf_x=0,
        min_samples_leaf=0,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        min_density_value=0.0,
        max_density_value=float("inf"),
        max_measure_value=float("inf"),
        boundaries_expansion_factor=0.0,
        max_samples: Optional[float] = 0.8,
        max_features: Optional[float] = 0.8,
        exploration_split_budget: int = 0,
        feature_split_fraction: Optional[float] = None,
        seed=42,
    ):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf_y = min_samples_leaf_y
        self.min_samples_leaf_x = min_samples_leaf_x
        self.min_samples_leaf = min_samples_leaf
        self.min_target_volume = min_target_volume
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_density_value = min_density_value
        self.max_density_value = max_density_value
        self.max_measure_value = max_measure_value
        self.boundaries_expansion_factor = boundaries_expansion_factor
        self.seed = seed
        self.max_samples = max_samples
        self.max_features = max_features
        self.exploration_split_budget = exploration_split_budget
        self.feature_split_fraction = feature_split_fraction

        super().__init__()

    def fit(self, X, y, sample_weights=None):
        """
        Fit the partition tree regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input using sklearn's validate_data (multi_output=True for multi-target regression)
        X, y = validate_data(
            self, X, y, accept_sparse=False, multi_output=True, skip_check_array=True
        )

        self._min_target_volume = self.min_target_volume * (y.max() - y.min())
        backend = PyPartitionForest(
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf_y=self.min_samples_leaf_y,
            min_samples_leaf_x=self.min_samples_leaf_x,
            min_samples_leaf=self.min_samples_leaf,
            min_target_volume=self._min_target_volume,
            max_depth=self.max_depth,
            min_split_gain=self.min_split_gain,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            seed=self.seed,
            max_samples=self.max_samples,
            max_features=self.max_features,
            min_density_value=self.min_density_value,
            max_density_value=self.max_density_value,
            max_measure_value=self.max_measure_value,
            exploration_split_budget=self.exploration_split_budget,
            feature_split_fraction=self.feature_split_fraction,
        )

        X_pol, y_pol, self._y_columns, self._categorical_metadata = (
            _prepare_regression_training_data(X, y)
        )

        if sample_weights is not None:
            sample_weights = pl.DataFrame({"sample_weights": sample_weights})

        backend.fit(X_pol, y_pol, sample_weights)
        self.partition_tree_ = backend

        return self

    def predict(self, X):
        """
        Predict regression targets for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted target values.
        """
        check_is_fitted(self)

        # Validate input using sklearn's validate_data (reset=False to check consistency)
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )

        # Convert to DataFrame for preprocessing
        # Convert X to DataFrame for preprocessing
        X_pol = _prepare_regression_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )

        # Get predictions from the partition tree
        predictions = self.partition_tree_.predict(X_pol)
        predictions_df = pd.DataFrame(predictions, columns=self._y_columns)

        # Return as numpy array with appropriate shape
        if len(self._y_columns) == 1:
            return predictions_df.iloc[:, 0].values
        else:
            return predictions_df.values

    def _predict_piecewise_proba(self, X):
        """
        Predict piecewise probability distributions for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        distributions : list of IntervalDistribution
            List of predicted piecewise probability distributions.
        """
        check_is_fitted(self)

        # Validate input using sklearn's validate_data (reset=False to check consistency)
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )

        # Convert to DataFrame for preprocessing
        X_pol = _prepare_regression_features(
            X, categorical_metadata=getattr(self, "_categorical_metadata", None)
        )

        # Get piecewise probability distributions from the partition tree
        piecewise_proba = self.partition_tree_.predict_proba(X_pol)

        return piecewise_proba

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionTreeRegressorSkpro(BaseProbaRegressor):
    """

    A probabilistic tree-based regressor that models the joint distribution.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    max_depth_x : int, default=1000
        Maximum depth for feature splits.
    max_depth_t : int, default=1000
        Maximum depth for target splits.
    min_samples_leaf : int, default=1
        Minimum number of samples required in a leaf node.
    min_samples_split : int, default=1
        Minimum number of samples required to split an internal node.
    objective : str, default="loglikelihood"
        Objective function for splitting criterion.
    random_state : int, default=20
        Random state for reproducibility.
    predict_method : str, default="mean"
        Method for making predictions.
    max_splits_to_search : float, default=1e10
        Maximum number of splits to consider during tree building.
    min_split_volume : float, default=1e-6
        Minimum volume required for a split.
    tree_builder : {"depth-first", "best-first"}, default="depth-first"
        Tree building strategy:
        - "depth-first": Standard depth-first search (complete exploration)
        - "best-first": Priority-based search (focuses on promising nodes)
    max_nodes_to_expand : int, optional
        Maximum number of nodes to expand (only used with tree_builder="best-first").
        If None, no limit is imposed.
    """

    _task = "regression"

    def __init__(
        self,
        max_iter=1000,
        min_samples_split=1,
        min_samples_leaf_y=0,
        min_samples_leaf_x=0,
        min_samples_leaf=0,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        min_density_value=0.0,
        max_density_value=float("inf"),
        max_measure_value=float("inf"),
        boundaries_expansion_factor=0.0,
        exploration_split_budget: int = 0,
        feature_split_fraction: Optional[float] = None,
        seed: Optional[int] = 42,
    ):
        self.max_iter = max_iter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf_y = min_samples_leaf_y
        self.min_samples_leaf_x = min_samples_leaf_x
        self.min_samples_leaf = min_samples_leaf
        self.min_target_volume = min_target_volume
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_density_value = min_density_value
        self.max_density_value = max_density_value
        self.max_measure_value = max_measure_value
        self.boundaries_expansion_factor = boundaries_expansion_factor
        self.exploration_split_budget = exploration_split_budget
        self.feature_split_fraction = feature_split_fraction
        self.seed = seed
        super().__init__()

    def _fit(self, X, y):

        self._min_target_volume = self.min_target_volume * (y.max() - y.min())

        exploration_budget = self.exploration_split_budget
        if self.exploration_split_budget <= 1:
            exploration_budget = int(self.max_iter * self.exploration_split_budget)

        self.partition_tree_ = PyPartitionTree(
            max_iter=self.max_iter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf_y=self.min_samples_leaf_y,
            min_samples_leaf_x=self.min_samples_leaf_x,
            min_samples_leaf=self.min_samples_leaf,
            min_target_volume=self._min_target_volume,
            max_depth=self.max_depth,
            min_split_gain=self.min_split_gain,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            min_density_value=self.min_density_value,
            max_density_value=self.max_density_value,
            max_measure_value=self.max_measure_value,
            exploration_split_budget=exploration_budget,
            feature_split_fraction=self.feature_split_fraction,
            seed=self.seed,
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
        # Use pdf_with_intervals to preserve per-row interval boundaries and densities
        intervals_per_row = []
        pdf_values_per_row = []
        for dist in piecewise_proba:
            row_intervals = []
            row_pdfs = []
            for pdf_val, (
                low,
                high,
                lower_closed,
                upper_closed,
            ) in dist.pdf_with_intervals():
                row_intervals.append(
                    (float(low), float(high), lower_closed, upper_closed)
                )
                row_pdfs.append(float(pdf_val))

            # Sort per low
            sorted_indices = np.argsort([interval[0] for interval in row_intervals])
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
        """
        Apply the tree to the input data, returning the leaf index for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        leaf_indices : list of list of int
            For each sample, a list of leaf indices (positions in the leaves array)
            that the sample belongs to. Use get_leaves_info() to get details about
            each leaf by position.
        """
        check_is_fitted(self)
        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)
        return self.partition_tree_.apply(X_pol)

    def get_leaves_info(self):
        """
        Get detailed information about all leaves in the tree.

        Returns
        -------
        leaves_info : list of dict
            A list of dictionaries, one per leaf, containing:
            - leaf_index: index of the leaf node in the tree
            - depth: depth of the leaf in the tree
            - n_samples: number of training samples in this leaf
            - partitions: dict mapping column names to their partition info
              - For continuous: {"type": "continuous", "low": float, "high": float,
                "lower_closed": bool, "upper_closed": bool}
              - For categorical: {"type": "categorical", "categories": [str, ...]}
            - indices_xy: list of sample indices matching both X and Y constraints
            - indices_x: list of sample indices matching X constraints only
            - indices_y: list of sample indices matching Y constraints only
            - feature_contributions: dict mapping column names to cumulative gain
        """
        check_is_fitted(self)
        return self.partition_tree_.get_leaves_info()

    def get_feature_importances(self, normalize: bool = True) -> dict:
        """
        Compute feature importances based on cumulative gain contributions.

        Feature importance for each feature is computed by summing the gains
        from all splits on that feature. Each split is counted exactly once
        (no double-counting across leaves).

        Parameters
        ----------
        normalize : bool, default=True
            If True, normalize importances to sum to 1.0.

        Returns
        -------
        feature_importances : dict
            Dictionary mapping feature names to their importance scores.
            Higher values indicate more important features.
        """
        check_is_fitted(self)
        importances = self.partition_tree_.get_feature_importances(normalize)

        # Sort by importance (descending)
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class PartitionForestRegressorSkpro(BaseProbaRegressor):
    """

    A probabilistic tree-based regressor that models the joint distribution.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    max_depth_x : int, default=1000
        Maximum depth for feature splits.
    max_depth_t : int, default=1000
        Maximum depth for target splits.
    min_samples_leaf : int, default=1
        Minimum number of samples required in a leaf node.
    min_samples_split : int, default=1
        Minimum number of samples required to split an internal node.
    objective : str, default="loglikelihood"
        Objective function for splitting criterion.
    random_state : int, default=20
        Random state for reproducibility.
    predict_method : str, default="mean"
        Method for making predictions.
    max_splits_to_search : float, default=1e10
        Maximum number of splits to consider during tree building.
    min_split_volume : float, default=1e-6
        Minimum volume required for a split.
    tree_builder : {"depth-first", "best-first"}, default="depth-first"
        Tree building strategy:
        - "depth-first": Standard depth-first search (complete exploration)
        - "best-first": Priority-based search (focuses on promising nodes)
    max_nodes_to_expand : int, optional
        Maximum number of nodes to expand (only used with tree_builder="best-first").
        If None, no limit is imposed.
    """

    _task = "regression"

    def __init__(
        self,
        n_estimators=100,
        max_iter=1000,
        min_samples_split=1,
        min_samples_leaf_y=0,
        min_samples_leaf_x=0,
        min_samples_leaf=0,
        min_target_volume=0.0,
        max_depth=5,
        min_split_gain=0.0,
        min_density_value=0.0,
        max_density_value=float("inf"),
        max_measure_value=float("inf"),
        boundaries_expansion_factor=0.0,
        max_samples=None,
        max_features=0.8,
        exploration_split_budget: int = 0,
        feature_split_fraction: Optional[float] = None,
        seed=42,
    ):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf_y = min_samples_leaf_y
        self.min_samples_leaf_x = min_samples_leaf_x
        self.min_samples_leaf = min_samples_leaf
        self.min_target_volume = min_target_volume
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_density_value = min_density_value
        self.max_density_value = max_density_value
        self.max_measure_value = max_measure_value
        self.boundaries_expansion_factor = boundaries_expansion_factor
        self.max_samples = max_samples
        self.max_features = max_features
        self.exploration_split_budget = exploration_split_budget
        self.feature_split_fraction = feature_split_fraction
        self.seed = seed
        super().__init__()

    def _fit(self, X, y):

        self._min_target_volume = self.min_target_volume * (y.max() - y.min())

        self.partition_forest_ = PyPartitionForest(
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf_y=self.min_samples_leaf_y,
            min_samples_leaf_x=self.min_samples_leaf_x,
            min_samples_leaf=self.min_samples_leaf,
            min_target_volume=self._min_target_volume,
            max_depth=self.max_depth,
            min_split_gain=self.min_split_gain,
            boundaries_expansion_factor=self.boundaries_expansion_factor,
            seed=self.seed,
            max_samples=self.max_samples,
            max_features=self.max_features,
            min_density_value=self.min_density_value,
            max_density_value=self.max_density_value,
            max_measure_value=self.max_measure_value,
            exploration_split_budget=self.exploration_split_budget,
            feature_split_fraction=self.feature_split_fraction,
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

    def _predict_proba(self, X):

        X_proc = _ensure_numeric_float64(_preprocess_X(X))
        X_pol = pl.DataFrame(X_proc)
        X_pol = _convert_string_columns_to_categorical(
            X_pol, categories_map=getattr(self, "_categorical_metadata", None)
        )
        X_pol = _ensure_numeric_float64(X_pol)

        piecewise_probas = self.partition_forest_.predict_trees_proba(X_pol)

        interval_dists = []
        for piecewise_proba in piecewise_probas:
            # Use pdf_with_intervals to preserve per-row interval boundaries and densities
            intervals_per_row = []
            pdf_values_per_row = []
            for dist in piecewise_proba:
                row_intervals = []
                row_pdfs = []
                for pdf_val, (
                    low,
                    high,
                    lower_closed,
                    upper_closed,
                ) in dist.pdf_with_intervals():
                    row_intervals.append(
                        (float(low), float(high), lower_closed, upper_closed)
                    )
                    row_pdfs.append(float(pdf_val))

                # Sort per low
                sorted_indices = np.argsort([interval[0] for interval in row_intervals])
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

        from skpro.distributions import Mixture

        mixture_piecewise = IntervalDistribution.from_mixture(
            distributions=interval_dists,
            weights=[1.0 / self.n_estimators] * self.n_estimators,
            index=X_proc.index,
            columns=self._y_columns,
        )
        return mixture_piecewise

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
