from decimal import Decimal, InvalidOperation
from math import gcd
from collections.abc import Mapping
from typing import List, Optional, Tuple

from partition_tree import TARGET_COLUMN
import numpy as np
import pandas as pd
import polars as pl


_AUTO_DTYPE_OVERRIDE = "auto"
_MAX_AUTO_QUANTIZED_DECIMALS = 9


def _preprocess_X(X):
    """Preprocess feature matrix X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features.

    Returns
    -------
    X : pd.DataFrame
        Preprocessed feature matrix.
    """
    # Handle various input types
    if hasattr(X, "copy"):
        X = X.copy()

    # Check for sparse matrices and reject them with proper error message
    from scipy import sparse

    if sparse.issparse(X):
        raise ValueError(
            "Sparse input is not supported. Please convert to dense array using .toarray()"
        )

    # Check if X is an object array containing sparse matrices
    if isinstance(X, np.ndarray) and X.dtype == object and X.size > 0:
        first_element = X.flat[0] if X.size > 0 else None
        if sparse.issparse(first_element):
            raise ValueError(
                "Sparse input is not supported. Please convert to dense array using .toarray()"
            )

    if not isinstance(X, pd.DataFrame):
        # Convert to numpy array first to handle lists and other types
        X = np.asarray(X)
        # Handle 1D arrays by reshaping them
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Make sure we have a valid 2D array
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if X.shape[1] == 0:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
            )
        X = pd.DataFrame(X, columns=[f"{i}" for i in range(X.shape[1])])

    # Convert non-numeric columns to categorical
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.Categorical(X[col])

    X.columns = [str(col) for col in X.columns]
    # X = pl.DataFrame(X)
    return X


def _preprocess_y(y, categorical_targets=None):
    """Preprocess target variable y.

    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    categorical_targets : list of str, optional
        List of target column names that should be treated as categorical.

    Returns
    -------
    y : pd.DataFrame
        Preprocessed target matrix.
    categorical_targets : list of str or None
        Updated categorical target column names.
    """
    if y is None:
        raise ValueError("y cannot be None for supervised learning")

    if hasattr(y, "copy"):
        y = y.copy()

    if not isinstance(y, pd.DataFrame):
        y = np.asarray(y)
        # Handle various y shapes
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = pd.DataFrame(y, columns=[TARGET_COLUMN])
    elif TARGET_COLUMN not in y.columns:
        y = y.copy()
        y.columns = [TARGET_COLUMN + "_" + str(col) for col in y.columns]

        if categorical_targets is not None:
            categorical_targets = [
                TARGET_COLUMN + "_" + str(col) for col in categorical_targets
            ]

    if categorical_targets is not None:
        for col in categorical_targets:
            if col in y.columns:
                y[col] = pd.Categorical(y[col])

    y.columns = [str(col) for col in y.columns]
    # y = pl.DataFrame(y)
    return y, categorical_targets


def _preprocess(X, y, categorical_targets=None):
    """Preprocess both X and y using separate preprocessing functions.

    This function maintains backward compatibility by calling the separate
    preprocessing functions.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    categorical_targets : list of str, optional
        List of target column names that should be treated as categorical.

    Returns
    -------
    X : pd.DataFrame
        Preprocessed feature matrix.
    y : pd.DataFrame
        Preprocessed target matrix.
    """
    X = _preprocess_X(X)
    y, categorical_targets = _preprocess_y(y, categorical_targets)
    return X, y


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


def _decimal_from_float(value: float) -> Decimal:
    try:
        return Decimal(str(float(value)))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not interpret {value!r} as a finite decimal value"
        ) from exc


def _infer_quantized_resolution(series: pd.Series) -> Optional[float]:
    non_null = series.dropna()
    if non_null.empty:
        return None

    finite_values = [float(value) for value in non_null.to_list() if np.isfinite(value)]
    if not finite_values:
        return None

    decimals = [_decimal_from_float(value) for value in finite_values]
    max_scale = max(
        max(-decimal.normalize().as_tuple().exponent, 0) for decimal in decimals
    )
    if max_scale > _MAX_AUTO_QUANTIZED_DECIMALS:
        return None

    scale_factor = Decimal(10) ** max_scale
    scaled_values = [
        int((decimal * scale_factor).to_integral_value()) for decimal in decimals
    ]

    quanta = []
    for scaled in scaled_values:
        if scaled != 0:
            quanta.append(abs(scaled))

    unique_scaled = sorted(set(scaled_values))
    quanta.extend(
        abs(right - left)
        for left, right in zip(unique_scaled, unique_scaled[1:])
        if right != left
    )
    quanta = [quantum for quantum in quanta if quantum > 0]
    if not quanta:
        return None

    gcd_scaled = quanta[0]
    for quantum in quanta[1:]:
        gcd_scaled = gcd(gcd_scaled, quantum)

    if gcd_scaled <= 0:
        return None

    resolution = float(Decimal(gcd_scaled) / scale_factor)
    if resolution <= 0.0 or not np.isfinite(resolution):
        return None

    max_abs_value = max(abs(value) for value in finite_values)
    if max_abs_value / resolution > np.iinfo(np.int64).max:
        return None

    scaled = np.asarray(finite_values, dtype=np.float64) / resolution
    rounded = np.round(scaled)
    tolerance = np.maximum(np.abs(scaled), 1.0) * 1e-9
    if not np.all(np.abs(scaled - rounded) <= tolerance):
        return None

    return resolution


def _resolve_regression_dtype_overrides(
    dtype_overrides,
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    X_processed: pd.DataFrame,
    y_processed: pd.DataFrame,
):
    from pyo3_partition_tree import Domain

    if dtype_overrides is None:
        return None

    if isinstance(dtype_overrides, str):
        if dtype_overrides != _AUTO_DTYPE_OVERRIDE:
            raise ValueError(
                "dtype_overrides must be None, a mapping, or 'auto' for regression"
            )
        dtype_overrides = {
            str(column): _AUTO_DTYPE_OVERRIDE for column in y_raw.columns
        }
    elif not isinstance(dtype_overrides, Mapping):
        raise ValueError(
            "dtype_overrides must be None, a mapping, or 'auto' for regression"
        )

    resolved = {}
    if y_raw.shape[1] == y_processed.shape[1]:
        target_name_map = dict(
            zip(map(str, y_raw.columns), map(str, y_processed.columns))
        )
    else:
        target_name_map = {}

    for column, override in dtype_overrides.items():
        column = str(column)
        if override != _AUTO_DTYPE_OVERRIDE:
            resolved_column = (
                target_name_map[column]
                if column in target_name_map
                and column not in X_processed.columns
                and column not in y_processed.columns
                else column
            )
            resolved[resolved_column] = override
            continue

        if column in X_processed.columns:
            processed_column = column
            series = X_processed[column]
        elif column in y_processed.columns:
            processed_column = column
            series = y_processed[column]
        elif column in target_name_map:
            processed_column = target_name_map[column]
            series = y_processed[processed_column]
        else:
            raise ValueError(
                f"dtype_overrides['{column}']='auto' refers to an unknown column"
            )

        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(
                f"dtype_overrides['{column}']='auto' requires a numeric column"
            )

        resolution = _infer_quantized_resolution(series)
        resolved[processed_column] = (
            Domain.quantized_continuous(resolution)
            if resolution is not None
            else Domain.continuous()
        )

    return resolved


def _prepare_regression_training_data(
    X, y, dtype_overrides=None
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str], dict, Optional[dict]]:
    y_df = _coerce_target_dataframe(y)
    y_columns = y_df.columns.to_list()

    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    X_processed, y_processed = _preprocess(X_df, y_df)
    resolved_dtype_overrides = _resolve_regression_dtype_overrides(
        dtype_overrides, X_df, y_df, X_processed, y_processed
    )

    X_processed = _convert_int_to_float64(X_processed)
    X_pol = pl.DataFrame(X_processed)
    X_pol = _convert_int_to_float64(X_pol)
    X_pol, categorical_metadata = _convert_string_columns_to_categorical(
        X_pol, return_categories=True
    )
    y_pol = pl.DataFrame(y_processed).cast(pl.Float64)

    return X_pol, y_pol, y_columns, categorical_metadata, resolved_dtype_overrides


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
