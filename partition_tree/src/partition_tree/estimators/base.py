from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from partition_tree import TARGET_COLUMN
import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils.validation import validate_data, check_is_fitted


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
