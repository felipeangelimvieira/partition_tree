from partition_tree.estimators.partition_tree import PartitionTreeRegressor
import pandas as pd
from sklearn.datasets import make_regression
import pytest


def test_cptree():

    X, y = make_regression(
        n_samples=2_000, noise=10, n_features=2, n_informative=1, random_state=20
    )

    y = pd.DataFrame(y, columns=["target"], dtype="float64")
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # Put y between 0.01 and 0.99
    y = (y - y.min()) / (y.max() - y.min())
    y = y * 0.98 + 0.01

    t = PartitionTreeRegressor(max_depth=10, min_samples_leaf=1)
    t.fit(X, y)

    t.predict(X)
