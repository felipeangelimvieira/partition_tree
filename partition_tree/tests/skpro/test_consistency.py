from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import numpy as np


def test_multiple_runs():
    X, y = load_digits(return_X_y=True)
    X_iris, y_iris = load_iris(return_X_y=True)
    from partition_tree.sklearn import PartitionTreeClassifier

    pt1 = PartitionTreeClassifier(
        max_leaves=200,
        max_depth=100,
        min_samples_x=1,
        random_state=42,
        min_samples_split=1,
    )

    pt1.fit(X, y)

    pt2 = PartitionTreeClassifier(
        max_leaves=200,
        max_depth=100,
        min_samples_x=1,
        random_state=42,
        min_samples_split=1,
    )

    pt2.fit(X_iris, y_iris)

    # Again, digits

    pt3 = PartitionTreeClassifier(
        max_leaves=200,
        max_depth=100,
        min_samples_x=1,
        random_state=42,
        min_samples_split=1,
    )
    pt3.fit(X, y)

    y_pred_proba_pt1 = pt1.predict_proba(X)
    y_pred_proba_pt3 = pt3.predict_proba(X)

    assert np.allclose(y_pred_proba_pt1, y_pred_proba_pt3)
