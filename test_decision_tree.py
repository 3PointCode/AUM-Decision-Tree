import numpy as np
from decision_tree import gini, split_dataset, weighted_gini

def test_gini_empty():
    y = np.array([])
    assert gini(y) == 0.0

def test_gini_pure_class_zero():
    y = np.array([0, 0, 0, 0])
    assert gini(y) == 0.0

def test_gini_pure_class_one():
    y = np.array([1, 1, 1, 1])
    assert gini(y) == 0.0

def test_gini_mixed_classes():
    y = np.array([0, 1, 0, 1])
    assert np.isclose(gini(y), 0.5)

def test_weighted_gini():
    y_left = np.array([0, 0])
    y_right = np.array([1, 1])
    assert weighted_gini(y_left, y_right) == 0.0

def test_split_dataset():
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ])
    y = np.array([0, 0, 1, 1])

    X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index=0, threshold=2.5)

    assert len(X_left) == 2
    assert len(X_right) == 2

    assert np.array_equal(y_left, np.array([0, 0]))
    assert np.array_equal(y_right, np.array([1, 1]))