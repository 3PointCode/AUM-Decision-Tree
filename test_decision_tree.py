import numpy as np
import pytest
from data_utils import quality_to_binary_target
from decision_tree import gini, split_dataset, weighted_gini, find_best_split, most_common_label, build_tree, predict_one, predict, DecisionTreeModel

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

def test_find_best_split_returns_valid_values():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])
    y = np.array([0, 0, 1, 1])

    feature, threshold, best_gini = find_best_split(X, y)

    assert feature == 0
    assert threshold is not None
    assert best_gini >= 0.0

def test_most_common_label():
    y = np.array([0, 1, 1, 1, 0])
    assert most_common_label(y) == 1

def test_build_tree_returns_leaf_for_pure_data():
    X = np.array([
        [1.0],
        [2.0],
        [3.0]
    ])
    y = np.array([1, 1, 1])

    tree = build_tree(X, y, max_depth=3)

    assert tree.value == 1
    assert tree.left is None
    assert tree.right is None

def test_predict_one_on_simple_tree():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])
    y = np.array([0, 0, 1, 1])

    tree = build_tree(X, y, max_depth=1)
    pred = predict_one(np.array([1.5]), tree)

    assert pred in [0, 1]

def test_predict_returns_array_of_correct_length():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])
    y = np.array([0, 0, 1, 1])

    tree = build_tree(X, y, max_depth=2)
    preds = predict(X, tree)

    assert len(preds) == len(X)

def test_quality_to_binary_target_uses_wine_quality_threshold():
    quality = np.array([3, 4, 5, 6, 7, 8])

    y = quality_to_binary_target(quality)

    expected = np.array([0, 0, 0, 1, 1, 1])
    assert np.array_equal(y, expected)

def test_tree_learns_simple_threshold_pattern():
    X = np.array([
        [9.0],
        [9.5],
        [10.0],
        [11.0],
        [11.5],
        [12.0],
    ])

    quality = np.array([4, 5, 5, 6, 7, 8])
    y = quality_to_binary_target(quality)

    tree = build_tree(X, y, max_depth=2)
    predictions = predict(X, tree)

    assert np.array_equal(predictions, y)

def test_decision_tree_model_fit_and_predict():
    X = np.array([
        [9.0],
        [9.5],
        [10.0],
        [11.0],
        [11.5],
        [12.0],
    ])

    quality = np.array([4, 5, 5, 6, 7, 8])
    y = quality_to_binary_target(quality)

    model = DecisionTreeModel(max_depth=2)
    model.fit(X, y)
    predictions = model.predict(X)

    assert np.array_equal(predictions, y)

def test_decision_tree_model_predict_before_fit_raises_error():
    model = DecisionTreeModel(max_depth=2)
    X = np.array([[10.0], [11.0]])

    try:
        model.predict(X)
        assert False
    except ValueError:
        assert True