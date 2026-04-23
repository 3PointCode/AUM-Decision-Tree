import numpy as np

# the bigger the gini value the more mixed classes are
def gini(y):
    if len(y) == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)

def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold

    X_left = X[left_mask]
    y_left = y[left_mask]
    X_right = X[right_mask]
    y_right = y[right_mask]

    return X_left, y_left, X_right, y_right

# the tree chooses the lowest weigted impurity after the split
def weighted_gini(y_left, y_right):
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0

    return (len(y_left) / n) * gini(y_left) + (len(y_right) / n) * gini(y_right)