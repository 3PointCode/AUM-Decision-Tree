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

# this function iterates through all features and thresholds to find the best split that minimizes the weighted gini impurity
def find_best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gini = float("inf")

    n_samples, n_features = X.shape

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])

        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)

            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            current_gini = weighted_gini(y_left, y_right)

            if current_gini < best_gini:
                best_gini = current_gini
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gini

# helper function to find the most common label in a set of labels
def most_common_label(y):
    if len(y) == 0:
        return None
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value