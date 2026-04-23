import numpy as np
import pandas as pd
from decision_tree import split_dataset, weighted_gini, gini
from data_utils import load_data

def main():
    X, y, df = load_data("data/WineQT.csv")
    feature_index = 10
    threshold = 10.0
    X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)

    print("Gini przed podziałem:", gini(y))
    print("Gini po podziale:", weighted_gini(y_left, y_right))
    print("Lewa część:", len(y_left))
    print("Prawa część:", len(y_right))

    print("Probe number:", X.shape[0])
    print("Features number:", X.shape[1])
    print("Class distribution:", np.bincount(y))
    print(df.head())

if __name__ == "__main__":
    main()
