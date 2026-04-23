from decision_tree import find_best_split
from data_utils import load_data

def main():
    X, y, df = load_data("data/WineQT.csv")

    best_feature, best_threshold, best_gini = find_best_split(X, y)

    print("Best feature:", best_feature)
    print("Best threshold:", best_threshold)
    print("Best gini:", best_gini)

if __name__ == "__main__":
    main()
