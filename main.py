from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from decision_tree import build_tree, predict
from data_utils import load_data

def main():
    X, y, df = load_data("data/WineQT.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # best result for depth=5 - Accuracy: 0.751 Precision: 0.768 Recall: 0.774 F1: 0.771
    for depth in range(1, 11):
        tree = build_tree(X_train, y_train, max_depth=depth, min_samples_split=2)
        y_pred = predict(X_test, tree)

        print(
            "Depth:", depth,
            "Accuracy:", accuracy_score(y_test, y_pred),
            "Precision:", precision_score(y_test, y_pred),
            "Recall:", recall_score(y_test, y_pred),
            "F1:", f1_score(y_test, y_pred)
        )

if __name__ == "__main__":
    main()
