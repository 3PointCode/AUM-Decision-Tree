from sklearn.model_selection import train_test_split
from decision_tree import DecisionTreeModel
from data_utils import load_data
from evaluation import evaluate_model, print_evaluation

def main():
    X, y, df = load_data("data/WineQT.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_depth = None
    best_f1 = -1
    best_results = None

    for depth in range(1, 11):
        model = DecisionTreeModel(max_depth=depth, min_samples_split=2)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred)

        print_evaluation(f"Custom Decision Tree, depth={depth}", results)

        if results["f1"] > best_f1:
            best_f1 = results["f1"]
            best_depth = depth
            best_results = results

    print("\nBest Custom Decision Tree")
    print("Best depth:", best_depth)
    print_evaluation("Best Custom Decision Tree", best_results)

if __name__ == "__main__":
    main()
