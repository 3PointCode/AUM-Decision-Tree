import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from decision_tree import DecisionTreeModel
from data_utils import load_data
from evaluation import evaluate_model, print_evaluation, evaluation_to_row
from baseline_models import get_baseline_models
from visualization import plot_metrics_comparison, plot_learning_curve, plot_confusion_matrix, plot_tuning_results

def main():
    os.makedirs("plots", exist_ok=True)

    X, y, df = load_data("data/WineQT.csv")

    # first split: final test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # second split: training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val)

    best_depth = None
    best_min_gain = None
    best_f1 = -1
    best_validation_results = None
    tuning_rows = []

    # tuning custom decision tree on validation set
    for min_gain in [0.0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
        for depth in range(1, 11):
            model = DecisionTreeModel(
                max_depth=depth, 
                min_samples_split=2,
                min_impurity_decrease=min_gain
            )

            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)

            results = evaluate_model(y_val, y_val_pred)

            tuning_rows.append({
                "max_depth": depth,
                "min_impurity_decrease": min_gain,
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
            })

            if results["f1"] > best_f1:
                best_f1 = results["f1"]
                best_min_gain = min_gain
                best_depth = depth
                best_validation_results = results

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv("results/tuning_results.csv", index=False)

    print("\nBest Custom Decision Tree on validation set")
    print("------------------------------------------")
    print("Best depth:", best_depth)
    print("Best min_impurity_decrease:", best_min_gain)
    print_evaluation("Validation result", best_validation_results)

    plot_tuning_results(tuning_df, save_path="plots/custom_tree_tuning_results.png")

    # learning curve for best custom tree
    train_sizes, train_scores, validation_scores = compute_learning_curve(
        X_train,
        y_train,
        X_val,
        y_val,
        best_depth,
        best_min_gain
    )

    plot_learning_curve(
        train_sizes,
        train_scores,
        validation_scores,
        metric_name="F1-score",
        save_path="plots/learning_curve_custom_tree.png"
    )

    all_results = []

    # final custom model: train on train + validation sets, evaluate on test set
    custom_model = DecisionTreeModel(
        max_depth=best_depth,
        min_samples_split=2,
        min_impurity_decrease=best_min_gain
    )

    custom_model.fit(X_train_val, y_train_val)
    custom_pred = custom_model.predict(X_test)
    
    custom_results = evaluate_model(y_test, custom_pred)
    print_evaluation("Custom Decision Tree - final results", custom_results)

    all_results.append(
        evaluation_to_row(
            "Custom Decision Tree",
            custom_results,
            {
                "max_depth": best_depth,
                "min_impurity_decrease": best_min_gain
            }
        )
    )

    plot_confusion_matrix(
        custom_results["confusion_matrix"],
        title="Custom Decision Tree - Confusion Matrix",
        save_path="plots/confusion_matrix_custom_tree.png"
    )

    # baseline models from scikit-learn
    baseline_models = get_baseline_models(best_depth)

    for name, model in baseline_models.items():
        model.fit(X_train_val, y_train_val)
        y_pred = model.predict(X_test)

        results = evaluate_model(y_test, y_pred)
        print_evaluation(name, results)

        all_results.append(
            evaluation_to_row(name, results)
        )

    results_df = pd.DataFrame(all_results)
    print("\nFinal comparison table")
    print("----------------------")
    print(results_df)

    results_df.to_csv("results/results.csv", index=False)
    print("\nSaved results to results/results.csv")

    plot_metrics_comparison(results_df, save_path="plots/model_metrics_comparison.png")

def compute_learning_curve(X_train, y_train, X_val, y_val, best_depth, best_min_gain):
    train_sizes = np.linspace(0.2, 1.0, 5)

    train_sizes_absolute = []
    train_scores = []
    validation_scores = []

    for size in train_sizes:
        subset_size = int(len(X_train) * size)

        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]

        model = DecisionTreeModel(
            max_depth=best_depth,
            min_samples_split=2,
            min_impurity_decrease=best_min_gain
        )

        model.fit(X_subset, y_subset)

        y_train_pred = model.predict(X_subset)
        y_val_pred = model.predict(X_val)

        train_f1 = f1_score(y_subset, y_train_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

        train_sizes_absolute.append(subset_size)
        train_scores.append(train_f1)
        validation_scores.append(val_f1)

    return train_sizes_absolute, train_scores, validation_scores

if __name__ == "__main__":
    main()
