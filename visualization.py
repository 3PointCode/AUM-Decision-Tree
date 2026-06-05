import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_comparison(results_df, save_path=None):
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = results_df["model"].values

    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, results_df[metric], width, label=metric)

    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Comparison of model performance metrics")
    plt.xticks(x + width * 1.5, model_names, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_learning_curve(train_sizes, train_scores, validation_scores, metric_name="F1-score", save_path=None):
    plt.figure(figsize=(8, 5))

    plt.plot(train_sizes, train_scores, marker="o", label="Training score")
    plt.plot(train_sizes, validation_scores, marker="o", label="Validation score")

    plt.xlabel("Training set size")
    plt.ylabel(metric_name)
    plt.title("Learning curve")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_hyperparameter_results(results_df, x_column, y_column="f1", save_path=None):
    plt.figure(figsize=(8, 5))

    plt.plot(results_df[x_column], results_df[y_column], marker="o")

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{y_column} depending on {x_column}")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_confusion_matrix(conf_matrix, title="Confusion matrix", save_path=None):
    plt.figure(figsize=(5, 4))

    plt.imshow(conf_matrix)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks([0, 1], ["Bad wine", "Good wine"])
    plt.yticks([0, 1], ["Bad wine", "Good wine"])

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()