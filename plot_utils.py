import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

from data_utils import get_class_names


def plot_histogram_balance_of_dataset(data_dir, title):
    classes = get_class_names(data_dir)

    # Count the number of files in each class directory
    class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}

    # Sort classes alphabetically or by frequency for consistency
    classes_sorted = sorted(class_counts.keys())
    counts_sorted = [class_counts[cls] for cls in classes_sorted]

    # Plot the histogram
    plt.bar(classes_sorted, counts_sorted, tick_label=[f"{cls}" for cls in classes_sorted])
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Rotate class names for better visibility if many
    plt.show()


def plot_loss(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)  # Epoch numbers

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, title):
    unique_classes = np.unique(true_labels + predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


def plot_multiclass_calibration_curve(y_true, y_pred_proba, title, class_names=None, n_bins=20, strategy='uniform'):
    """
    Plots calibration curves for a multiclass or binary classifier with actual class names.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels for the samples.
    - y_pred_proba: array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class (after softmax).
    - title: str
        Title of the plot.
    - class_names: list or array-like, default=None
        Names of the classes to use in the legend. If None, uses "Class 0", "Class 1", etc.
    - n_bins: int, default=20
        Number of bins to discretize the probability space.
    - strategy: {'uniform', 'quantile'}, default='uniform'
        Strategy to define the width of the bins.

    Returns:
    - None (plots the calibration curve)
    """
    # Ensure y_true and y_pred_proba are numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    # Check if y_pred_proba sums to 1 (softmax probabilities)
    if not np.allclose(y_pred_proba.sum(axis=1), 1, atol=1e-2):
        raise ValueError("y_pred_proba should be softmax probabilities that sum to 1 across classes.")

    # Determine number of classes
    n_classes = y_pred_proba.shape[1]

    # One-hot encode y_true for multi-class or handle binary case
    if n_classes == 2:
        y_true_binarized = y_true  # Binary classification, no binarization needed
    else:
        from sklearn.preprocessing import label_binarize
        y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    # Set class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Plot calibration curve
    plt.figure(figsize=(10, 7))

    if n_classes == 2:
        # Binary classification
        prob_true, prob_pred = calibration_curve(
            y_true_binarized,
            y_pred_proba[:, 1],  # Use the positive class probabilities
            n_bins=n_bins,
            strategy=strategy
        )
        plt.plot(prob_pred, prob_true, marker='o', label=class_names[1])
    else:
        # Multi-class classification
        for i in range(n_classes):
            prob_true, prob_pred = calibration_curve(
                y_true_binarized[:, i],
                y_pred_proba[:, i],
                n_bins=n_bins,
                strategy=strategy
            )
            plt.plot(prob_pred, prob_true, marker='o', label=class_names[i])

    # Plot the perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect calibration')

    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.show()



def plot_multiclass_roc_auc(y_true, y_pred_proba, title, class_labels=None):
    """
    Plots the ROC AUC curve for a multiclass or binary classifier.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels for the samples.
    - y_pred_proba: array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class.
    - class_labels: list or array-like, default=None
        Class labels to display in the legend. If None, use integers [0, ..., n_classes-1].
    """
    # Ensure y_true is a numpy array
    y_true = np.array(y_true)

    # Determine the number of classes
    n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 2

    # Handle binary classification
    if n_classes == 2:
        # Binary classification: No need for one-hot encoding
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])  # Use probabilities for class 1
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f'Class 1 (AUC = {roc_auc:.2f})')
    else:
        # Multiclass classification: One-hot encode y_true
        y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

        # Set class labels if not provided
        if class_labels is None:
            class_labels = [f'Class {i}' for i in range(n_classes)]

        # Initialize plot
        plt.figure(figsize=(10, 7))

        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')

    # Plot the diagonal line (no skill)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def plot_metrics_table(results):
    df = pd.DataFrame(results)

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the size as needed

    # Hide axes
    ax.axis("tight")
    ax.axis("off")

    # Add a table at the center
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    # Customize table style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))  # Auto-adjust column widths

    # Set header cell colors
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_fontsize(12)
            cell.set_facecolor("lightgrey")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    plt.show()
