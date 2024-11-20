import os

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
    plt.bar(classes_sorted, counts_sorted, tick_label=[f"Class {cls}" for cls in classes_sorted])
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


def plot_confusion_matrix(true_labels, predicted_labels):
    unique_classes = np.unique(true_labels + predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def plot_multiclass_calibration_curve(y_true, y_pred_proba, class_labels=None, n_bins=20, strategy='uniform'):
    """
    Plots calibration curves for a multiclass classifier.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels for the samples.
    - y_pred_proba: array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class.
    - class_labels: list or array-like, default=None
        Class labels to display in the legend. If None, use integers [0, ..., n_classes-1].
    - n_bins: int, default=10
        Number of bins to discretize the probability space.
    - strategy: {'uniform', 'quantile'}, default='uniform'
        Strategy to define the width of the bins.

    Returns:
    - None (plots the calibration curve)
    """
    # Ensure y_true is a numpy array
    y_true = np.array(y_true)
    n_classes = y_pred_proba[0].shape[0]

    # One-hot encode y_true
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    # Set class labels if not provided
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(n_classes)]

    # Plot calibration curve for each class
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(
            y_true_binarized[:, i],
            y_pred_proba[:, i],
            n_bins=n_bins,
            strategy=strategy
        )
        plt.plot(prob_pred, prob_true, marker='o', label=class_labels[i])

    # Plot the perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect calibration')

    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve for Multiclass Classifier')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def plot_multiclass_roc_auc(y_true, y_pred_proba, class_labels=None):
    """
    Plots the ROC AUC curve for a multiclass classifier.

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
    n_classes = y_pred_proba[0].shape[0]

    # One-hot encode y_true
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
    plt.title('ROC AUC Curve for Multiclass Classifier')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
