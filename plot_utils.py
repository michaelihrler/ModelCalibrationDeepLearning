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
    class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Rotate class names for better visibility
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


def plot_multiple_calibration_curves(y_true, y_pred_proba_list, labels, title, n_bins=10, strategy='uniform'):
    """
    Plots multiple calibration curves on the same plot with a shared y_true.

    Parameters:
        y_true (array): The ground truth array (shared for all models).
        y_pred_proba_list (list): A list of predicted probability arrays (one for each model).
        labels (list): A list of labels for each model/curve.
        title (str): The title of the plot.
        n_bins (int): Number of bins to use in the calibration curve.
        strategy (str): Strategy for binning ('uniform' or 'quantile').
    """
    plt.figure(figsize=(10, 7))

    for y_pred_proba, label in zip(y_pred_proba_list, labels):
        y_pred_proba = np.array(y_pred_proba)

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_pred_proba,  # Use the positive class probabilities
            n_bins=n_bins,
            strategy=strategy
        )

        # Plot the curve
        plt.plot(prob_pred, prob_true, marker='o', label=label)

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


def plot_histogram_confidence(y_pred_of_class, class_name, title):
    # Define bin edges for fixed intervals [0, 0.05, 0.1, ..., 1.0]
    bins = np.arange(0, 1.05, 0.1)  # Step size of 0.05

    # Plot the histogram with specified bins
    plt.hist(y_pred_of_class, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlim(0, 1)
    plt.xlabel(f'Confidence of class {class_name}')
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.show()



y_pred_proba_1 = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]

y_true_2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred_proba_2 = [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.6, 0.4]
y_pred_proba_3 = [0.2, 0.8, 0.3, 0.7, 0.8, 0.6, 0.5, 0.5, 0.6, 0.4]

labels = ['Model 1', 'Model 2', "Modael"]

# Plot multiple curves
plot_multiple_calibration_curves(
    y_true=y_true_2,
    y_pred_proba_list=[y_pred_proba_1, y_pred_proba_2, y_pred_proba_3],
    labels=labels,
    title="Calibration Curve for Multiple Models"
)
