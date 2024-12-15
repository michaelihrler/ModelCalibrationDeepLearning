import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_curve,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             precision_score,
                             recall_score,
                             f1_score,
                             accuracy_score, auc, roc_auc_score, precision_recall_curve
                             )

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
    """
    Plots the confusion matrix with labels NORMAL and PNEUMONIA and computes F1-score, precision, recall, and accuracy.

    Parameters:
        true_labels (array-like): Ground truth (binary) target values (0 for NORMAL, 1 for PNEUMONIA).
        predicted_labels (array-like): Predicted labels (0 for NORMAL, 1 for PNEUMONIA).
        title (str): Title for the confusion matrix plot.

    Returns:
        f1 (float): F1-score.
        precision (float): Precision score.
        recall (float): Recall score.
        accuracy (float): Accuracy score.
    """
    # Define class labels
    class_labels = ["NORMAL", "PNEUMONIA"]

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    # Plot confusion matrix with proper labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

    # Calculate metrics
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    accuracy = accuracy_score(true_labels, predicted_labels)

    return f1, precision, recall, accuracy


def plot_multiple_calibration_curves(true_labels, y_pred_proba_list, labels, title, n_bins=10, strategy='uniform'):
    """
    Plots multiple calibration curves on the same plot with a shared y_true.

    Parameters:
        true_labels (array): The ground truth array (shared for all models).
        y_pred_proba_list (list): A list of predicted probability arrays (one for each model).
        labels (list): A list of labels for each model/curve.
        title (str): The title of the plot.
        n_bins (int): Number of bins to use in the calibration curve.
        strategy (str): Strategy for binning ('uniform' or 'quantile').
    """
    # Convert to array if list provided
    true_labels = np.array(true_labels)

    plt.figure(figsize=(10, 7))

    for y_pred_proba, label in zip(y_pred_proba_list, labels):
        y_pred_proba = np.array(y_pred_proba)

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            true_labels,
            y_pred_proba,
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


def plot_roc_curve(true_labels, predicted_probabilities, model_label):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for the model.

    Parameters:
        true_labels (array-like): True binary labels.
        predicted_probabilities (array-like): Predicted probabilities for the positive class.
        model_label (str): Label for the model in the legend
    """
    # Convert to array if list provided
    true_labels = np.array(true_labels)
    # Calculate False Positive Rate, True Positive Rate, and G-means
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    gmeans = np.sqrt(tpr * (1 - fpr))
    optimal_ix = np.argmax(gmeans)  # Optimal threshold index

    plt.figure(figsize=(15, 10), dpi=400)

    # Plotting no-skill line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, label='No Skill')

    # Plotting the ROC curve
    plt.plot(fpr, tpr, marker='.', markersize=12, markerfacecolor='green',
             linewidth=4, color='red', label=model_label)

    # Highlighting the optimal point
    plt.scatter(fpr[optimal_ix], tpr[optimal_ix], marker='X', s=300, color='blue', label='Optimal Threshold')

    # Adding grid, ticks, and styling
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=24)
    plt.legend(loc="lower right", prop={"size": 20})
    plt.show()

    auc_score = roc_auc_score(true_labels, predicted_probabilities)
    return auc_score


def plot_metrics_table(results):
    results = round_results(results)
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


def round_results(results, decimals=3):
    rounded_results = {}
    for key, values in results.items():
        if key == "Metric":
            rounded_results[key] = values
        else:
            rounded_results[key] = [round(float(value), decimals) for value in values]
    return rounded_results


def plot_probability_histogram(true_labels, predicted_probabilities, n_bins=10):
    """
    Plots a histogram of predicted probabilities for the two classes (y==0 and y==1).
    Calculates and returns Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Parameters:
        true_labels (array-like): Ground truth binary labels (0 or 1).
        predicted_probabilities (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the predicted probabilities. Default is 10.

    Returns:
        ece (float): Expected Calibration Error.
        mce (float): Maximum Calibration Error.
    """
    true_labels = np.array(true_labels)
    predicted_probabilities = np.array(predicted_probabilities)

    # Separate predicted probabilities by class
    probabilities_0 = predicted_probabilities[true_labels == 0]
    probabilities_1 = predicted_probabilities[true_labels == 1]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities_0, bins=n_bins, alpha=0.6, label='y==0', color='blue', range=(0, 1))
    plt.hist(probabilities_1, bins=n_bins, alpha=0.6, label='y==1', color='orange', range=(0, 1))
    plt.legend()
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Histogram of Predicted Probabilities by Class')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(true_labels, predicted_probabilities, n_bins=n_bins, strategy='uniform')

    # Calculate weights from calibration curve bins
    bin_counts, bin_edges = np.histogram(predicted_probabilities, bins=n_bins, range=(0, 1))
    total_count = np.sum(bin_counts)

    # Align weights with calibration bins
    bin_weights = bin_counts / total_count  # Normalize
    valid_bins = bin_weights > 0  # Match bins with non-zero counts in calibration_curve
    bin_weights = bin_weights[valid_bins]

    # Ensure alignment of weights and calibration bins
    if len(bin_weights) != len(prob_true):
        bin_weights = bin_weights[:len(prob_true)]  # Trim weights to match valid calibration bins

    # Calculate ECE and MCE
    bin_errors = np.abs(prob_true - prob_pred)
    ece = np.sum(bin_errors * bin_weights)  # Weighted average
    mce = np.max(bin_errors)

    return ece, mce


def plot_pr_curve(true_labels, predicted_probabilities, model_label='Model'):
    """
    Plots the Precision-Recall (PR) curve for the model and calculates the optimal threshold, F1-score, and AUC.

    Parameters:
        true_labels (array-like): Ground truth (binary) target values.
        predicted_probabilities (array-like): Predicted probabilities for the positive class.
        model_label (str): Label for the model in the legend. Default is 'Model'.

    Returns:
        best_threshold (float): Threshold corresponding to the best F1-score.
        best_f1 (float): Best F1-score across all thresholds.
        pr_auc (float): Area under the Precision-Recall curve.
    """
    # Convert to array if list provided
    true_labels = np.array(true_labels)
    # Calculate precision, recall, and thresholds
    precision, recall, thresholds_pr = precision_recall_curve(true_labels, predicted_probabilities)

    # Compute F1 scores
    f1_scores = (2 * precision * recall) / (precision + recall)

    # Handle potential NaN values due to division by zero
    f1_scores = np.nan_to_num(f1_scores)

    # Find the optimal threshold (max F1 score)
    optimal_ix = np.argmax(f1_scores)
    best_f1 = f1_scores[optimal_ix]

    # Note: `precision_recall_curve` produces one fewer threshold than precision/recall
    if optimal_ix < len(thresholds_pr):
        best_threshold = thresholds_pr[optimal_ix]
    else:
        best_threshold = 1.0  # Handle edge cases where no valid threshold exists

    # Compute area under the PR curve
    pr_auc = auc(recall, precision)

    # Plot the PR curve
    no_skill = len(true_labels[true_labels == 1]) / len(true_labels)  # No-skill line
    plt.figure(figsize=(15, 10), dpi=400)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', color='gray')
    plt.plot(recall, precision, marker='.', color='red', label=model_label)
    plt.scatter(recall[optimal_ix], precision[optimal_ix], marker='X', s=300, color='blue', label='Optimal Threshold')

    # Adding grid, ticks, and styling
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('Precision-Recall Curve', fontsize=24)
    plt.legend(loc="lower right", prop={"size": 20})
    plt.show()

    return best_threshold, best_f1, pr_auc
