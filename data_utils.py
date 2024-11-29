import os

import numpy as np
from netcal.metrics import MCE, ECE


def get_class_names(train_dir):
    return [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]


def calculate_cece(y_true, prob_y, num_bins=20):
    y_true = np.array(y_true)
    prob_y = np.array(prob_y)
    num_classes = prob_y.shape[1]
    bins = np.linspace(0, 1, num_bins + 1)
    cece = 0

    for k in range(num_classes):  # For each class
        class_probs = prob_y[:, k]
        class_labels = (y_true == k).astype(int)
        class_cece = 0

        for i in range(num_bins):
            bin_mask = (class_probs >= bins[i]) & (class_probs < bins[i + 1])
            if bin_mask.sum() > 0:
                acc = np.mean(class_labels[bin_mask])
                conf = np.mean(class_probs[bin_mask])
                class_cece += abs(acc - conf) * bin_mask.sum()

        cece += class_cece / len(y_true)

    return cece / num_classes


def calculate_mce(y_true, prob_y, num_bins=20):
    y_true = np.array(y_true)
    prob_y = np.array(prob_y)
    mce_metric = MCE(bins=num_bins)
    return mce_metric.measure(prob_y, y_true)


def calculate_ece(y_true, prob_y, num_bins=20):
    y_true = np.array(y_true)
    prob_y = np.array(prob_y)
    ece_metric = ECE(bins=num_bins)
    return ece_metric.measure(prob_y, y_true)

def calculate_ace(y_true, prob_y, num_bins=20):
    y_true = np.array(y_true)
    prob_y = np.array(prob_y)
    ace_metric = ECE(bins=num_bins)
    return ace_metric.measure(prob_y, y_true)
