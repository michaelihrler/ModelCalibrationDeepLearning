import numpy as np


class HistogramBinning:
    """
    A class for implementing histogram binning for probability calibration.
    """

    def __init__(self, n_bins=10):
        """
        Initialize the histogram binning model.

        :param n_bins: Number of bins to use for calibration.
        """
        self.n_bins = n_bins
        self.bin_bounds = None
        self.bin_acc = None

    def fit(self, probabilities, labels):
        """
        Fit the histogram binning model using validation data.

        :param probabilities: Array of predicted probabilities (N, C).
        :param labels: Array of true labels (N,).
        """
        bins = np.linspace(0, 1, self.n_bins + 1)
        self.bin_bounds = bins
        bin_acc = []

        for i in range(len(bins) - 1):
            # Get indices of probabilities in the current bin
            in_bin = (probabilities >= bins[i]) & (probabilities < bins[i + 1])
            bin_total = np.sum(in_bin)

            if bin_total > 0:
                bin_correct = np.sum(labels[in_bin])
                bin_acc.append(bin_correct / bin_total)  # Empirical accuracy
            else:
                bin_acc.append(0)  # If no data falls in this bin, set accuracy to 0

        self.bin_acc = np.array(bin_acc)

    def forward(self, probabilities):
        """
        Calibrate probabilities using histogram binning.

        :param probabilities: Array of predicted probabilities (N, C).
        :return: Calibrated probabilities.
        """
        calibrated_probs = np.zeros_like(probabilities)
        for i in range(len(self.bin_bounds) - 1):
            in_bin = (probabilities >= self.bin_bounds[i]) & (probabilities < self.bin_bounds[i + 1])
            calibrated_probs[in_bin] = self.bin_acc[i]

        return calibrated_probs
