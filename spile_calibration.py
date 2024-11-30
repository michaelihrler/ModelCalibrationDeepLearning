from scipy.interpolate import UnivariateSpline
import numpy as np

class SplineCalibration:
    def __init__(self):
        self.models = None  # Store one spline model for each class

    def fit(self, probabilities, true_labels, smoothing_factor=1.0):
        """
        Train Spline Calibration on validation probabilities and true labels.
        :param probabilities: Predicted probabilities, shape [num_samples, num_classes].
        :param true_labels: Ground truth labels, shape [num_samples].
        :param smoothing_factor: Controls the smoothness of the spline.
        """
        num_classes = probabilities.shape[1]
        self.models = []

        for i in range(num_classes):
            class_probabilities = probabilities[:, i]
            binary_labels = (true_labels == i).astype(int)

            # Sort probabilities and corresponding labels
            sorted_indices = np.argsort(class_probabilities)
            class_probabilities = class_probabilities[sorted_indices]
            binary_labels = binary_labels[sorted_indices]

            # Fit a smoothing spline to the calibration curve
            spline = UnivariateSpline(class_probabilities, binary_labels, s=smoothing_factor)
            self.models.append(spline)

    def forward(self, probabilities):
        """
        Apply the trained Spline Calibration model.
        :param probabilities: Predicted probabilities, shape [num_samples, num_classes].
        :return: Calibrated probabilities, shape [num_samples, num_classes].
        """
        calibrated_probabilities = np.zeros_like(probabilities)

        for i, spline in enumerate(self.models):
            calibrated_probabilities[:, i] = spline(probabilities[:, i])
        return np.clip(calibrated_probabilities, 0, 1)  # Ensure probabilities are valid
