import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibration:
    def __init__(self):
        self.models = None  # Separate Isotonic Regression model for each class

    def fit(self, probabilities, true_labels):
        """
        Train Isotonic Regression for multi-class calibration.
        :param probabilities: Predicted probabilities of shape [num_samples, num_classes].
        :param true_labels: Ground truth labels of shape [num_samples].
        """
        num_classes = probabilities.shape[1]
        self.models = []

        for i in range(num_classes):
            # Extract probabilities for the current class
            class_probabilities = probabilities[:, i]
            # Create binary labels for the current class
            binary_labels = (true_labels == i).astype(int)

            # Fit isotonic regression for the current class
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(class_probabilities, binary_labels)
            self.models.append(ir)

    def forward(self, probabilities):
        """
        Apply the trained Isotonic Regression models to the probabilities.
        :param probabilities: Predicted probabilities of shape [num_samples, num_classes].
        :return: Calibrated probabilities of shape [num_samples, num_classes].
        """
        calibrated_probabilities = np.zeros_like(probabilities)
        for i, ir in enumerate(self.models):
            calibrated_probabilities[:, i] = ir.predict(probabilities[:, i])
        return calibrated_probabilities
