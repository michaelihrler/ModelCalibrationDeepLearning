import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

class BetaCalibration:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None

    def fit(self, probabilities, true_labels):
        """
        Train Beta Calibration on validation probabilities and true labels.
        :param probabilities: Predicted probabilities, shape [num_samples, num_classes].
        :param true_labels: Ground truth labels, shape [num_samples].
        """
        num_classes = probabilities.shape[1]
        self.models = []

        for i in range(num_classes):
            class_probabilities = probabilities[:, i]
            binary_labels = (true_labels == i).astype(int)

            # Optimize beta calibration parameters for this class
            def beta_loss(params):
                a, b, c = params
                calibrated = expit(a * logit(class_probabilities) + b * class_probabilities + c)
                return -np.sum(binary_labels * np.log(calibrated + 1e-12) +
                               (1 - binary_labels) * np.log(1 - calibrated + 1e-12))

            res = minimize(beta_loss, [1.0, 0.0, 0.0], method='L-BFGS-B')
            self.models.append(res.x)

    def forward(self, probabilities):
        """
        Apply the trained Beta Calibration model.
        :param probabilities: Predicted probabilities, shape [num_samples, num_classes].
        :return: Calibrated probabilities, shape [num_samples, num_classes].
        """
        calibrated_probabilities = np.zeros_like(probabilities)

        for i, params in enumerate(self.models):
            a, b, c = params
            calibrated_probabilities[:, i] = expit(a * logit(probabilities[:, i]) +
                                                   b * probabilities[:, i] + c)
        return calibrated_probabilities
