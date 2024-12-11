import torch
import torch.nn as nn
from torch.optim import LBFGS

class TemperatureScaling(nn.Module):
    def __init__(self):
        """
        Initialize the Temperature Scaling model.
        The temperature parameter is learned during fitting.
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize temperature > 1

    def forward(self, logits):
        """
        Apply temperature scaling to logits.
        Args:
            logits (torch.Tensor): Logits from the model of shape (n_samples, n_classes).
        Returns:
            torch.Tensor: Calibrated logits after temperature scaling.
        """
        return logits / self.temperature

    def fit(self, logits, labels):
        """
        Fit the temperature scaling model by finding the optimal temperature.
        Args:
            logits (torch.Tensor): Logits from the model of shape (n_samples, n_classes).
            labels (torch.Tensor): Ground truth labels of shape (n_samples,).
        """
        self.train()
        optimizer = LBFGS([self.temperature], lr=0.01)

        def closure():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()

    def predict(self, logits):
        """
        Apply temperature scaling and return calibrated probabilities.
        Args:
            logits (torch.Tensor): Logits from the model of shape (n_samples, n_classes).
        Returns:
            torch.Tensor: Calibrated probabilities of shape (n_samples, n_classes).
        """
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            probabilities = torch.softmax(scaled_logits, dim=1)
            return probabilities
