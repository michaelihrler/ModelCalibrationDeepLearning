import torch
import torch.nn as nn


class PlattScaling(nn.Module):
    def __init__(self, num_classes):
        super(PlattScaling, self).__init__()
        # One set of parameters (w, b) for each class
        self.w = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        """
        Apply Platt Scaling to multi-class logits.

        :param logits: Logits of shape [batch_size, num_classes].
        :return: Calibrated probabilities of shape [batch_size, num_classes].
        """
        return torch.sigmoid(self.w * logits + self.b)
