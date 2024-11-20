import torch
import torch.nn as nn


class PlattScaling(nn.Module):
    def __init__(self):
        super(PlattScaling, self).__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return torch.sigmoid(self.w * logits + self.b)
