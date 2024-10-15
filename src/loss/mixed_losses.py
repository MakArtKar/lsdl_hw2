from typing import List

import torch.nn as nn


class MixedLosses(nn.Module):
    def __init__(self, losses: List, coefs: List[float]):
        super().__init__()
        self.losses = losses
        self.coefs = coefs

    def forward(self, inputs, targets):
        result = 0
        for loss, coef in zip(self.losses, self.coefs):
            result += coef * loss(inputs, targets)
        return result
