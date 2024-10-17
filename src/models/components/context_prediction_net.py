from typing import List

import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return torch.reshape(x, self.shape)


class ContextPredictionNet(nn.Module):
    def __init__(self, num_classes: int, encoder: nn.Module, ignored_layers: List[str], fc: List[nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        for name in ignored_layers:
            setattr(self.encoder, name, nn.Identity())
        if fc is not None:
            self.fc = nn.Sequential(*fc)
        else:
            self.fc = nn.LazyLinear(num_classes)

    def forward(self, inputs: List[torch.Tensor]):
        outs = [self.encoder(inp) for inp in inputs]
        out = torch.cat(outs, dim=-1)
        return self.fc(out)
