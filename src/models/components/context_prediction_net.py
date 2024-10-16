from typing import List

import torch
import torch.nn as nn


class ContextPredictionNet(nn.Module):
    def __init__(self, num_classes: int, encoder: nn.Module, ignored_layers: List[str]):
        super().__init__()
        self.encoder = encoder
        for name in ignored_layers:
            setattr(self.encoder, name, nn.Identity())
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, inputs: list[torch.Tensor]):
        outs = [self.encoder(inp) for inp in inputs]
        out = torch.cat(outs, dim=-1)
        return self.fc(out)
