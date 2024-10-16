from typing import Tuple, Union

import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F


class PuzzleExtract(nn.Module):
    def __init__(self, size: Union[Tuple[int, int], int], scale: float = 0.9):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale

    def forward(self, x):
        cropped_size = (int(self.size[0] / self.scale), int(self.size[1] / self.scale))
        crop = T.RandomCrop((cropped_size[0] * 3, cropped_size[1] * 3), pad_if_needed=True, padding_mode='reflect')(x)
        tiles = []
        for i in range(3):
            for j in range(3):
                tile = F.crop(crop, i * cropped_size[0], j * cropped_size[1], *cropped_size)
                tiles.append(T.RandomCrop(self.size)(tile))
        return tiles
