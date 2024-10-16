import albumentations as A
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class SmartCompose(T.Compose):
    def __call__(self, x):
        x = [x]
        for t in self.transforms:
            out = []
            for inp in x:
                res = t(inp)
                if isinstance(res, torch.Tensor) or isinstance(res, Image.Image):
                    res = [res]
                out.extend(res)
            x = out
        return x


class AlbumentationWrapper(A.BasicTransform):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, data):
        data['image'] = self.transform(Image.fromarray(data['image']))
        if isinstance(data['image'], Image.Image):
            data['image'] = np.array(data['image'])
        return data
