import glob
import itertools
import math

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.transforms import PuzzleExtract, AlbumentationWrapper


class AnimalDataset(ImageFolder):
    def __init__(self, *args, **kwargs):
        self.alb_transform = kwargs['transform']
        kwargs['transform'] = None
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if self.alb_transform is not None:
            image = self.alb_transform(image=np.array(image))['image']
        return {'image': image, 'label': label}


class AnimalUnlabeledDataset(Dataset):
    class_names = [
        'butterfly',
        'cat',
        'chicken',
        'cow',
        'dog',
        'elephant',
        'horse',
        'sheep',
        'spider',
        'squirrel',
    ]

    def __init__(self, root: str, transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.data = []
        for img_path in glob.glob(self.root + "/*.jpg"):
            self.data.append(img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.data[idx]))
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return {'image': img, 'image_path': self.data[idx]}


class AnimalUnlabeledRotationDataset(AnimalUnlabeledDataset):
    def __init__(self, *args, num_angles: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.angles = np.linspace(0, 360, num_angles, endpoint=False)

    def __getitem__(self, idx):
        angle = int(np.random.choice(range(len(self.angles))))
        img = np.array(Image.open(self.data[idx]))
        h, w = img.shape[:2]
        if self.transform is not None:
            img = self.transform(image=img)['image']

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = F.pad(img, [w // 2, h // 2, w // 2, h // 2], padding_mode="reflect")
        img = F.rotate(img, self.angles[angle])
        img = F.center_crop(img, (h, w))
        return {'image': img, 'label': angle}


class AnimalUnlabeledContextPredictionDataset(AnimalUnlabeledDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        label = np.random.randint(low=0, high=8)
        center_img = data['image'][4]
        target_img = data['image'][label + int(label >= 4)]
        data["image"] = (center_img, target_img)
        data["label"] = label
        return data


def lexicographic_index(p):
    """
    !!!took from https://stackoverflow.com/questions/12146910/finding-the-lexicographic-index-of-a-permutation-of-a-given-array

    Return the lexicographic index of the permutation `p` among all
    permutations of its elements. `p` must be a sequence and all elements
    of `p` must be distinct.

    >>> lexicographic_index('dacb')
    19
    >>> from itertools import permutations
    >>> all(lexicographic_index(p) == i
    ...     for i, p in enumerate(permutations('abcde')))
    True
    """
    result = 0
    for j in range(len(p)):
        k = sum(1 for i in p[j + 1:] if i < p[j])
        result += k * math.factorial(len(p) - j - 1)
    return result


class AnimalUnlabeledJigsawPuzzlesDataset(AnimalUnlabeledDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        perm = np.random.permutation(np.arange(9))
        data['label'] = lexicographic_index(perm)
        data['image'] = [data['image'][idx] for idx in perm]
        return data


class AnimalUnlabeledJigsawPuzzlesPositionDataset(AnimalUnlabeledDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        perm = np.random.permutation(np.arange(9))
        data['label'] = torch.tensor(perm)
        data['image'] = [data['image'][idx] for idx in perm]
        return data
