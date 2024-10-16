import glob

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


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
