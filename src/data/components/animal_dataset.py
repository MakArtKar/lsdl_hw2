import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AnimalUnlabeledDataset(Dataset):
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
        return self.transform(img)
