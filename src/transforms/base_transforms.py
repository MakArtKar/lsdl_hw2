import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


def check_if_torchvision_type(img):
    return isinstance(img, torch.Tensor) or isinstance(img, Image.Image)


def check_if_albumentations_type(img):
    return isinstance(img, np.ndarray)


def torchvision_to_albumentations(img):
    if isinstance(img, Image.Image):
        return np.array(img, dtype='uint8')
    return img


def albumentations_to_torchvision(img):
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return img


def print_info(x):
    if check_if_albumentations_type(x) or check_if_torchvision_type(x):
        x = [x]
    for temp in x:
        print(temp.shape, end=' ')
    print()


class SmartCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data):
        inputs = data['image']
        if check_if_albumentations_type(inputs):
            inputs = [inputs]
        for t in self.transforms:
            out = []
            for x in inputs:
                res = t(image=x)['image']
                if check_if_albumentations_type(res) or isinstance(res, torch.Tensor):
                    res = [res]
                out.extend(res)
            inputs = out
        if len(inputs) == 1:
            data['image'] = inputs[0]
        else:
            data['image'] = inputs
        return data


class AlbumentationWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, **data):
        data['image'] = self.transform(Image.fromarray(data['image']))
        if check_if_torchvision_type(data['image']):
            data['image'] = [data['image']]
        for i in range(len(data['image'])):
            data['image'][i] = torchvision_to_albumentations(data['image'][i])
        if len(data['image']) == 1:
            data['image'] = data['image'][0]
        return data
