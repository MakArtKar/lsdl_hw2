import os
import zipfile
from typing import Dict, Tuple, Optional, Any

import albumentations as A
import kaggle
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset

from .components.animal_dataset import AnimalDataset, AnimalUnlabeledDataset


class AnimalLabeledDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        train_dataset = None,
        val_dataset = None,
        test_dataset = None,
        train_transforms: Optional[A.BasicTransform] = None,
        val_transforms: Optional[A.BasicTransform] = None,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = os.path.join(data_dir, 'lsdl_hw2/data')

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_dir):
            print('Downloading data')
            kaggle.api.authenticate()
            path, name = os.path.split(os.path.split(self.data_dir)[0])
            print(path)
            kaggle.api.competition_download_files('lsdl-hw-2', path=path)
            with zipfile.ZipFile(os.path.join(path, 'lsdl-hw-2.zip'), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(path, name))

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.train_dataset is not None:
                self.data_train = self.hparams.train_dataset(
                    root=os.path.join(self.data_dir, self.hparams.train_dataset.keywords['root']),
                    transform=self.hparams.train_transforms
                )
                if self.hparams.val_dataset:
                    self.data_val = self.hparams.val_dataset(
                        root=os.path.join(self.data_dir, self.hparams.val_dataset.keywords['root']),
                        transform=self.hparams.val_transforms
                    )
                else:
                    self.data_val = self.hparams.train_dataset(
                        root=os.path.join(self.data_dir, self.hparams.train_dataset.keywords['root']),
                        transform=self.hparams.val_transforms
                    )
                    indices = list(range(len(self.data_train)))
                    np.random.shuffle(indices)
                    train_num = int(self.hparams.train_val_test_split[0] * len(indices))
                    self.data_train = Subset(self.data_train, indices=indices[:train_num])
                    self.data_val = Subset(self.data_val, indices=indices[train_num:])

            if self.hparams.test_dataset is not None:
                self.data_test = self.hparams.test_dataset(
                    root=os.path.join(self.data_dir, self.hparams.test_dataset.keywords['root']),
                    transform=self.hparams.val_transforms
                )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == '__main__':
    dm = AnimalLabeledDataModule()
    dm.prepare_data()
    dm.setup()
