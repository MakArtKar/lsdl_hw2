import os
import zipfile
from typing import Dict, Tuple, Optional, Any

import kaggle
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from .components.animal_dataset import AnimalUnlabeledDataset


class AnimalLabeledDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = os.path.join(data_dir, 'lsdl_hw2/data')

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
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
            trainset = ImageFolder(os.path.join(self.data_dir, 'train/labeled'), transform=self.transforms)
            self.data_test = AnimalUnlabeledDataset(os.path.join(self.data_dir, 'test'), transform=self.transforms)
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
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
