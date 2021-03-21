import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ._datasets import SegmentationDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        category_ids,
        augments={"train": None, "val": None, "test": None},
        preprocess={"train": None, "val": None, "test": None},
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.category_ids = category_ids
        self.augments = augments
        self.preprocess = preprocess

    def setup(self, stage):
        self.train_dataset = SegmentationDataset(
            images_dir=os.path.join(self.data_dir, "train", "images"),
            masks_dir=os.path.join(self.data_dir, "train", "masks"),
            category_ids=self.category_ids,
            augments=self.augments["train"],
            preprocess=self.preprocess["train"],
        )
        self.val_dataset = SegmentationDataset(
            images_dir=os.path.join(self.data_dir, "val", "images"),
            masks_dir=os.path.join(self.data_dir, "val", "masks"),
            category_ids=self.category_ids,
            augments=self.augments["val"],
            preprocess=self.preprocess["val"],
        )
        self.test_dataset = SegmentationDataset(
            images_dir=os.path.join(self.data_dir, "test", "images"),
            masks_dir=os.path.join(self.data_dir, "test", "masks"),
            category_ids=self.category_ids,
            augments=self.augments["test"],
            preprocess=self.preprocess["test"],
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
