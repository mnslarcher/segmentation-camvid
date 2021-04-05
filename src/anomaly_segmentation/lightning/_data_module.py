import os
from typing import Optional, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .._typing import TransformsDict
from ..datasets import SegmentationDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        category_ids: Optional[Sequence] = None,
        num_workers: int = 1,
        pin_memory: bool = False,
        transforms: TransformsDict = {"train": None, "val": None, "test": None},
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.category_ids = category_ids
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transforms = transforms

    def prepare_data(self) -> None:  # type: ignore
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SegmentationDataset(
                images_dir=os.path.join(self.data_dir, "train", "images"),
                masks_dir=os.path.join(self.data_dir, "train", "masks"),
                category_ids=self.category_ids,
                transforms=self.transforms["train"],
            )
            self.val_dataset = SegmentationDataset(
                images_dir=os.path.join(self.data_dir, "val", "images"),
                masks_dir=os.path.join(self.data_dir, "val", "masks"),
                category_ids=self.category_ids,
                transforms=self.transforms["val"],
            )
        if stage == "test" or stage is None:
            self.test_dataset = SegmentationDataset(
                images_dir=os.path.join(self.data_dir, "test", "images"),
                masks_dir=os.path.join(self.data_dir, "test", "masks"),
                category_ids=self.category_ids,
                transforms=self.transforms["test"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
