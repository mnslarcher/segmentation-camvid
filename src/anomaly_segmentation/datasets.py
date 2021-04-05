import os
from typing import Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .io import read_rgb


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        category_ids: Optional[Sequence] = None,
        transforms: Optional[A.Compose] = None,
    ) -> None:
        self.file_names = sorted(os.listdir(images_dir))
        self.image_paths = [
            os.path.join(images_dir, file_name) for file_name in self.file_names
        ]
        self.mask_paths = [
            os.path.join(masks_dir, file_name) for file_name in self.file_names
        ]
        self.category_ids = category_ids
        self.transforms = transforms

        self._length = len(self.file_names)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        image = read_rgb(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.category_ids:
            masks = [(mask == category_id) for category_id in self.category_ids]
            mask = np.argmax([np.zeros_like(mask)] + masks, axis=0)

        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"].to(torch.int64)

        return image, mask

    def __len__(self) -> int:
        return self._length


class SegmentationInferenceDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        transforms: Optional[A.Compose] = None,
    ) -> None:
        self.file_names = sorted(os.listdir(images_dir))
        self.image_paths = [
            os.path.join(images_dir, file_name) for file_name in self.file_names
        ]
        self.transforms = transforms
        self._length = len(self.file_names)

    def __getitem__(self, idx: int) -> Union[np.ndarray, Tensor]:
        image = read_rgb(self.image_paths[idx])
        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image

    def __len__(self) -> int:
        return self._length
