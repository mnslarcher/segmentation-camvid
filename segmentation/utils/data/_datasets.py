import os
from typing import Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        category_ids: Sequence,
        augments: Optional[A.Compose] = None,
        preprocess: Optional[A.Compose] = None,
    ) -> None:
        self.file_names = sorted(os.listdir(images_dir))
        self.image_paths = [
            os.path.join(images_dir, file_name) for file_name in self.file_names
        ]
        self.mask_paths = [
            os.path.join(masks_dir, file_name) for file_name in self.file_names
        ]
        self.category_ids = category_ids
        self.augments = augments
        self.preprocess = preprocess
        self._length = len(self.file_names)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.augments:
            sample = self.augments(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocess:
            sample = self.preprocess(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self) -> int:
        return self._length


class SegmentationInferenceDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        augments: Optional[A.Compose] = None,
        preprocess: Optional[A.Compose] = None,
    ) -> None:
        self.file_names = sorted(os.listdir(images_dir))
        self.image_paths = [
            os.path.join(images_dir, file_name) for file_name in self.file_names
        ]
        self.augments = augments
        self.preprocess = preprocess
        self._length = len(self.file_names)

    def __getitem__(self, idx: int) -> Union[np.ndarray, Tensor]:
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augments:
            sample = self.augments(image=image)
            image = sample["image"]

        if self.preprocess:
            sample = self.preprocess(image=image)
            image = sample["image"]

        return image

    def __len__(self) -> int:
        return self._length
