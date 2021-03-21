import os

import albumentations as A
import cv2
from torch.utils.data import DataLoader

from segmentation.datasets import SegmentationDataset


def get_augments(size):
    transforms = A.Compose(
        [
            A.LongestMaxSize(size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT),
        ]
    )
    return transforms


def preprocess(image, mask):
    return {"image": image, "mask": (mask.sum(2) > 0) + mask.argmax(2).astype("uint8")}


def prepare_data(
    images_dir,
    masks_dir,
    images_output_dir,
    masks_output_dir,
    category_ids,
    image_size=1024,
    num_workers=16,
):
    os.makedirs(images_output_dir)
    os.makedirs(masks_output_dir)
    dataset = SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        category_ids=category_ids,
        augments=get_augments(image_size),
        preprocess=preprocess,
    )
    loader = DataLoader(dataset, shuffle=False, num_workers=num_workers)
    for file_name, (image, mask) in zip(dataset.file_names, loader):
        image = image.squeeze().numpy()
        mask = mask.squeeze().numpy()
        cv2.imwrite(os.path.join(images_output_dir, file_name), image)
        cv2.imwrite(os.path.join(masks_output_dir, file_name), mask)
