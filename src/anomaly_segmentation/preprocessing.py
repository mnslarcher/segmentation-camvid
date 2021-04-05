import os
from typing import List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import tqdm
from pycocotools.coco import COCO

from .io import read_rgb, write_rgb
from .model_selection import train_val_test_split


def crop_array(a: np.ndarray, bbox: Sequence[Union[int, float]]):
    x, y, w, h = np.round(bbox).astype(int)
    return a[y : y + h, x : x + w]


class DataPreparer:
    def __init__(
        self,
        images_dir: str,
        output_dir: str,
        bbox_coco_path: str,
        seg_coco_path: str,
        bbox_category: str,
        seg_categories: List[str],
        transforms: Optional[A.Compose] = None,
        val_size: Union[int, float] = 0.1,
        test_size: Union[int, float] = 0.1,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        ignore_missing: bool = False,
    ):
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.bbox_coco = COCO(bbox_coco_path)
        self.seg_coco = COCO(seg_coco_path)
        self.bbox_category_id = self.bbox_coco.getCatIds(catNms=bbox_category)[0]
        self.seg_category_ids = self.seg_coco.getCatIds(catNms=seg_categories)
        self.transforms = transforms
        self.ignore_missing = ignore_missing
        self.seg_img_file_name_to_id = {
            img["file_name"]: img_id for img_id, img in self.seg_coco.imgs.items()
        }
        self.multiclass = len(self.seg_category_ids) > 1
        self.cat_id_to_idx = {
            cat_id: idx
            for idx, cat_id in enumerate(sorted(self.seg_category_ids), self.multiclass)
        }
        self.image_ids = {}
        (
            self.image_ids["train"],
            self.image_ids["val"],
            self.image_ids["test"],
        ) = train_val_test_split(
            x=self.bbox_coco.catToImgs[self.bbox_category_id],
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    def _make_dirs(self):
        for f1 in ["train", "val", "test"]:
            for f2 in ["images", "masks"]:
                os.makedirs(os.path.join(self.output_dir, f1, f2))

    def _get_crops(
        self,
        image: np.ndarray,
        file_name: str,
        bbox: Sequence[Union[int, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        image = crop_array(image, bbox)
        mask = np.zeros((*image.shape[:2], len(self.cat_id_to_idx)), dtype="uint8")
        image_id = self.seg_img_file_name_to_id.get(file_name, None)
        if image_id is not None:
            for ann in self.seg_coco.imgToAnns[image_id]:
                if ann["category_id"] in self.cat_id_to_idx:
                    idx = self.cat_id_to_idx[ann["category_id"]]
                    mask_full = self.seg_coco.annToMask(ann)
                    mask_crop = crop_array(mask_full, bbox)
                    mask[..., idx] = np.max([mask[..., idx], mask_crop > 0], axis=0)

        if self.multiclass:
            mask = mask.argmax(2)
        else:
            mask = mask.squeeze()

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

    def _prepare_set(
        self,
        set_name: str,
    ) -> None:
        image_file_paths = set(os.listdir(self.images_dir))
        for image_id in tqdm.tqdm(self.image_ids[set_name]):
            file_name = self.bbox_coco.imgs[image_id]["file_name"]
            if file_name not in image_file_paths:
                if self.ignore_missing:
                    continue
                else:
                    raise ValueError(f"{file_name} not found.")

            image = read_rgb(os.path.join(self.images_dir, file_name))
            for ann in self.bbox_coco.imgToAnns[image_id]:
                if ann["category_id"] != self.bbox_category_id:
                    continue

                image_crop, mask_crop = self._get_crops(
                    image=image,
                    file_name=file_name,
                    bbox=ann["bbox"],
                )
                crop_file_name = (
                    "img={}-bbox=[{:.0f}, {:.0f}, {:.0f}, {:.0f}].png".format(
                        os.path.splitext(file_name)[0], *ann["bbox"]
                    )
                )

                write_rgb(
                    os.path.join(self.output_dir, set_name, "images", crop_file_name),
                    image_crop,
                )
                cv2.imwrite(
                    os.path.join(self.output_dir, set_name, "masks", crop_file_name),
                    mask_crop,
                )

    def prepare(self):
        self._make_dirs()
        self._prepare_set("train")
        self._prepare_set("val")
        self._prepare_set("test")
