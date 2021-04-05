from typing import Type

import cv2
import numpy as np
import yaml


def read_rgb(file_path: str) -> np.ndarray:
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_rgb(file_path: str, image: np.ndarray) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image)


def read_yaml(file_path: str, Loader: Type = yaml.SafeLoader) -> dict:
    with open(file_path) as f:
        d = yaml.load(f, Loader=Loader)

    return d
