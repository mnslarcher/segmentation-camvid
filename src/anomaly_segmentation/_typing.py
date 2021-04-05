from typing import Optional

import albumentations as A
from typing_extensions import TypedDict


class TransformsDict(TypedDict):
    train: Optional[A.Compose]
    val: Optional[A.Compose]
    test: Optional[A.Compose]
