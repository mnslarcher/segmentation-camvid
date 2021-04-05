from typing import Optional, Sequence, Tuple, Union

from sklearn.model_selection import train_test_split


def train_val_test_split(
    x: Sequence,
    val_size: Union[int, float] = 0.1,
    test_size: Union[int, float] = 0.1,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[Sequence, Sequence, Sequence]:
    if type(val_size) != type(test_size):
        raise ValueError(
            "val_size and test_size must have the same type, found "
            f"{type(val_size)} and {type(test_size)}."
        )

    x_train, x_test = train_test_split(
        x,
        test_size=val_size + test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    x_val, x_test = train_test_split(
        x_test,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        shuffle=shuffle,
    )
    return x_train, x_val, x_test
