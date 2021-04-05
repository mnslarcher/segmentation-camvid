import pydoc
from typing import Any, Optional


def object_from_dict(
    d: dict, parent: Optional[Any] = None, **default_kwargs: Any
) -> Any:
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)  # type: ignore
