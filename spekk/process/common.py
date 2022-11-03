from dataclasses import dataclass
from typing import Iterable, Tuple, Union

from spekk.shape import Shape


@dataclass
class Axis:
    name: str
    keep: bool = False
    becomes: Tuple[str] = ()

    def __repr__(self) -> str:
        repr_str = f'Axis("{self.name}"'
        if self.keep:
            repr_str += ", keep=True"
        if self.becomes:
            repr_str += f", becomes={self.becomes}"
        repr_str += ")"
        return repr_str


def concretize_item(shape: Shape, item: Union[Axis, Iterable]):
    return (
        shape.index(item.name, False)
        if isinstance(item, Axis)
        else [concretize_item(shape, a) for a in item]
        if isinstance(item, (list, tuple))
        else {
            concretize_item(shape, k): concretize_item(shape, v)
            for k, v in item.items()
        }
        if isinstance(item, dict)
        else item
    )


def concretize_axes(shape: Shape, *args, **kwargs) -> Tuple[list, dict]:
    args = [concretize_item(shape, arg) for arg in args]
    kwargs = {k: concretize_item(shape, v) for k, v in kwargs.items()}
    return args, kwargs
