from dataclasses import dataclass
from functools import reduce
from typing import Sequence, Tuple

from spekk.spec import Spec
from spekk.trees import Tree, traverse


@dataclass
class Axis:
    """A placeholder for an array axis, given by the name of that axis (dimension).

    In the context of a transformation, an Axis is a way to get the concrete axis-index
    of an array, and also to specify what happens to that dimension in the
    transformation.

    By default, the dimension is removed, which makes sense for common operations like
    numpy.sum or numpy.mean. If you want to keep the dimension, set keep=True. If you
    want to replace the dimension with something else, set becomes=(something, else).
    """

    dimension: str
    keep: bool = False
    becomes: Tuple[str] = ()

    def new_dimensions(self, dimensions: Sequence[str]) -> Tuple[str]:
        """Given a sequence of dimensions return the new dimensions after this Axis has
        been parsed.

        By default, the dimension is removed. If keep=True, the dimension is kept. If
        becomes is set, the dimension is replaced with the dimensions defined by the
        becomes field.

        Examples:
        >>> old_dimensions = ("a", "b", "c")
        >>> Axis("b").new_dimensions(old_dimensions)
        ('a', 'c')
        >>> Axis("b", keep=True).new_dimensions(old_dimensions)
        ('a', 'b', 'c')
        >>> Axis("b", becomes=("x", "y")).new_dimensions(old_dimensions)
        ('a', 'x', 'y', 'c')
        """
        if self.becomes:
            return reduce(
                lambda a, b: (
                    a + tuple(self.becomes) if b == self.dimension else a + (b,)
                ),
                dimensions,
                (),
            )
        elif self.keep:
            return tuple(dimensions)
        else:
            return tuple(d for d in dimensions if d != self.dimension)

    def __repr__(self) -> str:
        repr_str = f'Axis("{self.dimension}"'
        if self.keep:
            repr_str += ", keep=True"
        if self.becomes:
            repr_str += f", becomes={self.becomes}"
        repr_str += ")"
        return repr_str


class AxisConcretizationError(ValueError):
    def __init__(self, axis: Axis):
        super().__init__(f'Could not find dimension "{axis.dimension}" in the spec.')


def concretize_axes(spec: Spec, args: Tree, kwargs: Tree) -> Tuple[list, dict]:
    def traverse_fn(x: Tree, path: tuple):
        if isinstance(x, Axis):
            # Path is not for the spec
            index = spec.index_for(x.dimension)
            if index is None:
                raise AxisConcretizationError(x)
            return index
        else:
            return x

    args, kwargs = traverse(
        (args, kwargs), lambda x: isinstance(x, Axis), traverse_fn, use_path=True
    )
    return args, kwargs


if __name__ == "__main__":
    import doctest

    doctest.testmod()
