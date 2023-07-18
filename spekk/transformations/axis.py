"""An :class:`Axis` references an axis of an array by its corresponding dimension name 
in a spec."""

from dataclasses import dataclass
from functools import reduce
from typing import Sequence, Tuple

import spekk.trees as trees
from spekk.spec import Spec
from spekk.trees import Tree, has_treedef, leaves


@dataclass
class Axis:
    """A placeholder for an array axis, given by the name of that axis (dimension).

    In the context of a transformation, an :class:`Axis` is a way to get the concrete 
    axis-index of an array, and also to specify what happens to that dimension in the
    transformation.

    By default, the dimension is removed, which makes sense for common operations like
    :func:`numpy.sum` or :func:`numpy.mean`. If you want to keep the dimension, set 
    ``keep=True``. If you want to replace the dimension with something else, set 
    ``becomes=("something", "else")``.
    """

    dimension: str  #: The referenced dimension
    keep: bool = False  #: Whether to keep the dimension in the spec after referencing it (default ``False``, i.e. the dimension is removed from the spec).
    becomes: Tuple[str] = ()  #: If set, the dimension is replaced in the spec with the given dimensions.

    def new_dimensions(self, dimensions: Sequence[str]) -> Tuple[str]:
        """Given a sequence of dimensions return the new dimensions after this 
        :class:`Axis` has been parsed.

        By default, the dimension is removed. If ``keep=True``, the dimension is kept. 
        If ``becomes`` is set, the dimension is replaced with the dimensions defined by 
        the ``becomes`` field.

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
    """Convert any instance of :class:`Axis` in ``args`` and ``kwargs`` to the concrete 
    axis index, as defined by the spec.

    >>> spec = Spec(["a", "b"])
    >>> args = (Axis("a"), Axis("b"))
    >>> kwargs = {"baz": Axis("b")}
    >>> concretize_axes(spec, args, kwargs)
    ((0, 1), {'baz': 1})
    """
    state = (args, kwargs)
    for leaf in leaves(state, lambda x: isinstance(x, Axis) or not has_treedef(x)):
        if isinstance(leaf.value, Axis):
            index = spec.index_for(leaf.value.dimension)
            if index is None:
                raise AxisConcretizationError(leaf.value)
            state = trees.set(state, index, leaf.path)
    return state


if __name__ == "__main__":
    import doctest

    doctest.testmod()
