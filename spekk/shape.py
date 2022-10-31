from typing import Sequence, Union

from spekk.common import InvalidDimensionError


class Shape:
    """A collection of dimensions, representing different named axes of an array.

    >>> import numpy as np
    >>> arr = np.random.random((3,4,5))
    >>> arr.shape
    (3, 4, 5)
    >>> shape = Shape("a", "b", "c")
    >>> shape.index("b")
    1
    >>> summed_arr = np.sum(arr, axis=shape.index("b"))
    >>> summed_arr.shape
    (3, 5)
    """

    def __init__(self, *dims: str):
        if not all([isinstance(dim, str) for dim in dims]):
            raise ValueError(
                f"Dimensions must be strings, but got called with {dims} as arguments."
            )
        if len(dims) != len(set(dims)):
            raise ValueError(
                f"Dimensions may not contain duplicates, but got called with {dims} as arguments."
            )
        self.dims = dims

    def __add__(self, dim: Union[str, "Shape"]) -> "Shape":
        """Return a new shape with dim appended to this shape's dimensions.

        >>> shape = Shape("a", "b")
        >>> shape + "c"
        Shape('a', 'b', 'c')
        >>> shape + Shape("c", "d")
        Shape('a', 'b', 'c', 'd')
        """
        if isinstance(dim, str):
            return Shape(*self.dims, dim)
        elif isinstance(dim, Shape):
            return Shape(*self.dims, *dim.dims)
        else:
            return NotImplemented

    def __radd__(self, dim: Union[str, "Shape"]) -> "Shape":
        """Return a new shape with dim prepended to this shape's dimensions.

        >>> shape = Shape("a", "b")
        >>> "c" + shape
        Shape('c', 'a', 'b')
        """
        if isinstance(dim, str):
            return Shape(dim, *self.dims)
        elif isinstance(dim, Shape):
            return Shape(*dim.dims, *self.dims)
        else:
            return NotImplemented

    def __sub__(self, dim: Union[str, Sequence[str]]) -> "Shape":
        """Return a copy of this shape with dimension dim removed.

        >>> shape = Shape("a", "b", "c")
        >>> shape - "b"
        Shape('a', 'c')
        >>> shape - ["a", "b"]
        Shape('c',)
        """
        pred_str = lambda d: d != dim
        pred_seq = lambda d: d not in dim
        pred = pred_str if isinstance(dim, str) else pred_seq
        return Shape(*[d for d in self.dims if pred(d)])

    def index(self, dim: str, raises_exception: bool = False) -> Union[int, None]:
        """Return the index of the given dimension for this shape.

        If raises_exception is True then an exception is raised if the dimension is not
        present in this shape, otherwise it would return None.

        >>> shape = Shape("a", "b")
        >>> shape.index("a")
        0
        >>> shape.index("b")
        1
        >>> shape.index("c") is None
        True
        >>> shape.index("c", raises_exception=True)
        Traceback (most recent call last):
        spekk.common.InvalidDimensionError: This shape with dimensions ('a', 'b') does not contain the dimension 'c'.
        """
        if raises_exception:
            if dim not in self.dims:
                raise InvalidDimensionError(
                    f"This shape with dimensions {self.dims} does not contain the dimension '{dim}'."
                )
            return self.dims.index(dim)
        else:
            return self.dims.index(dim) if dim in self.dims else None

    def replaced(
        self,
        replaced_dim: str,
        replacement_dim: Union[str, Sequence[str]],
    ) -> "Shape":
        """Replace the dimension with something else.

        >>> shape = Shape("dim1", "dim2", "dim3")
        >>> shape.replaced("dim2", "foo")
        Shape('dim1', 'foo', 'dim3')
        >>> shape.replaced("dim2", ("foo", "bar"))
        Shape('dim1', 'foo', 'bar', 'dim3')
        """
        if not replaced_dim in self.dims:
            raise InvalidDimensionError(
                f"This shape with dimensions {self.dims} does not contain the dimension '{replaced_dim}'."
            )
        dims = []
        for d in self.dims:
            if d == replaced_dim:
                if isinstance(replacement_dim, str):
                    dims.append(replacement_dim)
                else:
                    dims += replacement_dim  # If d is not a str, it is a sequence (presumably)
            else:
                dims.append(d)
        return Shape(*dims)

    @property
    def ndim(self) -> int:
        """The number of dimensions of this shape.

        >>> shape = Shape("dim1", "dim2", "dim3")
        >>> shape.ndim
        3
        """
        return len(self.dims)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Shape):
            return False
        if tuple(self.dims) != tuple(other.dims):
            return False
        return True

    def __iter__(self):
        return iter(self.dims)

    def __repr__(self) -> str:
        return f"Shape{tuple(self.dims)}"

    def __hash__(self) -> int:
        return hash(tuple(self.dims))
