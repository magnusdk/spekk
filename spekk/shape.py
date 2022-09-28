from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union


@dataclass
class Shape:
    "A list of dimensions, representing different axes of an array."
    dims: Tuple[str]

    def __add__(self, dim: Union[str, "Shape", Any]) -> "Shape":
        "Add the dimensions of another shape to this one, returning a new copy."
        if isinstance(dim, str):
            return Shape((*self.dims, dim))
        elif isinstance(dim, Shape):
            return Shape((*self.dims, *dim.dims))
        else:
            return NotImplemented

    def __radd__(self, dim: Union[str, "Shape", Any]) -> "Shape":
        "Add the dimensions of this shape to another one, returning a new copy."
        if isinstance(dim, str):
            return Shape((dim, *self.dims))
        elif isinstance(dim, Shape):
            return Shape((*dim.dims, *self.dims))
        else:
            return NotImplemented

    def __sub__(self, dim: Union[str, Sequence[str]]) -> "Shape":
        "Return a copy of this shape with dimension dim removed."
        pred_str = lambda d: d != dim
        pred_seq = lambda d: d not in dim
        pred = pred_str if isinstance(dim, str) else pred_seq
        return Shape(tuple([d for d in self.dims if pred(d)]))

    def index(self, dim: str, raises_exception: bool = False) -> Union[int, None]:
        """Return the index of the given dimension for this shape.

        If raises_exception is True then an exception is raised if the dimension is not
        present in this shape, otherwise it would return None."""
        if raises_exception:
            return self.dims.index(dim)
        else:
            return self.dims.index(dim) if dim in self.dims else None

    def replaced(
        self,
        replaced_dim: str,
        replacement_dim: Union[str, Sequence[str]],
    ) -> "Shape":
        """Replace the dimension with something else.

        >>> shape = Shape(("dim1", "dim2", "dim3"))
        >>> shape
        Shape(dims=('dim1', 'dim2', 'dim3'))
        >>> shape.replaced("dim2", ("foo", "bar"))
        Shape(dims=('dim1', 'foo', 'bar', 'dim3'))
        """
        if not replaced_dim in self.dims:
            raise ValueError(
                f"""May not replace axis "{replaced_dim}" because it doesn't exist in this shape with dimensions {self.dims}."""
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
        return Shape(tuple(dims))

    @property
    def ndim(self) -> int:
        "The number of dimensions of this shape."
        return len(self.dims)

    def __hash__(self) -> int:
        return hash(tuple(self.dims))
