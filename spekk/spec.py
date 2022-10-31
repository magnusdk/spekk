from typing import Dict, List, Optional, Sequence, Set, Union, overload

from spekk.common import InvalidDimensionError, Specable, ValidationError
from spekk.shape import Shape

ShapeType = Union[Shape, Sequence[str]]


class Spec:
    """A collection of (optionally named) shapes representing for example the dimensions
    of a list of arguments.

    >>> import numpy as np
    >>> def my_func(a, b, c):
    ...     return np.sum(a, 1) + np.sum(c, 0) + b
    >>> spec = Spec([["foo", "bar"], [], ["bar"]])

    """

    @overload
    def __init__(self, shapes: Sequence[ShapeType]):
        ...

    @overload
    def __init__(self, shapes: Sequence[ShapeType], names: Optional[Sequence[str]]):
        ...

    @overload
    def __init__(self, shapes: Dict[str, ShapeType]):
        ...

    def __init__(
        self,
        shapes: Union[Dict[str, ShapeType], Sequence[ShapeType]],
        names: Optional[Sequence[str]] = None,
    ):
        if isinstance(shapes, dict):
            names = shapes.keys()
            shapes = shapes.values()
        shapes = [s if isinstance(s, Shape) else Shape(*s) for s in shapes]
        self.shapes = tuple(shapes)
        self.names = tuple(names) if names is not None else None

    def __sub__(self, dim: str) -> "Spec":
        """Return a new spec with the dimension removed from all of its shapes.

        >>> spec = Spec([["foo", "bar"], ["bar"]])
        >>> spec - "foo"
        Spec(
          ('bar',),
          ('bar',),
        )
        >>> spec - "bar"
        Spec(
          ('foo',),
          (),
        )
        """
        return Spec([shape - dim for shape in self.shapes])

    def indices_for(self, dim: str) -> List[Union[int, None]]:
        """Return the index of dimension dim for all shapes, None if the shape doesn't
        contain the dimension.

        >>> spec = Spec([["foo", "bar"], ["bar"]])
        >>> spec.indices_for("foo")
        [0, None]
        >>> spec.indices_for("bar")
        [1, 0]
        """
        if not self.has_dim(dim):
            raise InvalidDimensionError(
                f"Dimension '{dim}' does not exist in this spec:\n{repr(self)}"
            )
        return [shape.index(dim) for shape in self.shapes]

    @property
    def dims(self) -> Set[ShapeType]:
        """The dimensions contained in this spec, i.e. the union of the dimensions of
        all the shapes of this spec.

        >>> spec = Spec([["foo", "bar"], ["bar"]])
        >>> spec.dims == {"foo", "bar"}
        True
        >>> (spec - "bar").dims == {"foo"}
        True
        """
        all_dims = set()
        for shape in self.shapes:
            all_dims.update(shape.dims)
        return all_dims

    def has_dim(self, *dim: str) -> bool:
        """Return True if any of the shapes has the given dimension (or all of list of
        dimensions) dim.

        >>> spec = Spec([["foo", "bar"], ["bar"]])
        >>> spec.has_dim("foo")
        True
        >>> spec.has_dim("foo", "bar")
        True
        >>> spec.has_dim("foo", "baz")
        False
        """
        return all([d in self.dims for d in dim])

    def with_shape_at(
        self, shape_definitions: Dict[Union[int, str], ShapeType]
    ) -> "Spec":
        """Return a new copy of this spec with new shapes as specified by
        shape_definitions.

        >>> spec = Spec([["foo"], ["foo", "bar"]], ["arg1", "arg2"])
        >>> spec.with_shape_at({
        ...   "arg2": ["foo"],
        ...   "arg3": ["baz", "bar"]
        ... })
        Spec(
          arg1=('foo',),
          arg2=('foo',),
          arg3=('baz', 'bar'),
        )
        """
        if (
            any([isinstance(key, str) for key in shape_definitions.keys()])
            and self.names is None
        ):
            raise ValueError(
                "You may only update shapes by name (string) if the spec has named shapes."
            )

        new_spec = self
        new_shapes = [shape for shape in self.shapes]
        new_names = [name for name in self.names]
        for index_or_name, new_shape in shape_definitions.items():
            # Handle case where we update the shape by name (string)
            if isinstance(index_or_name, str):
                if index_or_name not in new_names:
                    new_names.append(index_or_name)
                try:
                    index = self.names.index(index_or_name)
                except:
                    index = len(self.shapes)
            else:
                index = index_or_name

            # Add the new shape to shapes
            new_shape = new_shape if isinstance(new_shape, Shape) else Shape(*new_shape)
            if index < len(self.shapes):
                new_shapes[index] = new_shape
            else:
                new_shapes.append(new_shape)

            new_spec = Spec(new_shapes, new_names)
        return new_spec

    def validate(self, data: Union[dict, Sequence]) -> None:
        """Raise a (hopefully) descriptive ValidationError if the data doesn't match
        this spec."""
        if len(data) != len(self.shapes):
            raise ValidationError(
                f"Mismatch between number of fields in data and number of specced fields (num fields in data: {len(data)}, num fields in spec: {len(self.shapes)})."
            )
        if isinstance(data, dict):
            if self.names is None:
                raise ValueError(
                    "May only validate against a dict when names have been defined for the spec."
                )
            data = [data[name] for name in self.names]
        for i, (item, shape) in enumerate(zip(data, self.shapes)):
            shape_name = f"'{self.names[i]}'" if self.names else f"at index {i}"
            if shape.ndim == 0:
                if isinstance(item, Specable) and not len(item.shape) == 0:
                    raise ValidationError(
                        f"The shape {shape_name} has no dimensions but the corresponding data item is an array with {len(item.shape)} dimensions."
                    )
            elif not isinstance(item, Specable):
                raise ValidationError(
                    f"The shape {shape_name} has dimensions {shape.dims} but the corresponding data item is not an array (it has no shape attribute). It is of type {type(item)}."
                )
            elif len(item.shape) != shape.ndim:
                raise ValidationError(
                    f"The shape {shape_name} has dimensions {shape.dims} but the corresponding data item has shape {item.shape}, which does not have the same number of dimensions."
                )

        # Check that the size of any dimension is the same across all arguments that
        # have that dimension.
        for dim in self.dims:
            shape_sizes_for_dim = [
                (shape_name, item.shape[index])
                for shape_name, item, index in zip(
                    self.names or range(len(self.shapes)),
                    data,
                    self.indices_for(dim),
                )
                if index is not None
            ]
            num_distinct_sizes = len(set([size for _, size in shape_sizes_for_dim]))
            assert num_distinct_sizes != 0
            if num_distinct_sizes != 1:
                size_overview_str = ", ".join(
                    [f"{name} has size {size}" for name, size in shape_sizes_for_dim]
                )
                raise ValidationError(
                    f"The size of a dimension must be the same for all arguments. The data has different sizes for dimension {dim}: {size_overview_str}."
                )

    def __getitem__(self, index: Union[str, int]) -> Shape:
        if isinstance(index, str):
            index = self.names.index(index)
        return self.shapes[index]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Spec):
            return False
        if tuple(self.shapes) != tuple(other.shapes):
            return False
        if self.names != other.names:
            return False
        return True

    def __repr__(self):
        if self.names:
            return (
                "Spec(\n  "
                + ",\n  ".join(
                    [
                        f"{name}={shape.dims}"
                        for name, shape in zip(self.names, self.shapes)
                    ]
                )
                + ",\n)"
            )
        else:
            return (
                "Spec(\n  "
                + ",\n  ".join([str(shape.dims) for shape in self.shapes])
                + ",\n)"
            )

    def __hash__(self) -> int:
        return hash(
            (tuple(self.shapes), tuple(self.names) if self.names is not None else ())
        )
