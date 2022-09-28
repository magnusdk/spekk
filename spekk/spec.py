from typing import Dict, List, Optional, Sequence, Set, Union, overload

from spekk.common import Specable, ValidationError
from spekk.shape import Shape

ShapeType = Union[Shape, Sequence[str]]
Name = str


class Spec:
    """A collection of (optionally named) shapes representing for example the dimensions
    of a list of arguments."""

    @overload
    def __init__(self, shapes: List[ShapeType]):
        ...

    @overload
    def __init__(self, shapes: List[ShapeType], names: Optional[List[Name]]):
        ...

    @overload
    def __init__(self, shapes: Dict[Name, ShapeType]):
        ...

    def __init__(
        self,
        shapes: Union[Dict[Name, ShapeType], List[ShapeType]],
        names: Optional[List[Name]] = None,
    ):
        if isinstance(shapes, dict):
            assert names is None
            names = [name for name in shapes.keys()]
            shapes = [
                shape if isinstance(shape, Shape) else Shape(shape)
                for shape in shapes.values()
            ]
        else:
            assert shapes is not None
            shapes = [
                shape if isinstance(shape, Shape) else Shape(shape) for shape in shapes
            ]
        self.shapes = shapes
        self.names = names

    def __sub__(self, dim: ShapeType) -> "Spec":
        "Return a copy of this spec with the dimension removed from all its shapes."
        return Spec([shape - dim for shape in self.shapes])

    def indices_for(
        self, dim: str, index_by_name: bool = False
    ) -> List[Union[int, None]]:
        """Return the indices of dimension dim for all shapes, None if the shape doesn't
        contain the dimension."""
        if index_by_name:
            return {
                name: shape.index(dim)
                for (name, shape) in zip(self.names, self.shapes)
                if dim in shape.dims
            }
        else:
            return [shape.index(dim) for shape in self.shapes]

    @property
    def dims(self) -> Set[ShapeType]:
        """The dimensions contained in this spec. The union of the dimensions of all the
        shapes of this spec."""
        all_dims = set()
        for shape in self.shapes:
            all_dims.update(shape.dims)
        return all_dims

    def has_dim(self, dim: Union[str, Sequence[str]]) -> bool:
        """Return True if any of the shapes has the given dimension (or all of list of
        dimensions) dim."""
        if isinstance(dim, str):
            dim = [dim]
        return all([d in self.dims for d in dim])

    def with_shape_at(self, **kwargs: Dict[Union[int, Name], ShapeType]) -> "Spec":
        """Return a new copy of this spec with new shapes as specified by kwargs.

        kwargs must be a mapping from the index or name of a shape to the new shape.
        Example:
        >>> spec = Spec([["foo"], ["foo", "bar"]], ["arg1", "arg2"])
        >>> spec
        Spec(
          arg1=['foo'],
          arg2=['foo', 'bar']
        )
        >>> spec.with_shape_at(arg2=["foo"], arg3=["baz", "bar"])
        Spec(
          arg1=['foo'],
          arg2=['foo'],
          arg3=['baz', 'bar']
        )
        """
        new_spec = self
        new_shapes = [shape for shape in self.shapes]
        new_names = [name for name in self.names]
        for index_or_name, new_shape in kwargs.items():
            if isinstance(index_or_name, str):
                try:
                    index = self.names.index(index_or_name)
                except:
                    index = len(self.shapes)
            else:
                index = index_or_name

            new_shape = new_shape if isinstance(new_shape, Shape) else Shape(new_shape)
            if index < len(self.shapes):
                new_shapes[index] = new_shape
            else:
                new_shapes.append(new_shape)
                if isinstance(index_or_name, str):
                    assert len(self.names) == len(
                        self.shapes
                    ), "Spec must have names if new shape is added by name."
                    new_names.append(index_or_name)

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

    def __getitem__(self, name_or_index: Union[str, int]) -> Shape:
        if isinstance(name_or_index, int):
            return self.shapes[name_or_index]
        else:
            return self.shapes[self.names.index(name_or_index)]

    def __eq__(self, other):
        if isinstance(other, Spec):
            return self.shapes == other.shapes
        else:
            return False

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
                + "\n)"
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
