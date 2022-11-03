from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence, Union

from spekk import Shape, Spec, apply_across_dim
from spekk.backends.base import backend
from spekk.process.common import Axis, concretize_axes


class Transformation(ABC):
    @abstractmethod
    def new_spec(self, old_spec: Spec) -> Spec:
        ...

    @abstractmethod
    def new_shape(self, old_shape: Shape) -> Shape:
        ...


class OuterTransformation(Transformation, ABC):
    @abstractmethod
    def __call__(self, f: Callable, spec: Spec) -> Callable:
        ...


class InnerTransformation(Transformation, ABC):
    @abstractmethod
    def __call__(self, f: Callable, shape: Shape) -> Callable:
        ...


@dataclass
class ForAll(OuterTransformation):
    axis: str

    def __call__(self, f: Callable, spec: Spec) -> Callable:
        return backend.vmap(f, in_axes=spec.indices_for(self.axis))

    def new_spec(self, old_spec: Spec) -> Spec:
        return old_spec - self.axis

    def new_shape(self, old_shape: Shape) -> Shape:
        return self.axis + old_shape

    def __repr__(self) -> str:
        return f'ForAll("{self.axis}")'

    def __hash__(self) -> int:
        return hash(("ForAll", self.axis))


@dataclass
class AtIndex(OuterTransformation):
    axis: str
    index: Union[int, Sequence[int]]

    def __call__(self, f: Callable, spec: Spec) -> Callable:
        indices = backend.array(self.index)

        def wrapped_f(*args):
            # Replace arguments that have the given axis by selecting the element at a
            # given index for that axis.
            take_indices = lambda arg, dim_index: backend.take(arg, indices, dim_index)
            new_args = apply_across_dim(args, spec, self.axis, take_indices)
            if indices.ndim == 0:
                return f(*new_args)
            else:
                return backend.vmap(f, in_axes=spec.indices_for(self.axis))(*new_args)

        return wrapped_f

    def new_spec(self, old_spec: Spec) -> Spec:
        return old_spec - self.axis

    def new_shape(self, old_shape: Shape) -> Shape:
        if isinstance(self.index, int):
            return old_shape
        else:
            # If not a single index, we keep the dimension
            return self.axis + old_shape

    def __repr__(self) -> str:
        return f'AtIndex("{self.axis}", index={self.index})'

    def __hash__(self) -> int:
        return hash(("AtIndex", self.axis, self.index))


class Apply(InnerTransformation):
    def __init__(self, f, *args, **kwargs):
        self.f, self.args, self.kwargs = f, args, kwargs
        self.replaces = {}

    def __call__(self, fn_to_wrap, shape: Shape) -> Callable:
        def inner(*inner_args) -> Callable:
            result = fn_to_wrap(*inner_args)
            args, kwargs = concretize_axes(shape, *self.args, **self.kwargs)
            return self.f(result, *args, **kwargs)

        return inner

    def replacing(self, axis: str, new_axis: Union[str, Sequence[str]]) -> "Apply":
        new_obj = Apply(self.f, *self.args, **self.kwargs)
        new_obj.replaces[axis] = new_axis
        return new_obj

    def new_spec(self, old_spec: Spec) -> Spec:
        return old_spec  # noop

    def new_shape(self, old_shape: Shape) -> Shape:
        shape = old_shape
        for arg in [*self.args, *self.kwargs.values()]:
            if isinstance(arg, Axis):
                if arg.becomes:
                    shape = shape.replaced(arg.name, arg.becomes)
                elif not arg.keep:
                    shape -= arg.name
        for replaced_axis, replacement_axis in self.replaces.items():
            shape = shape.replaced(replaced_axis, replacement_axis)

        return shape

    def __repr__(self) -> str:
        args_str = ", ".join([str(arg) for arg in self.args])
        kwargs_str = ", ".join([f"{k}={str(v)}" for k, v in self.kwargs.items()])
        repr_str = f"Apply({str(self.f.__qualname__)}"
        if self.args:
            repr_str += f", {args_str}"
        if self.kwargs:
            repr_str += f", {kwargs_str}"
        return repr_str + ")"

    def __hash__(self) -> int:
        # Not always safe, but should work in all but the edgiest of cases.
        return hash(("Apply", self.f, repr(self.args), repr(self.kwargs)))


class Transpose(Apply):
    def __init__(self, *axes: Sequence[Union[int, str, Axis]]):
        self.axes = axes
        super().__init__(
            backend.transpose,
            [Axis(axis) if isinstance(axis, str) else axis for axis in axes],
        )

    def new_shape(self, old_shape: Shape) -> Shape:
        return Shape(
            [
                old_shape.dims[axis]
                if isinstance(axis, int)
                else axis.name
                if isinstance(axis, Axis)
                else axis
                for axis in self.axes
            ]
        )

    def __repr__(self) -> str:
        return f"Transpose({', '.join(self.axes)})"

    def __hash__(self) -> int:
        return hash(tuple(self.axes))
