import inspect
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple, Union

from spekk import Shape, Spec
from spekk.process.kernels import Kernel
from spekk.process.transformations import InnerTransformation, OuterTransformation

Transformation = Union[Kernel, InnerTransformation, OuterTransformation]


def _build_recursively(
    spec: Spec,
    transformations: Sequence[Transformation],
) -> Tuple[Callable, Shape]:
    t, *rest = transformations
    if isinstance(t, OuterTransformation):
        # Recursive call with new spec
        f, shape = _build_recursively(t.new_spec(spec), rest)
        return t(f, spec), t.new_shape(shape)
    elif isinstance(t, InnerTransformation):
        f, shape = _build_recursively(spec, rest)
        return t(f, shape), t.new_shape(shape)
    elif isinstance(t, Kernel):
        t.validate(spec)
        return t.f, t.returned_shape
    else:
        raise TypeError(
            f"Transformations must either be objects of OuterTransformation, InnerTransformation, or Kernel, but one with type {type(t)} was encountered."
        )


@dataclass
class Process:
    data_spec: Spec
    transformations: Sequence[Transformation]
    prefilled_kwargs: dict = field(default_factory=dict)

    def _build(self) -> Tuple[Callable, Spec, Shape]:
        # The last transformation is applied first, therefore we reverse the list
        return _build_recursively(self.data_spec, reversed(self.transformations))

    def __call__(self, *args, **kwargs):
        f, shape = self._build()

        # Bind args and kwargs according to the kernel's signature.
        # We pass bound_args.args (positional args only) because vmap in_axes doesn't
        # work with keyword args.
        kwargs = {**self.prefilled_kwargs, **kwargs}
        kernel_f = self.transformations[0].f
        bound_args = inspect.signature(kernel_f).bind(*args, **kwargs)
        bound_args.apply_defaults()
        return f(*bound_args.args)

    @property
    def shape(self):
        f, shape = self._build()
        return shape

    def with_kwargs(self, **kwargs) -> "Process":
        "Return a copy of this beamformer that has some arguments already filled in."
        return Process(
            self.data_spec, self.transformations, {**self.prefilled_kwargs, **kwargs}
        )

    def __repr__(self) -> str:
        repr_str = f"Process("
        if self.transformations[1:]:
            for fn in self.transformations:
                repr_str += f"\n  {repr(fn)},"
            repr_str += "\n"
        else:
            repr_str += repr(self.transformations[0])
        return repr_str + ")"

    def __hash__(self) -> int:
        return hash(
            (self.data_spec, tuple(self.transformations), repr(self.prefilled_kwargs))
        )
