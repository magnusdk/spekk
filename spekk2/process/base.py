import inspect
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple, Union

from spekk2.process.transformations import Kernel, Transformation
from spekk2.spec import Spec


def _build_recursively(
    input_spec: Spec,
    transformations: Sequence[Union[Kernel, Transformation]],
) -> Tuple[Callable, Spec]:
    t, *rest = transformations
    if isinstance(t, Transformation):
        f, output_spec = _build_recursively(t.preprocess_spec(input_spec), rest)
        return t(f, input_spec, output_spec), t.postprocess_spec(output_spec)
    elif isinstance(t, Kernel):
        return t.f, t.output_spec
    else:
        raise TypeError(
            f"Transformations must either be a Transformation or a Kernel, but got {t}."
        )


@dataclass
class Process:
    input_spec: Spec
    transformations: Sequence[Transformation]
    prefilled_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.transformed_kernel, self.output_spec = _build_recursively(
            # The last transformation is applied first, therefore we reverse the list
            self.input_spec,
            reversed(self.transformations),
        )

    def __call__(self, *args, **kwargs):
        kwargs = {**self.prefilled_kwargs, **kwargs}
        # Bind args and kwargs according to the kernel's signature.
        # We pass bound_args.args (positional args only) because vmap in_axes doesn't
        # work well with keyword args.
        kernel_f = self.transformations[0].f
        bound_args = inspect.signature(kernel_f).bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self.transformed_kernel(*bound_args.args)

    def __repr__(self) -> str:
        # TODO: FIX ME
        repr_str = "Process(\n"
        repr_str += f"  {self.input_spec},\n"
        repr_str += "  ["
        if self.transformations[1:]:
            for fn in self.transformations:
                repr_str += f"\n    {repr(fn)},"
            repr_str += "\n"
        else:
            repr_str += repr(self.transformations[0])
        repr_str += "  ]\n"
        return repr_str + ")"

    def __hash__(self) -> int:
        return object.__hash__(self)
