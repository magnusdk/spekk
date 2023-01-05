from typing import Callable, Sequence, Union

from spekk import Spec
from spekk.process.transformations import (
    Kernel,
    Transformation,
    TransformationError,
)


def default_repr_fn(t: Union[Transformation, Kernel]) -> str:
    return f"    {repr(t)},"


def get_error_repr_fn(e: TransformationError):
    def repr_fn(t: Union[Transformation, Kernel]):
        return (
            f"⚠   {repr(t)},\n⚠     ↳ This step raised {repr(e.original_exception)}"
            if t is e.raised_by
            else f"    {repr(t)},"
        )

    return repr_fn


def get_process_repr(
    input_spec: Spec,
    transformations: Sequence[Transformation],
    repr_fn: Callable[[Union[Transformation, Kernel]], str],
):
    s = "Process(\n"
    s += f"  {input_spec},\n"
    s += "  [\n"
    if transformations[1:]:
        for t in transformations:
            s += repr_fn(t)
            s += "\n"
    else:
        s += repr(transformations[0])
    s += "  ]\n"
    s += ")"
    return s
