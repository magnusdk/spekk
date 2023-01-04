from typing import Callable, Sequence, Union

from spekk import Spec
from spekk.process.transformations import (
    Kernel,
    Transformation,
    TransformationException,
)


def default_repr_fn(t: Union[Transformation, Kernel]) -> str:
    return f"    {repr(t)},"


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
