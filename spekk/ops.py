from typing import Any, Callable, Sequence

from spekk.common import Specable
from spekk.spec import Spec


def apply_across_dim(
    data: Sequence[Any],
    data_spec: Spec,
    dim: str,
    f: Callable[[Specable, int], Specable],
) -> Sequence[Any]:
    return [
        f(data_item, axis) if axis is not None else data_item
        for axis, data_item in zip(data_spec.indices_for(dim), data)
    ]
