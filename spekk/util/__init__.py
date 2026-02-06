import spekk.util.dims as dims
import spekk.util.jax_util as jax_util
from spekk.util.dims import random_dim_name


def is_array_like(x):
    from spekk import ops

    return isinstance(x, (ops.array, bool, int, float, complex))


__all__ = [
    "dims",
    "jax_util",
    "random_dim_name",
]
