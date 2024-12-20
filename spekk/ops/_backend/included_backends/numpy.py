import functools

import numpy as np
from array_api_compat.numpy import *
from array_api_compat.numpy import _info

import spekk.ops._backend.common as common

__array_namespace_info__ = _info.__array_namespace_info__


def _python_vmap(f, in_axes=None):
    @functools.wraps(f)
    def wrapped(*args):
        nonlocal in_axes
        if in_axes is None:
            in_axes = [0] * len(args)
        sizes = {np.shape(x)[ax] for x, ax in zip(args, in_axes) if ax is not None}
        if len(sizes) != 1:
            raise ValueError(
                f"Inconsistent sizes among arguments for the given in_axes. {sizes=}"
            )
        (size,) = sizes

        results = []
        for i in range(size):
            results.append(f(*common.get_args_for_index(args, in_axes, i)))
        return np.stack(results, 1)

    return wrapped


vmap = common.get_vmap_fn(_python_vmap)
jit = lambda f: f  # There is no Numpy jit; just return function as-is.


def scan(fn, init, xs):
    carry = init
    result = []
    for x in xs:
        carry, y = fn(carry, x)
        result.append(y)
    return carry, np.stack(result)
