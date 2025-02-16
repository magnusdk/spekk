import functools
from typing import Sequence, Union

import numpy as np
from array_api_compat.numpy import *
from array_api_compat.numpy import _info
from numpy import *

import spekk.ops._backend.common as common

__array_namespace_info__ = _info.__array_namespace_info__


def getitem_along_axis(x, axis: int, i: int):
    slice_ = tuple([slice(None)] * axis + [i, ...])
    try:
        return x.__getitem__(slice_)
    except TypeError:
        try:
            return np.array(x).__getitem__(slice_)
        except Exception:
            raise ValueError(
                f"Cannot get item at index {i} along axis {axis} for {x!r}"
            )


def get_args_for_index(
    args: Sequence, in_axes: Sequence[Union[int, None]], i: int
) -> Sequence:
    return [
        getitem_along_axis(arg, a, i) if a is not None else arg
        for arg, a in zip(args, in_axes)
    ]


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
            results.append(f(*get_args_for_index(args, in_axes, i)))
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


def get_dtype_name(dtype):
    return dtype.name


def _is_backend_array(x):
    return isinstance(x, np.ndarray)


def flatten(x: np.ndarray) -> np.ndarray:
    return x.flatten()
