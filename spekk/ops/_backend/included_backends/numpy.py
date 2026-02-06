import functools
from typing import Sequence, Union

import numpy as np
from array_api_compat.numpy import *
from array_api_compat.numpy import _info

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


def jit(f, static_argnums: Sequence[int] = (), static_argnames: Sequence[str] = ()):
    return f


def _scan(fn, init, xs: tuple[np.ndarray, ...], unroll=None):
    from spekk import tree

    carry = init
    # Assume all dynamic arrays have the same size along axis 0.
    size = xs[0].shape[0]

    ys_dynamic = []
    ys_treedef = []
    for t in range(size):
        # For each dynamic array in xs, extract the t-th element along axis 0.
        xs_t = [arr[t] for arr in xs]
        # Reconstruct the tree corresponding to the t-th slice.
        carry, y = fn(carry, xs_t)
        y_dynamic, y_treedef = tree.flatten(y)
        ys_dynamic.append(y_dynamic)
        ys_treedef.append(y_treedef)

    ys_dynamic = [np.stack(a) for a in zip(*ys_dynamic)]
    ys = y_treedef.unflatten(ys_dynamic)
    return carry, ys


scan = common.get_scan_fn(_scan)


def _not_implemented(name: str):
    def _(*args, **kwargs):
        raise NotImplementedError(f"{name} is not implemented for Numpy.")

    return _


grad = _not_implemented("grad")
value_and_grad = _not_implemented("value_and_grad")
checkpoint = _not_implemented("checkpoint")


def get_dtype_name(dtype):
    return dtype.name


def _is_backend_array(x):
    return isinstance(x, np.ndarray)


def flatten(x: np.ndarray) -> np.ndarray:
    return x.flatten()


def to_numpy(x: np.ndarray) -> np.ndarray:
    return x


def correlate2d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    import scipy

    return scipy.signal.correlate2d(x1, x2)


def set_device(device):
    global active_device
    active_device = device


active_device = None
