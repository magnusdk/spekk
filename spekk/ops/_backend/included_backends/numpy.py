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
    from spekk.module import base as module_base

    carry = init
    flat_xs = module_base.flatten(xs)
    # Assume all dynamic arrays have the same size along axis 0.
    T = flat_xs.dynamic[0].shape[0]

    # Will hold the flattened outputs (list of lists of arrays) for each time step.
    results_flat = []

    for t in range(T):
        # For each dynamic array in xs, extract the t-th element along axis 0.
        xs_t_dynamic = [arr[t] for arr in flat_xs.dynamic]
        # Reconstruct the tree corresponding to the t-th slice.
        xs_t = flat_xs.unflatten(xs_t_dynamic)
        carry, y = fn(carry, xs_t)
        flat_y = module_base.flatten(y)
        results_flat.append(flat_y.dynamic)

    # Now, results_flat is a list of lists of arrays. The inner list corresponds to the tree's leaves.
    # For each leaf, stack all T outputs along a new first axis.
    num_leaves = len(results_flat[0])
    dynamic_result = []
    for leaf_idx in range(num_leaves):
        # Gather the same leaf from each time step.
        leaf_values = [results_flat[t][leaf_idx] for t in range(T)]
        dynamic_result.append(np.stack(leaf_values, axis=0))

    # Use the flattening scheme from the output tree to unflatten back to a tree structure.
    tree_result = flat_y.unflatten(dynamic_result)
    return carry, tree_result


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