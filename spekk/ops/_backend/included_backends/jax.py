from typing import Literal

import jax
import numpy
from jax.numpy import *

import spekk.ops._backend.common as common

vmap = common.get_vmap_fn(jax.vmap)
jit = common.get_jit_fn(jax.jit)
scan = common.get_scan_fn(jax.lax.scan)
grad = jax.grad
checkpoint = jax.checkpoint


def get_dtype_name(dtype):
    if not hasattr(dtype, "name"):
        dtype = dtype.dtype
    return dtype.name


def _is_backend_array(x):
    return isinstance(x, jax.Array)


def convolve1d(
    x: jax.Array,
    filter: jax.Array,
    *,
    mode: Literal["full", "same", "valid"],
    axis: int,
):
    from jax.scipy.signal import convolve

    return apply_along_axis(lambda m: convolve(m, filter, mode=mode), axis, x)


def flatten(x: jax.Array) -> jax.Array:
    return x.flatten()


def to_numpy(x: jax.Array) -> numpy.ndarray:
    return numpy.array(x)


def correlate2d(x1, x2) -> jax.Array:
    from jax.scipy.signal import convolve2d

    x2 = jax.numpy.flip(x2, axis=0)
    x2 = jax.numpy.flip(x2, axis=1)
    x2 = jax.numpy.conj(x2)
    output = convolve2d(x1, x2, mode="full")
    return output


def set_device(device):
    global active_device
    active_device = device


active_device = None
