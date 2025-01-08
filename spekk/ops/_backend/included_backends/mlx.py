import functools

import mlx.core as mx
import numpy as np
from mlx.core import *

empty = mx.zeros
empty_like = mx.zeros_like
bool = mx.bool_
float64 = np.float64
complex128 = np.complex128


def _wrap_numpy_fn(np_fn):
    @functools.wraps(np_fn)
    def wrapped(*args, **kwargs):
        return mx.array(np_fn(*args, **kwargs))

    return wrapped


asarray = _wrap_numpy_fn(np.asarray)


class dtype:
    def __init__(self, dtype):
        self.dtype = dtype

    @property
    def name(self): ...
