from typing import Callable, Optional, Tuple, TypeVar, Union

from spekk.module.base import Module
from spekk.ops._backend import backend
from spekk.ops._types import Dim, Dims
from spekk.ops.array_object import array
from spekk.ops.creation_functions import arange
from spekk.ops.data_type_functions import astype
from spekk.ops.manipulation_functions import moveaxis

TFunc = TypeVar("TFunc", bound=Callable)
TCarry = TypeVar("TCarry")
TInputData = TypeVar("TInputData", bound=Module)
TOutputData = TypeVar("TOutputData")
TReducedOutputData = TypeVar("TReducedOutputData")


def deg2rad(x: array) -> array:
    return x / 180 * backend.pi


def rad2deg(x: array) -> array:
    return x * 180 / backend.pi


def nan_to_num(
    x: array,
    nan: Optional[float] = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> array:
    from spekk import ops

    if nan is not None:
        x = ops.where(x == ops.nan, nan, x)
    if posinf is not None:
        x = ops.where(x == ops.inf, posinf, x)
    if neginf is not None:
        x = ops.where(x == -ops.inf, neginf, x)
    return x


def take_along_dim(x: array, i: array, dim: Dim) -> array:
    # Ensure dim is the first axis. This makes it easier to keep track of dimensions.
    if x._dims.index(dim) != 0:
        x = moveaxis(x, dim, 0)

    common_dims = set(x._dims) & set(i._dims) - {dim}
    slices = [slice(None)] * x.ndim
    slices[x._dims.index(dim)] = i._data
    for d in common_dims:
        dim_idx = x._dims.index(d)
        dim_size = x.shape[dim_idx]
        broadcastable_shape = [1] * i.ndim
        broadcastable_shape[i._dims.index(d)] = dim_size
        slices[dim_idx] = backend.reshape(backend.arange(dim_size), broadcastable_shape)

    data = x._data[tuple(slices)]
    dims = i._dims + [d for d in x._dims if d not in i._dims and d != dim]
    return array(data, dims)


def expand_slice_to_axis(s: Union[slice, int, array], axis: int):
    """Make a given slice s work along a given axis.

    Example:
    >>> from spekk import ops
    >>> arr = ops.reshape(ops.arange(8), (2, 2, 2))
    >>> s = expand_slice_to_axis(slice(1, 2), 1)
    >>> arr[s]
    array(shape=(2, 1, 2), dims=[?, ?, ?], dtype=int64, data=[[[2 3]]
    <BLANKLINE>
     [[6 7]]])
    >>> arr[s] = -1
    >>> arr
    array(shape=(2, 2, 2), dims=[?, ?, ?], dtype=int64, data=[[[ 0  1]
      [-1 -1]]
    <BLANKLINE>
     [[ 4  5]
      [-1 -1]]])
    """
    return (slice(None),) * axis + (s, ...)


def jit(f: TFunc) -> TFunc:
    return backend.jit(f)


def scan_over_dim(
    f: Callable[[TCarry, TInputData], Tuple[TCarry, TOutputData]],
    data: TInputData,
    dim: Dim,
    *,
    init: TCarry,
) -> Tuple[TCarry, TOutputData]:

    def scan_fn(carry, i):
        return f(carry, data.slice_dim(dim)[i])

    n = data.dim_size(dim)
    return backend.scan(scan_fn, init, arange(n))


def map_reduce_over_dim(
    map_f: Callable[[TInputData], TOutputData],
    reduce_f: Callable[[TReducedOutputData, TOutputData], TReducedOutputData],
    data: TInputData,
    dim: Dim,
    *,
    init: Optional[TCarry] = None,
) -> TReducedOutputData:
    def scan_fn(carry, i):
        x = map_f(data.slice_dim(dim)[i])
        return reduce_f(carry, x), i

    n = data.dim_size(dim)
    if init is None:
        init = map_f(data.slice_dim(dim)[0])
        xs = backend.arange(1, n)
    else:
        xs = backend.arange(n)

    carry, _ = backend.scan(scan_fn, init, xs)
    return carry


if __name__ == "__main__":
    import doctest

    from spekk import ops

    ops.backend.set_backend("numpy")
    doctest.testmod()
