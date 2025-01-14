from typing import Callable, Optional, Tuple, TypeVar, Union

from spekk.module.base import Module
from spekk.ops._backend import backend
from spekk.ops._types import Dim
from spekk.ops._util import prepare_slicing_along_dim
from spekk.ops.array_object import array

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
    x, slices, dims = prepare_slicing_along_dim(x, i, dim)
    data = x.data[slices]
    return array(data, dims)


def update_indices_along_dim(
    f: Callable[[array], array], x: array, i: array, x_dim: Dim, i_dim: Dim
) -> array:
    from spekk import ops

    x, slices, dims = prepare_slicing_along_dim(x, i, x_dim)
    updated_x_slices = f(array(x.data[slices], dims))
    updated_x_slices._dims = [x_dim if d == i_dim else d for d in updated_x_slices.dims]
    x, updated_x_slices = ops.broadcast_arrays(x, updated_x_slices)
    updated_data = backend._setitem_impl(x.data, slices, updated_x_slices.data)
    return array(updated_data, x.dims)


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
    include_index: bool = False,
) -> Tuple[TCarry, TOutputData]:
    from spekk import ops

    def scan_fn(carry, i):
        args = [carry, data.slice_dim(dim)[i]]
        if include_index:
            args.append(i)
        return f(*args)

    init, y0 = scan_fn(init, 0)
    n = data.dim_size(dim)
    result, ys = backend.scan(scan_fn, init, backend.arange(1, n))
    ys = ops.concat([ops.array([y0]), ys])
    return result, ys


def reduce_over_dim(
    f: Callable[[TCarry, TInputData], Tuple[TCarry, TOutputData]],
    data: TInputData,
    dim: Dim,
    *,
    init: TCarry,
    include_index: bool = False,
):
    def scan_fn(carry, i):
        args = [carry, data.slice_dim(dim)[i]]
        if include_index:
            args.append(i)
        return f(*args), i

    init, _ = scan_fn(init, 0)
    n = data.dim_size(dim)
    result, _ = backend.scan(scan_fn, init, backend.arange(1, n))
    return result


def map_reduce_over_dim(
    map_f: Callable[[TInputData], TOutputData],
    reduce_f: Callable[[TReducedOutputData, TOutputData], TReducedOutputData],
    data: TInputData,
    dim: Dim,
    *,
    init: TCarry,
    include_index: bool = False,
) -> TReducedOutputData:
    def scan_fn(carry, i):
        x = map_f(data.slice_dim(dim)[i])
        args = [carry, x]
        if include_index:
            args.append(i)
        return reduce_f(*args), i

    init, _ = scan_fn(init, 0)
    n = data.dim_size(dim)
    carry, _ = backend.scan(scan_fn, init, backend.arange(1, n))
    return carry


if __name__ == "__main__":
    import doctest

    from spekk import ops

    ops.backend.set_backend("numpy")
    doctest.testmod()
