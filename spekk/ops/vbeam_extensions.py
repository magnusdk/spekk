from spekk.ops._backend import backend
from spekk.ops._types import Dim
from spekk.ops.array_object import array
from spekk.ops.data_type_functions import astype
from spekk.ops.manipulation_functions import moveaxis


def int32(x: array, copy: bool = True) -> array:
    return astype(x, backend.int32, copy=copy)


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


def jit(f):
    return backend.jit(f)
