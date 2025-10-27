from typing import Optional, Union
import numpy
import torch
from torch import *
from array_api_compat.torch import *
from array_api_compat.torch import _info

import spekk.ops._backend.common as common

__array_namespace_info__ = _info.__array_namespace_info__

vmap = common.get_vmap_fn(torch.vmap)
jit = common.get_jit_fn(torch.compile)


def scan(fn, init, xs, unroll=None):
    from spekk.module import base as module_base

    carry = init
    flat_xs = module_base.flatten(xs)
    # Assume all dynamic arrays have the same size along axis 0.
    T = flat_xs.dynamic[0].shape[0]

    # Will hold the flattened outputs (list of lists of arrays) for each time step.
    results_flat = []

    for t_np in numpy.arange(0, T, dtype=numpy.int64):
        t = t_np.item()
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
    # for leaf_idx in numpy.arange(0, num_leaves, dtype=int):
    for leaf_idx_np in numpy.arange(0, num_leaves, dtype=numpy.int64):
        leaf_idx = leaf_idx_np.item()        
        # Gather the same leaf from each time step.
        leaf_values = [results_flat[t][leaf_idx] for t in range(T)]
        dynamic_result.append(torch.stack(leaf_values, axis=0))

    # Use the flattening scheme from the output tree to unflatten back to a tree structure.
    tree_result = flat_y.unflatten(dynamic_result)
    return carry, tree_result


def get_dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def _is_backend_array(x):
    return isinstance(x, torch.Tensor)


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.flatten()    

def to_numpy(x: torch.Tensor) -> numpy.ndarray:
    return x.cpu().numpy()

def correlate2d(x1, x2) -> torch.Tensor:
    pad = (x2.shape[0]-1, x2.shape[1]-1)
    x2 = torch.conj(x2)
    a = torch.nn.functional.conv2d(x1[None, None, :,:], x2[None, None, :,:], padding=pad)[0,0]  
    return a 

def pad(x, pad_width: tuple, mode: str='constant', reflect_type: Union[str, None]=None):
    
    if mode=="edge":
        mode = "replicate"

    pad_width_flat = []
    if isinstance(pad_width, builtins.int): 
        for d in builtins.range(x.ndim):
            pad_width_flat += [pad_width, pad_width]
        pad_width = pad_width_flat

    elif isinstance(pad_width[0], builtins.int):
        for d in builtins.range(x.ndim):
            pad_width_flat += [pad_width[0], pad_width[1]]
        pad_width = pad_width_flat

    else:
        pad_width_flat = []
        for ps in pad_width:
            for p in ps[::-1]:
                pad_width_flat.append(p)

        # reverse order:
        pad_width_flat.reverse()
        pad_width = pad_width_flat

    return torch.nn.functional.pad(x, pad=pad_width, mode=mode)

# "take" is part of array compat lib but there is a mismatch where numpy/jax squeezes the output 
# array if indices is a number, whereas torch keep the singleton dimension.
# def take(x: torch.Tensor, indices: torch.Tensor, /, *, axis: Optional[int] = None, **kwargs) -> torch.Tensor:
def take(x: torch.Tensor, indices: torch.Tensor, /, *, axis = None, **kwargs) -> torch.Tensor:
    if axis is None:
        if x.ndim != 1:
            raise ValueError("axis must be specified when ndim > 1")
        axis = 0

    arr = torch.index_select(x, axis, indices, **kwargs)

    # Missing in array compat api
    if arr.shape[axis]==1:
        arr = arr.squeeze(axis)

    return arr

def set_device(device):
    global active_device
    active_device = device


# Set default active device. Tried in order: ["cuda", "mps", "cpu"].
if torch.cuda.is_available():
    active_device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    active_device = "mps"
else:
    active_device = "cpu"
