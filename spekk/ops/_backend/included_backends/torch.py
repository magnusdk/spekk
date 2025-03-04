import numpy
import torch
from torch import *
from array_api_compat.torch import *
from array_api_compat.torch import _info

import spekk.ops._backend.common as common

__array_namespace_info__ = _info.__array_namespace_info__

vmap = common.get_vmap_fn(torch.vmap)
jit = common.get_jit_fn(torch.compile)


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

