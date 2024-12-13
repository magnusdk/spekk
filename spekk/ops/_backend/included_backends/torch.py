import torch
from array_api_compat.torch import *

import spekk.ops._backend.common as common

vmap = common.get_vmap_fn(torch.vmap)
jit = common.get_jit_fn(torch.compile)


def scan(fn, init, xs):
    carry = init
    result = []
    for x in xs:
        carry, y = fn(carry, x)
        result.append(y)
    return carry, np.stack(result)
