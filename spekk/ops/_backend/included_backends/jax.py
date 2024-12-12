import jax
from jax.numpy import *

import spekk.ops._backend.common as common

vmap = common.get_vmap_fn(jax.vmap)
jit = common.get_jit_fn(jax.jit)
