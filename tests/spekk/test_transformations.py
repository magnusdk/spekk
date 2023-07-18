import functools

import jax
import jax.numpy as jnp

from spekk import Spec
from spekk.transformations import *


def kernel_fn(a: float, b: float, c: float):
    return a * b + a * c[0] + b * c[1]


data = {
    "a": jnp.ones((2, 3, 4)),
    "b": jnp.ones((4, 2)),
    "c": [jnp.ones((2, 5)), jnp.ones((3,))],
}
spec = Spec(
    {
        "a": ["a", "b", "c"],
        "b": ["c", "a"],
        "c": (["a", "d"], ["b"]),
    }
)

p = compose(
    kernel_fn,
    ForAll("c", jax.vmap),
    ForAll("a", jax.vmap),
    ForAll("b", jax.vmap),
    Reduce.Sum("d", reduce_impl=functools.reduce),
    Apply(jnp.sum, Axis("b")),
    Wrap(jax.jit),
).build(spec)
result = p(**data)
print(result.shape)
