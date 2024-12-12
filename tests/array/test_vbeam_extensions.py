import random

import numpy as np
from tqdm import tqdm

import spekk.array as ops

ops.backend.set_backend("numpy")

for _ in tqdm(range(1000)):
    dims = {name: size + 1 for name, size in zip("abcdefghijk", range(10))}
    dim = random.choice(list(dims.keys()))
    dims_without_dim = [k for k in dims.keys() if k != dim]
    x_dims = random.sample(dims_without_dim, random.randint(0, len(dims_without_dim)))
    x_dims.append(dim)
    random.shuffle(x_dims)
    i_dims = random.sample(dims_without_dim, random.randint(0, len(dims_without_dim)))

    # x_dims = ["a", "d", "b"]
    # i_dims = ["a", "d", "b"]
    # dim = "d"
    x = ops.array(np.random.randn(*[dims[d] for d in x_dims]), x_dims)
    i = ops.array(
        np.random.randint(0, dims[dim], tuple(dims[d] for d in i_dims)), i_dims
    )

    dim_sizes = {d: s for d, s in zip(x.dims, x.shape)} | {
        d: s for d, s in zip(i.dims, i.shape)
    }
    expected_output_dim_sizes = {d: s for d, s in dim_sizes.items() if d != dim}
    try:
        result = ops.take_along_dim(x, i, dim)
        assert expected_output_dim_sizes == {
            d: s for d, s in zip(result.dims, result.shape)
        }
    except Exception:
        print(f"{x.dims=}")
        print(f"{i.dims=}")
        print(f"{dim=}")
        print(f"{expected_output_dim_sizes=}")
        print(f"{result.dims=}")
        print(f"{result.shape=}")
