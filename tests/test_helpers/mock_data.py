import numpy as np

from spekk import Spec, trees


def generate_mock_data(
    spec: Spec,
    seed: int = 0,
    min_dimension_size=1,
    max_dimension_size=4,
):
    rng = np.random.default_rng(seed)
    dim_sizes = {
        dim: rng.integers(min_dimension_size, max_dimension_size)
        for dim in spec.dimensions
    }
    return trees.update_leaves(
        spec.tree,
        spec.is_leaf,
        lambda dims: rng.random([dim_sizes[dim] for dim in dims]),
    )
