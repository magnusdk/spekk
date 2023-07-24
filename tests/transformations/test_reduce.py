import hypothesis as ht
import numpy as np
from test_helpers.generators.spec import kwargs_specs
from test_helpers.mock_data import generate_mock_data

from spekk import Spec, trees
from spekk.transformations import reduce


def _add_leaves(o1: trees.Tree, o2: trees.Tree):
    state = o1
    for leaf in trees.leaves(state, lambda x: isinstance(x, (np.ndarray, float, int))):
        state = trees.set(state, leaf.value + trees.get(o2, leaf.path), leaf.path)
    return state


def _sum_tree_with_numpy(tree: trees.Tree, spec: Spec, dimension: str):
    dim_size = spec.size(tree, dimension)
    for leaf in trees.leaves(spec.index_for(dimension), is_axis):
        axis = leaf.value
        if axis is None:
            # If the axis is None it will be added to itself for each iteration, aka
            # dim_size times.
            tree = trees.update(tree, lambda x: x * dim_size, leaf.path)
        else:
            # If the axis is an integer, it represents the dimension and is summed over.
            tree = trees.update(tree, lambda x: np.sum(x, axis), leaf.path)
    return tree


is_axis = lambda x: isinstance(x, int) or x is None
is_array_leaf = lambda x: isinstance(x, (np.ndarray, float, int))


@ht.given(kwargs_specs())
def test_reduce_addition_is_same_as_sum(spec: Spec):
    data = generate_mock_data(spec)
    for dimension in spec.dimensions:
        reduce_result = reduce.specced_map_reduce(
            dict, _add_leaves, data, spec, dimension
        )
        sum_result = _sum_tree_with_numpy(data, spec, dimension)
        assert trees.are_equal(reduce_result, sum_result, is_array_leaf, np.array_equal)
