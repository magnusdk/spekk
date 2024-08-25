import numpy as np
import pytest

from spekk.common import vmap_for_loop_impl
from spekk.trees import Tree


class CustomTreeLike:
    def __init__(self, sum, prod):
        self.sum = sum
        self.prod = prod

    def __spekk_keys__(self):
        return ["sum", "prod"]

    def __spekk_children__(self):
        return [self.sum, self.prod]

    def __spekk_is_leaf__(self):
        return Tree.is_tree_like

    def __spekk_recreate__(self, keys, children):
        kwargs = dict(zip(keys, children))
        return CustomTreeLike(**kwargs)


def test_vmap_for_loop_impl():
    def f(x, y):
        return x + y

    f_vmapped = vmap_for_loop_impl(f, [0, None])
    f_vmapped = vmap_for_loop_impl(f_vmapped, [None, 0])
    result = f_vmapped(np.ones(5), np.ones(4))
    np.testing.assert_equal(result, 2 * np.ones((4, 5)))


def test_vmap_for_loop_impl_custom_type():
    def f(x, y):
        return CustomTreeLike(x + y, x * y)

    f = vmap_for_loop_impl(f, [0, None])
    f = vmap_for_loop_impl(f, [None, 0])
    result = f(np.ones(5) * 3, np.ones(4) * 3)
    assert isinstance(result, CustomTreeLike)
    np.testing.assert_equal(result.sum, 6 * np.ones((4, 5)))
    np.testing.assert_equal(result.prod, 9 * np.ones((4, 5)))


if __name__ == "__main__":
    pytest.main([__file__])
