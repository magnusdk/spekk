import pytest

from spekk.trees import NotTreeLikeError, Tree, traverse, traverse_iter


class CustomTreeLike:
    def __init__(self, value):
        self.value = value

    def __spekk_keys__(self):
        return ["value"]

    def __spekk_children__(self):
        return [self.value]

    def __spekk_is_leaf__(self):
        return lambda x: not Tree.is_tree_like(x)

    def __spekk_recreate__(self, keys, children):
        kwargs = dict(zip(keys, children))
        return CustomTreeLike(**kwargs)

    def __eq__(self, other):
        return isinstance(other, CustomTreeLike) and other.value == self.value


def test_tree_raises_error_for_non_tree_like():
    with pytest.raises(NotTreeLikeError):
        Tree(1)  # 1 is not tree-like


def test_is_tree_like():
    assert Tree.is_tree_like({"a": 1})
    assert not Tree.is_tree_like(1)
    assert Tree.is_tree_like(CustomTreeLike("Hello, World!"))


def test_recreating_tree_like():
    tree = CustomTreeLike("Hello, World!")
    keys = Tree.keys(tree)
    children = Tree.children(tree)
    recreated_tree = Tree.recreate(tree, keys, children)
    assert tree == recreated_tree


def test_recreating_nested_tree_like():
    tree = {"a": CustomTreeLike([[[CustomTreeLike({"b": "Hello, World!"})]]])}
    # Traversing a tree with the identity function is like recreating the whole tree
    recreated_tree = traverse(lambda x: x, tree)
    assert tree == recreated_tree


def test_tree_get():
    data = {"a": {"b": [{"c": 1}, {"d": 2}]}}
    assert Tree.get(data) == data
    assert Tree.get(Tree(data)) == data
    assert Tree.get(data, "a", "b", 0) == {"c": 1}
    assert Tree.get(data, "a", "b", 1, "d") == 2
    with pytest.raises(KeyError):
        assert Tree.get(data, "a", "b", 2)
    with pytest.raises(KeyError):
        assert Tree.get(data, "k")


def test_tree_set():
    data = {"a": [1, 2, 3]}
    assert Tree.set(data, "a", "foo") == {"a": "foo"}


def test_tree_update():
    data = {"a": [1, 2, 3]}
    assert Tree.update(data, "a", len) == {"a": 3}


def test_treelens_raises_key_error():
    tree = Tree({"a": {"b": [{"c": 1}, {"d": 2}]}})
    with pytest.raises(KeyError):
        assert tree.at("a", "k")


def test_treelens_get():
    tree = Tree({"a": {"b": [{"c": 1}, {"d": 2}]}})
    assert tree.at("a", "b", 0, "c").get() == 1
    assert tree.at("a", "b", 1, "d").get() == 2
    assert tree.at("a", "b").get() == [{"c": 1}, {"d": 2}]


def test_treelens_set():
    data = {"a": {"b": [{"c": 1}, {"d": 2}]}}
    new_data = Tree(data).at("a", "b", 0, "c").set(100)
    assert new_data == {"a": {"b": [{"c": 100}, {"d": 2}]}}


def test_treelens_update():
    tree = Tree({"a": {"b": {"c": [1, 2, 3]}}})
    assert tree.at("a", "b", "c").update(len) == {"a": {"b": {"c": 3}}}


def test_traverse():
    # We can have stateful callbacks when traversing nodes
    # Let's store the order of visited nodes in the traversed_nodes variable
    traversed_nodes = []
    tree = {"a": {"b": [{"c": 1}, {"d": 2}]}}
    traverse(lambda x: traversed_nodes.append(x) or x, tree)
    # Nodes are visited depth-first left-to-right
    assert traversed_nodes == [
        1,  # Leaf
        {"c": 1},
        2,  # Leaf
        {"d": 2},
        [{"c": 1}, {"d": 2}],
        {"b": [{"c": 1}, {"d": 2}]},
        {"a": {"b": [{"c": 1}, {"d": 2}]}},
    ]

    # Leaves/nodes can be modified as we go to get a new modified copy of the tree
    # Let's modify the leaves (ints) of this tree
    new_tree = traverse(lambda x: x * 2 if isinstance(x, int) else x, tree)
    assert new_tree == {"a": {"b": [{"c": 2}, {"d": 4}]}}
    # The original tree is unchanged
    assert tree == {"a": {"b": [{"c": 1}, {"d": 2}]}}

    # Let's modify some of the nodes (f.ex. all lists)
    # Let's only take the first element of each list
    new_tree = traverse(lambda x: x[0] if isinstance(x, list) else x, tree)
    assert new_tree == {"a": {"b": {"c": 1}}}
    # The original tree is unchanged
    assert tree == {"a": {"b": [{"c": 1}, {"d": 2}]}}


def test_traverse_multiple():
    # An arbitrary example:
    # Take every even-indexed element if the node in tree2 is True, every odd-indexed
    # element if the node in tree2 is False
    result = traverse(
        lambda node1, node2: (
            node1[::2] if node2 is True else node1[1::2] if node2 is False else node1
        ),
        {"a": {"b": [list(range(10)), list(range(10))]}},
        {"a": {"b": [True, False]}},
    )
    assert result == {"a": {"b": [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]}}


def test_traverse_iter():
    tree = {"a": {"b": [{"c": 1}, {"d": 2}]}}
    traversed_nodes = list(traverse_iter(tree))
    # Nodes are visited depth-first left-to-right
    assert traversed_nodes == [
        1,  # Leaf
        {"c": 1},
        2,  # Leaf
        {"d": 2},
        [{"c": 1}, {"d": 2}],
        {"b": [{"c": 1}, {"d": 2}]},
        {"a": {"b": [{"c": 1}, {"d": 2}]}},
    ]


def test_tree_leaves():
    assert Tree.leaves({"a": [1, 2], "b": {"c": 3}}) == [1, 2, 3]


def test_flatten():
    tree = {"a": [1, 2], "b": {"c": 3}}
    flattened_tree, unflatten = Tree.flatten(tree)
    assert flattened_tree == [1, 2, 3]
    assert unflatten(flattened_tree) == tree

    tree = CustomTreeLike({"a": [1, 2], "b": {"c": 3}})
    flattened_tree, unflatten = Tree.flatten(tree)
    assert flattened_tree == [1, 2, 3]
    assert unflatten(flattened_tree) == tree


def test_merge():
    merged_trees = Tree.merge(
        {"a": 1, "b": 11},
        {"a": 2, "b": 12},
        {"a": 3, "b": 13},
        {"a": 4, "b": 14},
        by=list,
    )
    assert merged_trees == {"a": [1, 2, 3, 4], "b": [11, 12, 13, 14]}


def test_repr():
    tree = Tree({"a": [1, 2], "b": {"c": 3}})
    assert repr(tree) == "Tree({'a': [1, 2], 'b': {'c': 3}})"


if __name__ == "__main__":
    pytest.main([__file__])
