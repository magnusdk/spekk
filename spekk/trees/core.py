from dataclasses import dataclass
from typing import Any, Callable, Generator

from spekk.trees.registry import Tree, treedef


def update(tree: Tree, f: Callable[[Tree], Tree], path: tuple):
    """Update the subtree at the given path.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4}
    >>> update(tree, lambda x: x + 10, ("a", 1, "b"))
    {'a': [1, {'b': 12}, 3], 'c': 4}
    """
    if not path:
        return f(tree)

    key, *remaining_path = path
    td = treedef(tree)
    values = [
        update(td.get(k), f, remaining_path) if k == key else td.get(k)
        for k in td.keys()
    ]
    return td.create(td.keys(), values)


def get(tree: Tree, path: tuple):
    """Get the subtree at the given path.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4}
    >>> get(tree, ("a", 1, "b"))
    2
    """
    if not path:
        return tree

    key, *remaining_path = path
    td = treedef(tree)
    return get(td.get(key), remaining_path)


def set(tree: Tree, value: Any, path: tuple):
    """Set the value of the subtree at the given path.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4}
    >>> set(tree, 42, ("a", 1, "b"))
    {'a': [1, {'b': 42}, 3], 'c': 4}
    """
    return update(tree, lambda _: value, path)


def remove(tree: Tree, path: tuple):
    """Remove the value of the subtree at the given path.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4, "d": 5}
    >>> remove(tree, ("a", 1, "b"))
    {'a': [1, {}, 3], 'c': 4, 'd': 5}
    >>> remove(tree, ("a", 1))
    {'a': [1, 3], 'c': 4, 'd': 5}
    >>> remove(tree, ("c",))
    {'a': [1, {'b': 2}, 3], 'd': 5}
    """
    def remove_sub_tree(tree):
        td = treedef(tree)
        keys = [k for k in td.keys() if k != path[-1]]
        values = [td.get(k) for k in keys]
        return td.create(keys, values)

    return update(tree, remove_sub_tree, path[:-1])


def merge(t1: Tree, t2: Tree, preserve_order: int = "first") -> Tree:
    """Merge two trees (assuming this is possible).

    The order of the keys in the merged tree is determined by the preserve_order. If it
    is first, then the order is the same as the first tree. If it is last, then the
    order is the same as the second tree. Any other new keys are appended to the end of
    the tree.

    >>> merge({"a": 1, "b": 2}, {"c": 3})
    {'a': 1, 'b': 2, 'c': 3}
    >>> merge({"a": 1, "b": 2}, {"b": 7, "c": 3})
    {'a': 1, 'b': 7, 'c': 3}

    Merging two lists treats the indices as keys, and the second tree overwrites the
    indices of the first one.
    >>> merge([1, 2, 3], [4, 5])
    [4, 5, 3]
    """
    treedef1, treedef2 = treedef(t1), treedef(t2)

    if preserve_order == "first":
        merged = dict(zip(treedef1.keys(), treedef1.values()))
        for key in treedef2.keys():
            merged[key] = treedef2.get(key)
    elif preserve_order == "last":
        merged = dict(zip(treedef2.keys(), treedef2.values()))
        for key in treedef1.keys():
            if key not in merged:
                merged[key] = treedef1.get(key)
    else:
        raise ValueError("preserve_order must be either 'first' or 'last'")

    return treedef1.create(merged.keys(), merged.values())


@dataclass
class TraversalItem:
    "An object returned from the traverse and leaves generator functions."
    value: Any
    path: tuple
    is_leaf: bool


def traverse(
    tree: Tree,
    is_leaf: Callable[[Tree], bool],
    path: tuple = (),
) -> Generator[TraversalItem, None, None]:
    """Traverse a tree and yield all nodes (subtrees and leaves) as TraversalItem
    objects. The traversal is depth-first, left-to-right.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4}
    >>> for item in traverse(tree, is_leaf=lambda x: isinstance(x, int)):
    ...     print(item.value)
    1
    2
    {'b': 2}
    3
    [1, {'b': 2}, 3]
    4
    {'a': [1, {'b': 2}, 3], 'c': 4}
    """
    if is_leaf(tree):
        yield TraversalItem(tree, path, True)
    else:
        td = treedef(tree)
        values = []
        for key in td.keys():
            subtrees = list(traverse(td.get(key), is_leaf, path + (key,)))
            for subtree in subtrees:
                yield subtree
            values.append(subtrees[-1].value)
        yield TraversalItem(td.create(td.keys(), values), path, False)


def leaves(
    tree: Tree,
    is_leaf: Callable[[Tree], bool],
    path: tuple = (),
) -> Generator[TraversalItem, None, None]:
    """Traverse a tree and yield all leaves as TraversalItem objects. The traversal is
    depth-first, left-to-right.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4}
    >>> for item in leaves(tree, is_leaf=lambda x: isinstance(x, int)):
    ...     print(item.value)
    1
    2
    3
    4
    """
    return (t for t in traverse(tree, is_leaf, path) if t.is_leaf)


def filter(
    tree: Tree,
    is_leaf: Callable[[Tree], bool],
    predicate: Callable[[Tree], bool],
    path: tuple = (),
) -> Generator[TraversalItem, None, None]:
    """Remove all subtrees for which the predicate returns False.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4, "d": 5}
    >>> is_leaf = lambda x: isinstance(x, int)
    >>> filter(tree, is_leaf, lambda x: x % 2 == 1 if is_leaf(x) else True)
    {'a': [1, {}, 3], 'd': 5}
    """
    state = tree
    for t in traverse(tree, is_leaf, path):
        if not predicate(t.value):
            state = remove(state, t.path)
    return state


if __name__ == "__main__":
    import doctest

    doctest.testmod()
