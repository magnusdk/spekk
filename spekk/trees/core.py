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


def set(tree: Tree, value: Any, path: tuple):
    """Set the value of the subtree at the given path.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4}
    >>> set(tree, 42, ("a", 1, "b"))
    {'a': [1, {'b': 42}, 3], 'c': 4}
    """
    return update(tree, lambda _: value, path)


def remove(tree: Tree, path: tuple):
    """Remove the value of the subtree at the given path.

    >>> tree = {"a": [1, {"b": 2}, 3], "c": 4}
    >>> remove(tree, ("a", 1, "b"))
    {'a': [1, {}, 3], 'c': 4}
    >>> remove(tree, ("a", 1))
    {'a': [1, 3], 'c': 4}
    """

    def remove_sub_tree(tree):
        td = treedef(tree)
        values = [td.get(k) for k in td.keys() if k != path[-1]]
        return td.create(td.keys(), values)

    return update(tree, remove_sub_tree, path[:-1])


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
