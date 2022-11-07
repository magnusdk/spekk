# TODO: Add docstrings.

from typing import Any, Mapping, Sequence, Union

from spekk2.trees.registry import repr, treedef

Tree = Union[Mapping[Any, "Tree"], Sequence["Tree"], Any]


def update(tree, f, path):
    if not path:
        return f(tree)

    key, *remaining_path = path
    keys, get, create, repr = treedef(tree)
    values = [
        update(get(tree, k), f, remaining_path) if k == key else get(tree, k)
        for k in keys
    ]
    return create(keys, values)


def set(tree, value, path):
    return update(tree, lambda _: value, path)


def remove(tree, path):
    def remove_sub_tree(tree):
        keys, get, create, repr = treedef(tree)
        values = [get(tree, k) for k in keys if k != path[-1]]
        return create(keys, values)

    return update(tree, remove_sub_tree, path[:-1])


def traverse(tree, is_leaf, f):
    if is_leaf(tree):
        return f(tree)

    keys, get, create, repr = treedef(tree)
    values = [traverse(get(tree, k), is_leaf, f) for k in keys]
    return f(create(keys, values))


def traverse_with_state(tree, is_leaf, f, state, path=()):
    if is_leaf(tree):
        return f(state, tree, path)

    keys, get, create, repr = treedef(tree)
    values = []
    for k in keys:
        state, new_value = traverse_with_state(
            get(tree, k), is_leaf, f, state, path + (k,)
        )
        values.append(new_value)
    return f(state, create(keys, values), path)


def tree_repr(tree, is_leaf):
    return traverse(tree, is_leaf, repr)
