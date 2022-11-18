# TODO: Add docstrings.

from typing import Any, Mapping, Sequence, Union

from spekk2.trees import registry

Tree = Union[Mapping[Any, "Tree"], Sequence["Tree"], Any]


def update(tree, f, path):
    if not path:
        return f(tree)

    key, *remaining_path = path
    keys, get, create, repr = registry.treedef(tree)
    values = [
        update(get(tree, k), f, remaining_path) if k == key else get(tree, k)
        for k in keys
    ]
    return create(keys, values)


def set(tree, value, path):
    return update(tree, lambda _: value, path)


def remove(tree, path):
    def remove_sub_tree(tree):
        keys, get, create, repr = registry.treedef(tree)
        values = [get(tree, k) for k in keys if k != path[-1]]
        return create(keys, values)

    return update(tree, remove_sub_tree, path[:-1])


def traverse(tree, is_leaf, f, path=(), use_path=False):
    if is_leaf(tree):
        return f(tree, path) if use_path else f(tree)

    keys, get, create, repr = registry.treedef(tree)
    values = [traverse(get(tree, k), is_leaf, f, path + (k,), use_path) for k in keys]
    new_tree = create(keys, values)
    return f(new_tree, path) if use_path else f(new_tree)


def traverse_with_state(tree, is_leaf, f, state, path=(), use_path=False):
    if is_leaf(tree):
        return f(state, tree, path) if use_path else f(state, tree)

    keys, get, create, repr = registry.treedef(tree)
    values = []
    for k in keys:
        state, new_value = traverse_with_state(
            get(tree, k), is_leaf, f, state, path + (k,), use_path
        )
        values.append(new_value)
    new_tree = create(keys, values)
    return f(state, new_tree, path) if use_path else f(state, new_tree)


def tree_repr(tree, is_leaf):
    return traverse(tree, is_leaf, lambda x: registry.repr(x) if not is_leaf(x) else x)