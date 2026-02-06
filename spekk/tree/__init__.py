"""Tree operations for nested Python structures.

A "tree" is any nested structure of tuples, lists, dicts, sets, or Modules.
The "leaves" are all other values (numbers, strings, arrays, None, etc.).

Static vs Dynamic Values
------------------------
When flattening a tree, values are either "dynamic" (included in the leaves list)
or "static" (stored in the TreeDef structure itself). Static values become part
of the tree's structure and are restored as-is during unflatten.

There are three ways to make values static:

1. **is_static predicate**: Pass `is_static=lambda x: ...` to `flatten()`.
   Values where the predicate returns True are stored in the TreeDef.

2. **Module static fields**: Mark fields with `field(static=True)` on Module
   subclasses. These fields are always static regardless of predicates.

3. **static_value wrapper**: Wrap any value with `static_value(x)`. The wrapped
   value is stored in the TreeDef (unwrapped). Useful for explicitly marking
   specific values as static, e.g., in JIT compilation.

Example:
    >>> obj = {"a": [1, 2], "b": (3, {4, 5})}
    >>> leaves, treedef = flatten(obj)
    >>> leaves
    (1, 2, 3, 4, 5)
    >>> treedef
    TreeDef({'a': [*, *], 'b': (*, {*, *})})
    >>> treedef.unflatten([10, 20, 30, 40, 50])
    {'a': [10, 20], 'b': (30, {40, 50})}

Example with static_value:
    >>> obj = {"a": 1, "b": static_value("config")}
    >>> leaves, treedef = flatten(obj)
    >>> leaves
    (1,)
    >>> treedef.unflatten([42])
    {'a': 42, 'b': 'config'}
"""

import functools
from typing import Any, Callable, NamedTuple

import spekk.tree.registry as registry
from spekk.tree.registry import Leaf, StaticLeaf, TreeDef, get_destructure


class static_value[T]:
    """Wrapper that marks a value as static during flatten.

    The wrapped value will be stored in the TreeDef structure rather than
    in the leaves list. During unflatten, the original value is restored.
    """

    __slots__ = ("value",)

    def __init__(self, value: T):
        self.value = value

    def __repr__(self) -> str:
        return f"static_value({self.value!r})"


class Flattened(NamedTuple):
    leaves: tuple
    treedef: TreeDef | Leaf | StaticLeaf


def flatten(
    tree: Any,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    is_static: Callable[[Any], bool] | None = None,
) -> Flattened:
    """Flatten an object into a list of leaves and a TreeDef structure.

    Args:
        tree: Any Python object (supports tuple, list, dict, set, Module, or leaves)
        is_leaf: Optional predicate to treat certain objects as leaves.
        is_static: Optional predicate to treat certain objects as static.
            Static objects are stored in the treedef and not included in leaves.

    Returns:
        A tuple of (leaves, treedef) where leaves is a list and treedef captures structure
    """
    # Check for static_value wrapper first (unwrap and store inner value in treedef)
    if isinstance(tree, static_value):
        return Flattened((), freeze_as_treedef(tree.value))
    if is_static is not None and is_static(tree):
        return Flattened((), freeze_as_treedef(tree))

    if is_leaf is not None and is_leaf(tree):
        return Flattened((tree,), Leaf())

    destructure = get_destructure(tree)
    if destructure is None:
        # No destructuring function has been registered; return the object as a leaf.
        return Flattened((tree,), Leaf())

    children_iter, make_treedef = destructure(tree)
    leaves = []
    child_defs = []
    for child in children_iter:
        child_leaves, child_def = flatten(child, is_leaf=is_leaf, is_static=is_static)
        leaves.extend(child_leaves)
        child_defs.append(child_def)
    return Flattened(
        tuple(leaves),
        make_treedef(tuple(child_defs)),
    )


def map(
    fn: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Apply a function to all leaves of one or more trees.

    Args:
        fn: Function to apply to each leaf (or corresponding leaves if multiple trees)
        tree: A tree-like object.
        *rest: More tree-like objects with the same structure as tree.
        is_leaf: Optional predicate to treat certain objects as leaves.

    Returns:
        A new object with the same structure but with fn applied to each leaf
    """
    leaves_first_tree, treedef = flatten(tree, is_leaf=is_leaf)
    all_leaves = [leaves_first_tree]
    for other_tree in rest:
        # Assume the treedef is the same for all trees.
        leaves_other_tree, _ = flatten(other_tree, is_leaf=is_leaf)
        all_leaves.append(leaves_other_tree)

    # Call fn on each leaf, or with multiple leaves if multiple trees are given.
    # We use zip(*all_leaves) to convert all_leaves into a sequence of:
    # [(first_leaf_0, first_leaf_1, ...),
    #  (second_leaf_0, second_leaf_1, ...),
    #  ...etc]
    mapped_leaves = (fn(*leaf_tuple) for leaf_tuple in zip(*all_leaves))
    return treedef.unflatten(mapped_leaves)


def reduce(
    fn: Callable[[Any, Any], Any],
    tree: Any,
    initial: Any = None,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Reduce all leaves of a tree to a single value.

    Args:
        fn: Binary function to combine leaves (accumulator, leaf) -> new_accumulator
        tree: Any Python object (supports tuple, list, dict, set, Module, or leaves)
        initial: Optional initial value. If not provided, uses the first leaf.
        is_leaf: Optional predicate to treat certain objects as leaves.

    Returns:
        The result of reducing all leaves with fn
    """
    leaves, _ = flatten(tree, is_leaf=is_leaf)
    if initial is None:
        return functools.reduce(fn, leaves)
    return functools.reduce(fn, leaves, initial)


def freeze_as_treedef(obj: Any) -> TreeDef | StaticLeaf:
    """Convert an object to a frozen TreeDef representation.

    Recursively converts containers (list, dict, tuple, set, Module) to their
    TreeDef equivalents. Leaf values become StaticLeaf nodes.

    The result is hashable only if all leaf values are hashable.
    """
    destructure = get_destructure(obj)
    if destructure is None:
        return StaticLeaf(obj)

    children_iter, make_treedef = destructure(obj)
    child_defs = tuple(freeze_as_treedef(child) for child in children_iter)
    return make_treedef(child_defs)


__all__ = [
    "registry",
    "flatten",
    "freeze_as_treedef",
    "map",
    "reduce",
    "static_value",
]

if __name__ == "__main__":
    import doctest

    doctest.testmod()
