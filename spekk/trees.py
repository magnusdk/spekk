import functools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

_RAISE_KEY_ERROR = object()
T = TypeVar("T")
V = TypeVar("V")


@dataclass
class TreeLensBranchInfo:
    "Info about the path taken from the original lensed-over tree to the tree-lens."

    parent: "TreeLens"
    key: Any


@dataclass
class TreeLens:
    """A functional lens into a Tree. See Tree.at method. Updating a TreeLens returns a
    modified Tree with the same structure as the lensed-over tree."""

    tree: "Tree"  # The lensed-over tree
    branch_info: Optional[TreeLensBranchInfo] = None

    def at(self, key, *remaining: Any) -> "TreeLens":
        sub = self.__class__(Tree.get(self.tree, key), TreeLensBranchInfo(self, key))
        return sub.at(*remaining) if remaining else sub

    def get(self) -> Any:
        return Tree.get(self.tree)

    def set(self, value: Any) -> Any:
        if self.branch_info is None:
            return Tree.get(value)
        updated_parent_tree = Tree.set(
            self.branch_info.parent.tree,
            self.branch_info.key,
            value,
        )
        return self.branch_info.parent.set(updated_parent_tree)

    def update(self, f: Callable[[Any], Any], *args, **kwargs) -> Any:
        return self.set(f(self.get(), *args, **kwargs))


def _inferring_staticmethod(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        # Convert object to Tree if the method is called as a staticmethod
        if not isinstance(self, Tree):
            self = Tree.infer(self)
        return f(self, *args, **kwargs)

    return wrapped


def _recreate_sequence(keys: Sequence[int], children: Sequence[Any]) -> Sequence[Any]:
    seq = [None] * len(keys)
    for k, v in zip(keys, children):
        seq[k] = v
    return seq


class NotTreeLikeError(Exception): ...


class Tree:
    """A Tree is a wrapper around any object that can be decomposed into a set of keys
    and values (and a recreate-fn that recreates the object with given keys and values).
    """

    def __init__(
        self,
        value: T,
        *,
        keys_fn: Optional[Callable[[], Iterable[Any]]] = None,
        children_fn: Optional[Callable[[], Iterable[Any]]] = None,
        is_leaf_fn: Optional[Callable[[T], bool]] = None,
        recreate_fn: Optional[Callable[[Sequence[Any], Sequence[Any]], T]] = None,
    ):
        self.value = value
        if any(fn is None for fn in [keys_fn, children_fn, recreate_fn]):
            if not all(fn is None for fn in [keys_fn, children_fn, recreate_fn]):
                raise ValueError(
                    "Either all fns must be provided or none of them "
                    "(in which case they will be inferred)."
                )
            tree = Tree.infer(value)
            keys_fn = tree._keys_fn
            children_fn = tree._children_fn
            is_leaf_fn = tree._is_leaf_fn
            recreate_fn = tree._recreate_fn
        self._keys_fn = keys_fn
        self._children_fn = children_fn
        self._is_leaf_fn = is_leaf_fn
        self._recreate_fn = recreate_fn

    @_inferring_staticmethod
    def keys(tree: "Tree") -> Sequence[Any]:
        return tree._keys_fn()

    @_inferring_staticmethod
    def children(tree: "Tree") -> Sequence[Any]:
        return tree._children_fn()

    def is_leaf(obj: T) -> bool:
        """The object is a leaf if it can't be inferred as a Tree or if its _is_leaf_fn
        returns True."""
        try:
            tree = Tree.infer(obj)
            return tree._is_leaf_fn()(tree.value)
        except NotTreeLikeError:
            return True

    @_inferring_staticmethod
    def recreate(tree: "Tree", keys: Sequence[Any], children: Sequence[Any]) -> T:
        "Recreate the object with the given keys and children."
        return tree._recreate_fn(keys, children)

    @staticmethod
    def is_tree_like(obj: T) -> bool:
        "The object is Tree-like if it can be inferred as a Tree."
        try:
            Tree.infer(obj)
            return True
        except NotTreeLikeError:
            return False

    @_inferring_staticmethod
    def leaves(tree: "Tree") -> List[Any]:
        leaves = []
        traverse(lambda node: leaves.append(node) if Tree.is_leaf(node) else None, tree)
        return leaves

    @_inferring_staticmethod
    def at(tree: "Tree", *path: Any) -> TreeLens:
        """Return a lensed version of this tree with the given path. It lets you modify
        the tree (immutably, by returning a new copy) at a nested path.

        Example 1:
        >>> tree = {"a": [0, {"b": 1}, 2]}  # Some arbitrary tree
        >>> lensed_tree = Tree(tree).at("a", 1, "b")
        >>> assert lensed_tree.get() == 1
        >>> new_tree = lensed_tree.update(lambda x: x*10)
        >>> assert new_tree == {"a": [0, {"b": 10}, 2]}
        """
        return TreeLens(tree).at(*path)

    def get(tree: T, *path: Any, default: Any = _RAISE_KEY_ERROR) -> Any:
        # Base case
        if not path:
            return tree.value if isinstance(tree, Tree) else tree
        # Recursively keep going until base case
        key, *remaining_path = path
        for k, child in zip(Tree.keys(tree), Tree.children(tree)):
            if k == key:
                return Tree.get(child, *remaining_path)
        # Invalid path
        if default is _RAISE_KEY_ERROR:
            raise KeyError(f"Key {key} not found (full path: {path}).")
        return default

    @_inferring_staticmethod
    def set(tree: "Tree", key: Any, value: Any) -> Any:
        "Set the value of the tree at key."
        new_keys_and_children = dict(zip(tree.keys(), tree.children()))
        new_keys_and_children[key] = value
        return tree.recreate(
            new_keys_and_children.keys(), new_keys_and_children.values()
        )

    def update(tree: T, key: Any, f: Callable[[Any], Any], *args, **kwargs) -> T:
        return Tree.set(tree, key, f(Tree.get(tree, key), *args, **kwargs))

    @staticmethod
    def infer(node: T) -> "Tree":
        if isinstance(node, Tree):
            return node
        # Handle lists, tuples, and dicts
        elif isinstance(node, (list, tuple)):
            keys_fn = lambda: range(len(node))
            children_fn = lambda: node
            is_leaf_fn = lambda: lambda value: False
            recreate_fn = lambda keys, children: node.__class__(
                _recreate_sequence(keys, children)
            )
            return Tree(
                node,
                keys_fn=keys_fn,
                children_fn=children_fn,
                is_leaf_fn=is_leaf_fn,
                recreate_fn=recreate_fn,
            )
        elif isinstance(node, dict):
            keys_fn = lambda: node.keys()
            children_fn = lambda: node.values()
            is_leaf_fn = lambda: lambda value: False
            recreate_fn = lambda keys, children: dict(zip(keys, children))
            return Tree(
                node,
                keys_fn=keys_fn,
                children_fn=children_fn,
                is_leaf_fn=is_leaf_fn,
                recreate_fn=recreate_fn,
            )
        # Handle duck-typed Tree objects
        elif (
            hasattr(node, "__spekk_keys__")
            and hasattr(node, "__spekk_children__")
            and hasattr(node, "__spekk_recreate__")
        ):
            return Tree(
                node,
                keys_fn=getattr(node, "__spekk_keys__"),
                children_fn=getattr(node, "__spekk_children__"),
                is_leaf_fn=getattr(
                    node,
                    "__spekk_is_leaf__",
                    lambda: (lambda obj: not Tree.is_tree_like(obj)),
                ),
                recreate_fn=getattr(node, "__spekk_recreate__"),
            )
        else:
            raise NotTreeLikeError(
                f"Object with type {type(node)} can not be inferred as a Tree."
            )

    def flatten(
        tree: T,
        *,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> Tuple[tuple, Callable[[Sequence[Any]], T]]:
        # flattened_tree will be appended to inside add_leaf_to_flattened_tree
        flattened_tree = []

        def add_leaf_to_flattened_tree(obj: Any) -> Any:
            if not Tree.is_tree_like(obj) or (is_leaf is not None and is_leaf(obj)):
                flattened_tree.append(obj)
                # Return the index as it was before appending (subtract 1)
                return len(flattened_tree) - 1
            return obj

        # Traverse the tree, replacing all non-tree-like nodes with the index of its
        # corresponding slot in the flattened_tree list.
        treedef = traverse(add_leaf_to_flattened_tree, tree, should_stop=is_leaf)
        return tuple(flattened_tree), _UnflattenTree(treedef)

    def merge(tree: T, *other_trees: T, by: Callable[[Sequence[V]], V] = list) -> T:
        """Merge the leaves of all given trees. By default, the leaves are combined
        into a list.

        Args:
            tree: The "main" tree structure.
            other_trees: The other trees that will merge with the main tree.
            by: A function that decides how leaf nodes are merged together. By default,
                leaves from different trees are merged together as a list.
        Returns:
            A tree with the same structure as the first argument, where each leaf are
            merged across all trees.

        Example:
        >>> import numpy as np
        >>> merged_trees = Tree.merge(
        ...     {"a": 1, "b": 11},
        ...     {"a": 2, "b": 12},
        ...     {"a": 3, "b": 13},
        ...     {"a": 4, "b": 14},
        ... )
        >>> assert merged_trees == {"a": [1, 2, 3, 4], "b": [11, 12, 13, 14]}
        """
        flattened_tree, unflatten = Tree.flatten(tree)
        other_flattened_trees = [Tree.flatten(t)[0] for t in other_trees]
        merged_flattened_trees = [
            by(leaves) for leaves in zip(flattened_tree, *other_flattened_trees)
        ]
        return unflatten(merged_flattened_trees)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Tree):
            return False
        return self.value == other.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"


class _UnflattenTree:
    def __init__(self, treedef):
        self.treedef = treedef

    def __call__(self, flattened_tree: Sequence[Any]):
        """Traverse the tree and replace any int (index) by its corresponding value
        in the flattened_tree list. The required information for this is available in a
        local state variable."""
        return traverse(
            lambda x: flattened_tree[x] if isinstance(x, int) else x,
            self.treedef,
        )

    def __repr__(self):
        class Placeholder:
            def __repr__(self):
                return "*"

        s = repr(
            traverse(
                lambda x: Placeholder() if isinstance(x, int) else x,
                self.treedef,
            )
        )
        return f"_UnflattenTree(treedef={s})"


def traverse(
    f: Callable[..., T],
    main_tree: T,
    *other_trees: T,
    should_stop: Callable[..., bool] = None,
) -> T:
    """Traverse a tree depth-first, left-to-right, optionally along with other Tree
    objects (presumably with similar shape), applying f to each leaf and each node.

    Stops recursively traversing children if a non-tree-like tree is encountered or when
    should_stop is given and return True for the trees.

    Return a new copy of the tree with f applied to each leaf and node."""
    trees = [main_tree, *other_trees]

    # Stop early if at least one tree is not tree_like or if should_stop (when provided)
    tree_values = [Tree.get(other) for other in trees]
    if (
        any(not Tree.is_tree_like(tree) for tree in tree_values)
        or any(Tree.is_leaf(x) for x in tree_values)
        or (should_stop is not None and should_stop(*tree_values))
    ):
        return f(*trees)

    # Ensure trees are trees
    trees = [Tree.infer(other) for other in trees]

    # Traverse children recursively (depth-first)
    traversed_children = [
        traverse(f, *children, should_stop=should_stop)
        for children in zip(*(Tree.children(tree) for tree in trees))
    ]

    # Recreate the original object with traversed (mapped) children
    main_tree, *other_trees = trees
    return f(
        main_tree.recreate(main_tree.keys(), traversed_children),
        *(other.value for other in other_trees),
    )


def traverse_iter(tree: T, *other: T, should_stop: Callable[..., bool] = None):
    """Yield all leaves and nodes by traversing the tree(s) depth-first, left-to-right.

    Stops recursively traversing children if a non-tree-like tree is encountered or when
    should_stop is given and return True for the trees.

    If more than one tree is provided, the trees are traversed together and tuples are
    yielded."""
    all_trees = [tree, *other]

    # Stop early if at least one tree is not tree_like or if should_stop (when provided)
    tree_values = [Tree.get(other) for other in all_trees]
    if (
        any(not Tree.is_tree_like(tree) for tree in tree_values)
        or any(Tree.is_leaf(x) for x in tree_values)
        or (should_stop is not None and should_stop(*tree_values))
    ):
        yield all_trees if other else tree
    else:
        # Ensure trees are trees
        all_trees = [Tree.infer(other) for other in all_trees]

        # Traverse recursively and yield from children
        for children in zip(*(Tree.children(tree) for tree in all_trees)):
            yield from traverse_iter(*children, should_stop=should_stop)

        # Yield the tree nodes themselves
        yield all_trees if other else tree


def traverse_leaves(
    f: Callable[..., T],
    main_tree: T,
    *other_trees: T,
    is_leaf: Callable[..., bool],
):
    return traverse(
        lambda x, *others: f(x, *others) if is_leaf(x, *others) else x,
        main_tree,
        *other_trees,
        should_stop=is_leaf,
    )


def traverse_leaves_iter(
    main_tree: T,
    *other_trees: T,
    is_leaf: Callable[..., bool],
):
    # The function outputs a tuple if there are other_trees as well
    _is_leaf = lambda xs: is_leaf(*xs) if other_trees else is_leaf
    yield from (
        x
        for x in traverse_iter(main_tree, *other_trees, should_stop=is_leaf)
        if _is_leaf(x)
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
