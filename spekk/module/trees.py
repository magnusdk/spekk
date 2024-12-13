import dataclasses
import functools
from typing import Any, Callable, Iterable, List, Optional, Sequence, TypeVar, Union

from spekk.module.base import Module

_RAISE_KEY_ERROR = object()
TreeLike = TypeVar("TreeLike")


@dataclasses.dataclass
class TreeLensBranchInfo:
    "Info about the path taken from the original lensed-over tree to the tree-lens."

    parent: "TreeLens"
    key: Any


@dataclasses.dataclass
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
        value: TreeLike,
        *,
        keys_fn: Optional[Callable[[], Iterable[Any]]] = None,
        children_fn: Optional[Callable[[], Iterable[Any]]] = None,
        recreate_fn: Optional[
            Callable[[Sequence[Any], Sequence[Any]], TreeLike]
        ] = None,
        is_marked_as_static: Optional[Callable[[Any], bool]] = None,
    ):
        self.value = value
        if any(
            fn is None
            for fn in [keys_fn, children_fn, recreate_fn, is_marked_as_static]
        ):
            if not all(
                fn is None
                for fn in [keys_fn, children_fn, recreate_fn, is_marked_as_static]
            ):
                raise ValueError(
                    "Either all fns must be provided or none of them "
                    "(in which case they will be inferred)."
                )
            tree = Tree.infer(value)
            keys_fn = tree._keys_fn
            children_fn = tree._children_fn
            recreate_fn = tree._recreate_fn
            is_marked_as_static = tree._is_marked_as_static
        self._keys_fn = keys_fn
        self._children_fn = children_fn
        self._recreate_fn = recreate_fn
        self._is_marked_as_static = is_marked_as_static

    @_inferring_staticmethod
    def keys(tree: "Tree") -> Sequence[Any]:
        return tree._keys_fn()

    @_inferring_staticmethod
    def children(tree: "Tree") -> Sequence[Any]:
        return tree._children_fn()

    def is_leaf(obj: TreeLike) -> bool:
        """The object is a leaf if it can't be inferred as a Tree."""
        try:
            Tree.infer(obj)
            return False
        except NotTreeLikeError:
            return True

    @_inferring_staticmethod
    def recreate(
        tree: "Tree", keys: Sequence[Any], children: Sequence[Any]
    ) -> TreeLike:
        "Recreate the object with the given keys and children."
        return tree._recreate_fn(keys, children)

    @_inferring_staticmethod
    def is_marked_as_static(tree: "Tree", key: Any) -> bool:
        """Return True if the key of the tree is marked as static. This is mostly
        relevant for Modules."""
        return tree._is_marked_as_static(key)

    @staticmethod
    def is_tree_like(obj: TreeLike) -> bool:
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

    def get(tree: TreeLike, *path: Any, default: Any = _RAISE_KEY_ERROR) -> Any:
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

    def update(
        tree: TreeLike, key: Any, f: Callable[[Any], Any], *args, **kwargs
    ) -> TreeLike:
        return Tree.set(tree, key, f(Tree.get(tree, key), *args, **kwargs))

    @staticmethod
    def infer(node: TreeLike) -> "Tree":
        from spekk.ops.array_object import array

        if isinstance(node, Tree):
            return node
        # Handle lists, tuples, and dicts
        elif isinstance(node, (list, tuple)):
            keys_fn = lambda: range(len(node))
            children_fn = lambda: node
            recreate_fn = lambda keys, children: node.__class__(
                _recreate_sequence(keys, children)
            )
            return Tree(
                node,
                keys_fn=keys_fn,
                children_fn=children_fn,
                recreate_fn=recreate_fn,
                is_marked_as_static=lambda _: False,
            )
        elif isinstance(node, dict):
            keys_fn = lambda: node.keys()
            children_fn = lambda: node.values()
            recreate_fn = lambda keys, children: dict(zip(keys, children))
            return Tree(
                node,
                keys_fn=keys_fn,
                children_fn=children_fn,
                recreate_fn=recreate_fn,
                is_marked_as_static=lambda _: False,
            )
        # Handle Modules
        elif isinstance(node, Module):

            def is_marked_as_static(key):
                for f in dataclasses.fields(node):
                    if f.name == key:
                        return f.metadata.get("static", False)

            return Tree(
                node,
                keys_fn=lambda: [f.name for f in dataclasses.fields(node)],
                children_fn=lambda: [
                    getattr(node, f.name) for f in dataclasses.fields(node)
                ],
                recreate_fn=lambda keys, children: node.__class__(
                    **dict(zip(keys, children))
                ),
                is_marked_as_static=is_marked_as_static,
            )
        elif isinstance(node, array):
            keys_fn = lambda: ["data", "dims"]
            children_fn = lambda: [node.data, node.dims]
            recreate_fn = lambda keys, children: array(*children)
            is_marked_as_static = lambda key: key == "dims"
            return Tree(
                node,
                keys_fn=keys_fn,
                children_fn=children_fn,
                recreate_fn=recreate_fn,
                is_marked_as_static=is_marked_as_static,
            )
        # Handle duck-typed
        elif (
            hasattr(node, "__spekk_keys__")
            and hasattr(node, "__spekk_children__")
            and hasattr(node, "__spekk_create__")
        ):
            return Tree(
                node,
                keys_fn=getattr(node, "__spekk_keys__"),
                children_fn=getattr(node, "__spekk_children__"),
                recreate_fn=getattr(node, "__spekk_create__"),
                is_marked_as_static=getattr(
                    node, "__spekk_is_marked_as_static_", lambda _: False
                ),
            )
        else:
            raise NotTreeLikeError(
                f"Object with type {type(node)} can not be inferred as a Tree."
            )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Tree):
            return False
        return self.value == other.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"


def traverse(
    f: Callable[..., TreeLike],
    main_tree: TreeLike,
    *other_trees: TreeLike,
    should_stop: Callable[..., bool] = None,
) -> TreeLike:
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


def traverse_iter(
    tree: TreeLike, *other: TreeLike, should_stop: Callable[..., bool] = None
):
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
    f: Callable[..., TreeLike],
    main_tree: TreeLike,
    *other_trees: TreeLike,
    is_leaf: Callable[..., bool],
):
    return traverse(
        lambda x, *others: f(x, *others) if is_leaf(x, *others) else x,
        main_tree,
        *other_trees,
        should_stop=is_leaf,
    )


def traverse_leaves_iter(
    main_tree: TreeLike,
    *other_trees: TreeLike,
    is_leaf: Callable[..., bool],
):
    # The function outputs a tuple if there are other_trees as well
    _is_leaf = lambda xs: is_leaf(*xs) if other_trees else is_leaf
    yield from (
        x
        for x in traverse_iter(main_tree, *other_trees, should_stop=is_leaf)
        if _is_leaf(x)
    )


class _IndexInFlattened:
    def __init__(self, i: int):
        self.i = i

    def __repr__(self):
        return "*"


class _TreeDef:
    def __init__(
        self,
        data: Union[dict, Any],
        recreate: Optional[Callable],
    ):
        self.data = data
        self.recreate = recreate

    def unflatten(self, flattened_obj: list) -> TreeLike:
        if self.recreate is None:
            if isinstance(self.data, _IndexInFlattened):
                return flattened_obj[self.data.i]
            return self.data

        values = []
        for v in self.data.values():
            if isinstance(v, _TreeDef):
                values.append(v.unflatten(flattened_obj))
            elif isinstance(v, _IndexInFlattened):
                values.append(flattened_obj[v.i])
            else:
                values.append(v)
        return self.recreate(self.data.keys(), values)

    def __repr__(self):
        return repr(self.data)


@dataclasses.dataclass
class FlattenedResult:
    dynamic: tuple
    paths: list
    treedef: _TreeDef
    static: list


def is_array_like(x):
    from spekk import ops

    return isinstance(x, (float, int, complex)) or ops.backend._is_backend_array(x)


def is_not_array_like(x):
    return not is_array_like(x)


def flatten(
    obj: TreeLike,
    is_static: Callable[[Any], bool] = is_not_array_like,
    is_tree_like: Callable[[Any], bool] = Tree.is_tree_like,
) -> FlattenedResult:
    dynamic = []
    paths = []
    static = []
    i = 0

    def _flatten(obj: Union[Tree, Any], path: list = []):
        nonlocal i
        try:
            obj = Tree(obj)
            dummy = {}
            for key, node in zip(obj.keys(), obj.children()):
                if is_tree_like(node):
                    dummy[key] = _flatten(node, path + [key])
                elif obj.is_marked_as_static(key) or is_static(node):
                    static.append(node)
                    dummy[key] = node
                else:
                    dynamic.append(node)
                    paths.append(path + [key])
                    dummy[key] = _IndexInFlattened(i)
                    i += 1

            return _TreeDef(dummy, obj.recreate)
        except NotTreeLikeError:
            # The obj is not a Tree
            paths.append(path)
            if is_static(obj):
                static.append(obj)
                return _TreeDef(obj, None)
            else:
                dynamic.append(obj)
                return _TreeDef(_IndexInFlattened(0), None)

    treedef = _flatten(obj)
    dynamic = tuple(dynamic)
    return FlattenedResult(dynamic, paths, treedef, static)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
