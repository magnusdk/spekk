"""Module for abstracting over tree-like data structures; in essence, everything that 
is tree-like can be represented as a mapping of keys and values."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence, Union

Tree = Union[Mapping[Any, "Tree"], Sequence["Tree"], Any]


class TreeDef(ABC):
    """A :class:`TreeDef` is an abstraction for a tree-like data structure.

    We must be able to get the subtrees of the tree (keys and get) and create a copy
    with updated values (create). Anything that can support these operations can be used
    as a tree.
    """

    def __init__(self, tree: Tree):
        self.tree = tree

    @abstractmethod
    def keys(self) -> Sequence:
        "Get the keys that can be used to get each subtree."

    @abstractmethod
    def get(self, key: Any):
        "Get the subtree at the given key."

    def values(self) -> Sequence:
        "Get the subtrees for each key in the treedef."
        return [self.get(k) for k in self.keys()]

    @abstractmethod
    def create(self, keys: Sequence, values: Sequence) -> Tree:
        """Create a new instance of the tree with the given keys and values.

        >>> td = treedef({"a": 1, "b": 2})
        >>> td.create(["a", "b"], [3, 4])
        {'a': 3, 'b': 4}
        """

    @staticmethod
    def new_class(
        keys_fn: Callable[[Tree], Sequence],
        get_fn: Callable[[Tree, Any], Any],
        create_fn: Callable[[Sequence, Sequence], Tree],
    ) -> "TreeDef":
        "Helper function for creating a new :class:`TreeDef` class."

        class _TreeDef(TreeDef):
            def keys(self) -> Sequence:
                return keys_fn(self.tree)

            def get(self, key: Any):
                return get_fn(self.tree, key)

            def create(self, keys: Sequence, values: Sequence) -> Tree:
                return create_fn(keys, values)

        return _TreeDef

    def items(self):
        return zip(self.keys(), self.values())


class DuckTypedTreeDef(TreeDef):
    """An object can be a :class:`TreeDef` if it has the dunder-methods:
    ``__spekk_treedef_keys__``, ``__spekk_treedef_get__``, and
    ``__spekk_treedef_create__``."""

    def __init__(self, obj: Any):
        if not (
            hasattr(obj, "__spekk_treedef_keys__")
            and hasattr(obj, "__spekk_treedef_get__")
            and hasattr(obj, "__spekk_treedef_create__")
        ):
            raise ValueError(
                f"Object with type {obj.__class__} does not have the required "
                "dunder-methods to be a treedef."
            )
        self.obj = obj

    def keys(self) -> Sequence:
        return getattr(self.obj, "__spekk_treedef_keys__")()

    def get(self, key: Any):
        return getattr(self.obj, "__spekk_treedef_get__")(key)

    def create(self, keys: Sequence, values: Sequence) -> Any:
        return getattr(self.obj, "__spekk_treedef_create__")(keys, values)


# A registry of types to TreeDef's.
type_registry = {}


def dispatch_treedef(tree: Tree):
    "If the tree itself is a :class:`TreeDef`, return it."
    if isinstance(tree, TreeDef):
        return tree


def dispatch_by_type(tree: Tree):
    """Given a tree, return the :class:`TreeDef` for its type (through the
    ``type_registry``)."""
    t = type(tree)
    if t in type_registry:
        return type_registry[type(tree)](tree)


def dispatch_by_duck_type(tree: Tree):
    """Given a tree, return a :class:`TreeDef` if it has the required dunder-methods.
    See :class:`DuckTypedTreeDef` for more details."""
    try:
        return DuckTypedTreeDef(tree)
    except ValueError:
        return None


# A registry of functions that can be used to get a TreeDef for a given tree (just a
# list of functions that are tried in order until one of them returns a TreeDef).
dispatch_fn_registry = [dispatch_treedef, dispatch_by_type, dispatch_by_duck_type]


def register_type(t: type, treedef: TreeDef):
    "Register a new :class:`TreeDef` by type."
    type_registry[t] = treedef


def register_dispatch_fn(dispatch_fn: Callable[[Tree], Union[TreeDef, None]]):
    """Register a new :class:`TreeDef` by dispatch function.

    Multiple dispatch functions may be registered, in which case they will be tried in
    order until one of them returns a :class:`TreeDef`. Note that
    :func:`dispatch_by_type` and :func:`dispatch_treedef` always comes first and takes
    precendence."""
    dispatch_fn_registry.append(dispatch_fn)


def treedef(tree: Tree) -> TreeDef:
    """Return the :class:`TreeDef` (if registered) for the given tree (``dict``,
    ``list``, and ``tuple`` are registered by default)."""
    for dispatch_fn in dispatch_fn_registry:
        td = dispatch_fn(tree)
        if td:
            return td
    raise ValueError(
        f"No TreeDef found for object with type {tree.__class__}. Perhaps you need to "
        "register one?"
    )


def has_treedef(tree: Tree) -> bool:
    """Return ``True`` if a :class:`TreeDef` is registered for the given tree."""
    try:
        treedef(tree)
        return True
    except ValueError:
        return False


# Register some basic tree types
register_type(
    dict,
    TreeDef.new_class(
        lambda d: d.keys(),
        lambda d, k: d[k],
        lambda keys, values: {k: v for k, v in zip(keys, values)},
    ),
)
register_type(
    list,
    TreeDef.new_class(
        lambda l: range(len(l)),
        lambda l, i: l[i],
        lambda keys, values: list(values),
    ),
)
register_type(
    tuple,
    TreeDef.new_class(
        lambda t: range(len(t)),
        lambda t, i: t[i],
        lambda keys, values: tuple(values),
    ),
)


try:
    import fastmath

    @register_dispatch_fn
    def dispatch_fastmath_module(obj):
        if isinstance(obj, fastmath.Module):
            tree = fastmath.Tree(obj)
            return TreeDef.new_class(
                fastmath.Tree.keys,
                fastmath.Tree.get,
                tree.recreate,
            )(tree)

except ImportError:
    pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()
