from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence, Union

Tree = Union[Mapping[Any, "Tree"], Sequence["Tree"], Any]


class TreeDef(ABC):
    """A TreeDef is an abstraction for a tree-like data structure.

    We must be able to get the subtrees of the tree (keys and get) and create a copy
    with updated values (create). Anything that can support these operations can be used
    as a tree.
    """

    def __init__(self, tree: Tree):
        self.tree = tree

    @abstractmethod
    def keys(self) -> Sequence:
        "Get the keys that can be used to get each subtree."
        ...

    @abstractmethod
    def get(self, key: Any):
        "Get the subtree at the given key."
        ...

    @abstractmethod
    def create(self, keys: Sequence, values: Sequence) -> Tree:
        """Create a new instance of the tree with the given keys and values.

        >>> td = treedef({"a": 1, "b": 2})
        >>> td.create(["a", "b"], [3, 4])
        {'a': 3, 'b': 4}
        """
        ...

    @staticmethod
    def new_class(
        keys_fn: Callable[[Tree], Sequence],
        get_fn: Callable[[Tree, Any], Any],
        create_fn: Callable[[Sequence, Sequence], Tree],
    ) -> "TreeDef":
        "Helper function for creating a new TreeDef class."

        class _TreeDef(TreeDef):
            def keys(self) -> Sequence:
                return keys_fn(self.tree)

            def get(self, key: Any):
                return get_fn(self.tree, key)

            def create(self, keys: Sequence, values: Sequence) -> Tree:
                return create_fn(keys, values)

        return _TreeDef


# A registry of types to TreeDef's.
type_registry = {}


def dispatch_by_type(tree: Tree):
    "Given a tree, return the TreeDef for its type (through the type_registry)."
    t = type(tree)
    if t in type_registry:
        return type_registry[type(tree)](tree)


# A registry of functions that can be used to get a TreeDef for a given tree (just a
# list of functions that are tried in order until one of them returns a TreeDef).
dispatch_fn_registry = [dispatch_by_type]


def register_type(t: type, treedef: TreeDef):
    "Register a new TreeDef by type."
    type_registry[t] = treedef


def register_dispatch_fn(dispatch_fn: Callable[[Tree], Union[TreeDef, None]]):
    """Register a new TreeDef by dispatch function.

    Multiple dispatch functions may be registered, in which case they will be tried in
    order until one of them returns a TreeDef. Note that dispatch_by_type always comes
    first and takes precendence."""
    dispatch_fn_registry.append(dispatch_fn)


def treedef(tree: Tree) -> TreeDef:
    """Return the TreeDef (if registered) for the given tree. Dicts, lists, and tuples
    are registered by default."""
    for dispatch_fn in dispatch_fn_registry:
        td = dispatch_fn(tree)
        if td:
            return td
    raise ValueError(
        f"No TreeDef found for {repr(tree)}. Perhaps you need to register one?"
    )


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

if __name__ == "__main__":
    import doctest

    doctest.testmod()
