":class:`TreeLens` is a functional interface to a tree-like data structure."

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

from spekk.trees.core import (
    filter,
    has_path,
    remove,
    set,
    traverse,
    update,
    update_leaves,
)
from spekk.trees.registry import Tree, treedef

TSelf = TypeVar("TSelf", bound="TreeLens")


class TreeLens:
    """A functional interface to a Tree.

    A lens is an object in functional programming (FP) that allows you to access and
    modify a value in a nested structure in an immutable way.
    """

    def __init__(self, tree: Tree = ()):
        # Ensure that there are no nested TreeLens objects:
        for t in traverse(tree, self.is_leaf):
            if isinstance(t.value, TreeLens):
                tree = set(tree, t.value.tree, t.path)
        self.tree = tree

    def __getitem__(self: TSelf, path: Union[Any, Tuple[Any]]) -> TSelf:
        """Get the value or subtree at the given path.

        If you want to index by a single tuple, e.g. you have a dict with tuples as
        keys, then you should use get(…) instead. This is because there is no way of
        distinguishing between a single tuple argument and a tuple of multiple arguments
        passed to __getitem__(…).
        >>> key = ("a", "tuple", "key")
        >>> tree = TreeLens({key: 1, "a": {"tuple": {"key": "nested_value"}}})
        >>> tree[key]   # This will incorrectly return the nested value
        TreeLens(nested_value)
        >>> tree[key,]  # Adding a comma helps Python distinguish between the two cases
        TreeLens(1)
        >>> tree.get([key])  # get(…) works as expected
        TreeLens(1)
        """
        if not isinstance(path, tuple):
            # NOTE: If you actually want to index by a tuple, e.g. you have a dict with
            # tuples as keys, use get() instead
            path = (path,)
        return self.get(path)

    def get(self: TSelf, path: Sequence[Any]) -> TSelf:
        """Get the value or subtree at the given path.

        >>> tree = TreeLens({"a": {"b": [1, 2, 3]}, "d": [3]})
        >>> tree.get(["a", "b"])
        TreeLens([1, 2, 3])
        >>> tree.get(["a", "b", 1])
        TreeLens(2)
        """
        tree = self.tree
        for k in path:
            if k not in treedef(tree).keys():
                return self.copy_with(None)  # Return with None tree
            tree = tree[k]
        return self.copy_with(tree)

    def has_subtree(self, path: Sequence[Any]) -> bool:
        """Return True if the given path exists in the tree.

        >>> tree = TreeLens({"a": {"b": [1, 2, 3]}})
        >>> tree.has_subtree(["a", "b", 1])
        True
        >>> tree.has_subtree(["a", "c"])
        False
        """
        return has_path(self.tree, path)

    def set(self: TSelf, value: Any, path: Sequence[Any]) -> TSelf:
        """Set the value or subtree at the given path.

        >>> tree = TreeLens({"a": {"b": [1, 2, 3]}, "d": [3]})
        >>> tree.set(5, ["a", "b", 1])
        TreeLens({'a': {'b': [1, 5, 3]}, 'd': [3]})
        """
        return self.copy_with(set(self.tree, value, path))

    def update_subtree(self: TSelf, f: Callable, path: Sequence[Any]) -> TSelf:
        """Update the value or subtree at the given path.

        >>> tree = TreeLens({"a":{"b": [1, 2, 3]}, "d":[3]})
        >>> tree.update_subtree(lambda x: x + 10, ["a", "b", 1])
        TreeLens({'a': {'b': [1, 12, 3]}, 'd': [3]})
        """
        return self.copy_with(update(self.tree, f, path))

    def update_leaves(self: TSelf, f: Callable, path: Sequence[Any] = ()) -> TSelf:
        "See :func:`~spekk.trees.core.update_leaves`."
        return self.copy_with(update_leaves(self.tree, self.is_leaf, f, path))

    def remove_subtree(self: TSelf, path: Sequence[Any]) -> TSelf:
        """Remove the value or subtree at the given path.

        >>> tree = TreeLens({"a": {"b": [1, 2, 3]}, "d": [3]})
        >>> tree.remove_subtree(["a", "b", 1])
        TreeLens({'a': {'b': [1, 3]}, 'd': [3]})
        """
        return self.copy_with(remove(self.tree, path))

    def is_leaf(self, tree: Tree) -> bool:
        """Return True if the given tree is a leaf.

        By default, this is True if the tree has not been registered with the treedef
        registry, but should be overridden for more specialized trees."""
        try:
            treedef(tree)
            return False
        except ValueError:
            return True

    def prune_empty_branches(
        self: Union[TSelf, Tree],
        is_leaf: Optional[Callable[[Tree], bool]] = None,
    ) -> Union[TSelf, Tree]:
        """Remove all empty subtrees.

        May be called as a static method where self is a Tree."""
        if is_leaf is None:
            is_leaf = self.is_leaf
        tree = self.tree if isinstance(self, TreeLens) else self
        not_empty = lambda tree: is_leaf(tree) or len(tree) > 0
        pruned_tree = filter(tree, is_leaf, not_empty)
        if isinstance(self, TreeLens):
            pruned_tree = self.copy_with(pruned_tree)
        return pruned_tree

    @property
    def at(self) -> "_TreeNavigator":
        """Interface for working on subtrees. This is a convenience method that
        provides the same functionality as set(…) and update_subtree(…), but with a
        (potentially) more readable syntax.

        >>> tree = TreeLens({"a": {"b": [1, 2, 3]}, "d": [3]})

        It can be used to set the value at the given path:
        >>> tree.at["a", "b", 1].set(5)
        TreeLens({'a': {'b': [1, 5, 3]}, 'd': [3]})

        Or update it, given a function:
        >>> tree.at["a", "b", 1].update(lambda x: x+10)
        TreeLens({'a': {'b': [1, 12, 3]}, 'd': [3]})
        """
        return _TreeNavigator(self)

    def __iter__(self):
        return iter([self.copy_with(t) for t in self.tree])

    def copy_with(self: TSelf, tree: Tree) -> TSelf:
        "Return a copy of this object with the given tree."
        return self.__class__(tree)

    def __repr__(self):
        return f"TreeLens({self.tree})"


@dataclass
class _TreeNavigator:
    """Object that can modify a :class:`TreeLens` at a given path.

    See :attr:`TreeLens.at` for more information."""

    tree: TreeLens
    path: tuple = ()

    def set(self, value: Any) -> TreeLens:
        return self.tree.set(value, self.path)

    def update(self, f: Callable, *args, **kwargs) -> TreeLens:
        partial_f = lambda x: f(x, *args, **kwargs)
        return self.tree.update_subtree(partial_f, self.path)

    def __getitem__(self, path) -> "_TreeNavigator":
        "(Further) index the tree at the given path."
        if not isinstance(path, tuple):
            # NOTE: A quirk of Python's __getitem__ causes surprising behavior if users
            # try to index by a tuple. For example, tree.at[(1,2)] will be treated as
            # tree.at[1,2], which is likely not what the user wants. As a workaround,
            # the user should add a comma after the tuple: tree.at[(1,2),]
            path = (path,)
        return _TreeNavigator(self.tree, self.path + path)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
