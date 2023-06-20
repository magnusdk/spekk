from functools import reduce
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from spekk.trees.core import remove, set, update
from spekk.trees.registry import Tree, treedef


class TreeLens:
    """A functional interface to a Tree.

    A lens is an object in functional programming (FP) that allows you to access and
    modify a value in a nested structure in an immutable way.
    """

    def __init__(self, tree: Optional[Tree] = None, **kwargs: Tree):
        if not (bool(tree) ^ bool(kwargs)):
            raise ValueError("Must specify either tree or kwargs (not both).")
        self.tree = tree or kwargs

    def __getitem__(self, path: Union[Any, Tuple[Any]]) -> Tree:
        """Get the value or subtree at the given path.

        If you want to index by a single tuple, e.g. you have a dict with tuples as
        keys, then you should use get(…) instead. This is because there is no way of
        distinguishing between a single tuple argument and a tuple of multiple arguments
        passed to __getitem__(…).
        >>> key = ("a", "tuple", "key")
        >>> tree = TreeLens({key: 1, "a": {"tuple": {"key": "nested_value"}}})
        >>> tree[key]   # This will incorrectly return the nested value
        'nested_value'
        >>> tree[key,]  # Adding a comma helps Python distinguish between the two cases
        1
        >>> tree.get([key])  # get(…) works as expected
        1
        """
        if not isinstance(path, tuple):
            # NOTE: If you actually want to index by a tuple, e.g. you have a dict with
            # tuples as keys, use get() instead
            path = (path,)
        return self.get(path)

    def get(self, path: Sequence[Any]) -> Tree:
        """Get the value or subtree at the given path.

        >>> tree = TreeLens(a={"b": [1, 2, 3]}, d=[3])
        >>> tree.get(["a", "b"])
        [1, 2, 3]
        >>> tree.get(["a", "b", 1])
        2
        """
        return reduce(lambda tree, k: tree[k], path, self.tree)

    def has_subtree(self, path: Sequence[Any]) -> bool:
        """Return True if the given path exists in the tree.

        >>> tree = TreeLens(a={"b": [1, 2, 3]})
        >>> tree.has_subtree(["a", "b", 1])
        True
        >>> tree.has_subtree(["a", "c"])
        False
        """
        try:
            self.get(path)
            return True
        except (KeyError, TypeError):
            return False

    def set(self, value: Any, path: Sequence[Any]) -> "TreeLens":
        """Set the value or subtree at the given path.

        >>> tree = TreeLens(a={"b": [1, 2, 3]}, d=[3])
        >>> tree.set(5, ["a", "b", 1])
        TreeLens({'a': {'b': [1, 5, 3]}, 'd': [3]})
        """
        return self.copy_with(set(self.tree, value, path))

    def update_subtree(self, f: Callable, path: Sequence[Any]) -> "TreeLens":
        """Update the value or subtree at the given path.

        >>> tree = TreeLens(a={"b": [1, 2, 3]}, d=[3])
        >>> tree.update_subtree(lambda x: x + 10, ["a", "b", 1])
        TreeLens({'a': {'b': [1, 12, 3]}, 'd': [3]})
        """
        return self.copy_with(update(self.tree, f, path))

    def remove_subtree(self, path: Sequence[Any]) -> "TreeLens":
        """Remove the value or subtree at the given path.

        >>> tree = TreeLens(a={"b": [1, 2, 3]}, d=[3])
        >>> tree.remove_subtree(["a", "b", 1])
        TreeLens({'a': {'b': [1, 3]}, 'd': [3]})
        """
        return self.copy_with(remove(self.tree, path))

    def copy_with(self, new_tree: Tree) -> "TreeLens":
        """Return a copy of this TreeLens with the given tree.

        This is useful for subclasses that want to use set(…), update_subtree(…), or
        remove_subtree(…). If copy_with is not overriden, these methods will return an
        object of type TreeLens, not of the subclass."""
        return TreeLens(new_tree)

    def is_leaf(self, tree: Tree) -> bool:
        """Return True if the given tree is a leaf.

        By default, this is True if the tree has not been registered with the treedef
        registry, but should be overridden for more specialized trees."""
        try:
            treedef(tree)
            return False
        except ValueError:
            return True

    def __repr__(self):
        return f"TreeLens({self.tree})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
