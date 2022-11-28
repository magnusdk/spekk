from functools import reduce
from typing import Callable, Optional, Sequence, Set

import spekk.trees.core as trees
from spekk.trees import Tree, TreeLens, leaves, traverse, treedef


class Spec(TreeLens):
    """A spec describes the dimensions of each array in a tree of arrays.

    A tree consists of nested dictionaries and/or sequences. The leaves of the tree are
    sequences of dimension names (strings).

    For example, let's say we have a set of images and a set of captions. Each image has
    a width, a height, and 3 channels (RGB), and each caption has a set of tokens. They
    are both stored in batches. This can be specced as follows:
    >>> spec = Spec(image   = ["batch", "width", "height", "channels"],
    ...             caption = ["batch", "tokens"])

    Notice that both the image and the caption have a batch dimension. This dimension
    should be processed simultaneously for both images and captions as it semantically
    describes the same dimension.

    Spec thus allows for describing arbitrary data structures containing multiple
    arrays, with both different and shared dimensions.
    """

    def __init__(self, tree: Optional[Tree] = None, **kwargs: Tree):
        if (tree is not None) & bool(kwargs):
            raise ValueError(
                f"May not specify both a tree and kwargs. Got {tree=} and {kwargs=}."
            )
        self.tree = kwargs or tree

    def is_leaf(self, tree: Tree) -> bool:
        """The leaves of a spec is a list of dimension names."""
        return tree is None or (
            isinstance(tree, Sequence) and all(isinstance(x, str) for x in tree)
        )

    def remove_dimension(self, dimension: str) -> "Spec":
        """Remove the given dimension from the spec, searching recursively.

        >>> spec = Spec(signal=["transmits", "receivers"],
        ...             receiver={"position": ["receivers"], "direction": []})
        >>> spec.remove_dimension("receivers")
        Spec({'signal': ['transmits'], 'receiver': {'position': [], 'direction': []}})
        """
        state = self
        for leaf in leaves(self.tree, self.is_leaf):
            if dimension in leaf.value:
                state = state.set([x for x in leaf.value if x != dimension], leaf.path)
        return state

    def index_for(self, dimension: str, path: tuple = ()) -> Tree:
        """Return the indices of the given dimension in the spec.

        >>> spec = Spec(signal=["transmits", "receivers"],
        ...             receiver={"position": ["receivers"], "direction": []})
        >>> spec.index_for("receivers")
        {'signal': 1, 'receiver': {'position': 0, 'direction': None}}
        """
        state = self.tree
        for leaf in leaves(self.tree, self.is_leaf):
            index = leaf.value.index(dimension) if dimension in leaf.value else None
            state = trees.set(state, index, leaf.path)
        return state

    @property
    def dimensions(self) -> Set[str]:
        """Return all dimensions in the spec.

        >>> spec = Spec(signal=["transmits", "receivers"],
        ...             receiver={"position": ["receivers"], "direction": []},
        ...             point_position=["transmits", "points"])
        >>> sorted(spec.dimensions)
        ['points', 'receivers', 'transmits']
        """
        return reduce(
            lambda dims, leaf: dims.union(leaf.value),
            leaves(self.tree, self.is_leaf),
            set(),
        )

    def has_dimension(self, *dimensions: str) -> bool:
        """Return True if the spec has the given dimension(s).

        >>> spec = Spec(signal=["transmits", "receivers"],
        ...             receiver={"position": ["receivers"], "direction": []})
        >>> spec.has_dimension("transmits", "receivers")
        True
        >>> spec.has_dimension("frames", "transmits", "receivers")
        False
        """
        return all(dim in self.dimensions for dim in dimensions)

    def add_dimension(self, dimension: str, path: tuple = (), index: int = 0) -> "Spec":
        """TODO: Docs and tests"""
        current_dims = self.get(path)
        current_dims = current_dims if current_dims is not None else []
        if not self.is_leaf(current_dims):
            raise ValueError(
                f"The provided path does not lead to a dimensions definition. \
Dimensions must be a list of strings, but got {current_dims} at the path {path}."
            )
        new_dims = (
            tuple(current_dims[:index]) + (dimension,) + tuple(current_dims[index:])
        )
        return self.set(new_dims, path)

    def replace(self, replacements: Tree) -> "Spec":
        """Update the spec by replacing subtrees with corresponding subtrees in the
        replacements tree.

        >>> spec = Spec(signal=["transmits", "receivers"],
        ...             receiver={"position": ["receivers"], "direction": []})
        >>> spec.replace({"receiver": {"direction": ["transmits"]}})
        Spec({'signal': ['transmits', 'receivers'], 'receiver': {'position': ['receivers'], 'direction': ['transmits']}})
        """
        state = self
        for replacement in traverse(replacements, self.is_leaf):
            if replacement.value is None:
                state = state.remove_subtree(replacement.path)
            elif replacement.is_leaf:
                state = state.set(replacement.value, replacement.path)
        return state

    def update_leaves(self, f: Callable[[Sequence[str]], Sequence[str]]) -> "Spec":
        """

        >>> spec = Spec(foo=["a", "b"], bar=["c"])
        >>> spec.update_leaves(lambda dims: dims + ["new_dim"])
        Spec({'foo': ['a', 'b', 'new_dim'], 'bar': ['c', 'new_dim']})
        """
        state = self
        for leaf in leaves(self.tree, self.is_leaf):
            state = state.set(f(leaf.value), leaf.path)
        return state

    def copy_with(self, new_tree: Tree) -> "Spec":
        return Spec(new_tree)

    def __eq__(self, other) -> bool:
        """Return True if the specs are equal.

        >>> spec = Spec(signal=["transmits", "receivers"])
        >>> spec == Spec(signal=["transmits", "receivers"])
        True
        >>> spec == Spec(signal=["frames", "transmits", "receivers"])
        False
        >>> spec == Spec(signal=["transmits", "receivers"], foo=["bar"])
        False
        >>> spec == Spec(foo=["bar"])
        False
        """
        if not isinstance(other, Spec):
            return False
        else:
            for subtree in traverse(other.tree, self.is_leaf):
                if subtree.is_leaf:
                    if not self.has_subtree(subtree.path):
                        return False
                    if len(subtree.value) != len(self.get(subtree.path)):
                        return False
                    for d1, d2 in zip(subtree.value, self.get(subtree.path)):
                        if d1 != d2:
                            return False
                else:
                    if (
                        treedef(self.get(subtree.path)).keys()
                        != treedef(subtree.value).keys()
                    ):
                        return False
            return True

    def __repr__(self):
        if self.tree is None:
            return "Spec()"
        return f"Spec({self.tree})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
