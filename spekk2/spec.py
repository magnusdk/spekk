from functools import reduce
from typing import Callable, Optional, Sequence, Set

from spekk2.trees import Tree, TreeLens, traverse, traverse_with_state, tree_repr
from spekk2.trees.registry import get_keys, get_values


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
        Spec({signal: ['transmits'], receiver: {position: [], direction: []}})
        """
        return Spec(
            traverse(
                self.tree,
                self.is_leaf,
                lambda subtree: (
                    [x for x in subtree if x != dimension]
                    if self.is_leaf(subtree)
                    else subtree
                ),
            )
        )

    def index_for(self, dimension: str, path: tuple = ()) -> Tree:
        """Return the indices of the given dimension in the spec.

        >>> spec = Spec(signal=["transmits", "receivers"],
        ...             receiver={"position": ["receivers"], "direction": []})
        >>> spec.index_for("receivers")
        {'signal': 1, 'receiver': {'position': 0, 'direction': None}}
        """
        return traverse(
            self.get(path),
            self.is_leaf,
            lambda subtree: (
                (subtree.index(dimension) if dimension in subtree else None)
                if self.is_leaf(subtree)
                else subtree
            ),
        )

    @property
    def dimensions(self) -> Set[str]:
        """Return all dimensions in the spec.

        >>> spec = Spec(signal=["transmits", "receivers"],
        ...             receiver={"position": ["receivers"], "direction": []},
        ...             point_position=["transmits", "points"])
        >>> sorted(spec.dimensions)
        ['points', 'receivers', 'transmits']
        """
        return traverse(
            self.tree,
            self.is_leaf,
            lambda subtree: (
                set(subtree)
                if self.is_leaf(subtree)
                else reduce(set.union, get_values(subtree), set())
            ),
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
        Spec({signal: ['transmits', 'receivers'], receiver: {position: ['receivers'], direction: ['transmits']}})
        """

        def f(state: Spec, tree: Tree, path: tuple) -> Spec:
            if tree is None:
                return state.remove_subtree(path), tree
            elif self.is_leaf(tree):
                return state.set(tree, path), tree
            else:
                return state, tree

        state, _ = traverse_with_state(
            replacements, self.is_leaf, f, self, use_path=True
        )
        return state

    def update_leaves(self, f: Callable[[Sequence[str]], Sequence[str]]) -> "Spec":
        state = traverse(
            self.tree,
            self.is_leaf,
            lambda x: f(x) if self.is_leaf(x) else x,
        )
        return Spec(state)

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

            def f(state: bool, tree: Tree, path: tuple) -> bool:
                if self.is_leaf(tree):
                    new_state = (
                        state
                        and self.has_subtree(path)
                        and len(tree) == len(self.get(path))
                        and all(d1 == d2 for d1, d2 in zip(tree, self.get(path)))
                    )
                    return new_state, tree
                else:
                    new_state = state and get_keys(self.get(path)) == get_keys(tree)
                    return new_state, state

            state, _ = traverse_with_state(
                other.tree, self.is_leaf, f, True, use_path=True
            )
            return state

    def __repr__(self):
        if self.tree is None:
            return "Spec()"
        return f"Spec({tree_repr(self.tree, self.is_leaf)})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()