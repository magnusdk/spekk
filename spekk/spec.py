"""Module containing the :class:`Spec` class — the most important component of the 
``spekk`` library."""

from functools import reduce
from typing import Dict, Optional, Sequence, Set, Union

import spekk.trees.core as trees
from spekk.trees import Tree, TreeLens, leaves, register_dispatch_fn, traverse, treedef
from spekk.trees.registry import Tree


def _is_spec_leaf(tree: Optional[Tree]):
    """A Spec-tree is a leaf if it is None or a sequence of strings.

    >>> _is_spec_leaf(None)
    True
    >>> _is_spec_leaf(["a", "b"])
    True

    Anything else is not a leaf, in the following case a list of list of strings:
    >>> _is_spec_leaf([["a"], ["c"]])
    False
    """
    if tree is None:
        return True
    if isinstance(tree, Sequence) and all(isinstance(x, str) for x in tree):
        return True
    if isinstance(tree, Spec):
        return _is_spec_leaf(tree.tree)
    return False


class Spec(TreeLens):
    """In a nested tree of arrays, a Spec describes the dimensions of the arrays. Spec
    is a subclass of :class:`TreeLens` which takes the ``tree`` as an argument when
    constructing an object.

    The tree of a Spec is a nested data-structure consisting of dictionaries and
    sequences, where the leaves are sequences of strings. An example of a Spec is as
    follows:

    >>> spec = Spec({"foo": ["a", "b"], "bar": ["b"]})

    The above ``spec`` describes a dictionary of arrays. As data, it could look
    something like this:

    >>> import numpy as np
    >>> data = {"foo": np.ones([2, 3]), "bar": np.ones([3])}

    Note that the structure of the ``spec`` mirrors the structure of the data, but
    where each array has been replaced with a list of strings, representing the
    dimensions of the arrays. Note also that the second dimension of the ``"foo"``
    array share the same name as the first dimension of the ``"bar"`` array, meaning
    that they are semantically the same dimension. This is better understood with a
    more concrete example:

    >>> spec = Spec({"image":   ["batch", "width", "height", "channels"],
    ...              "caption": ["batch", "tokens"]})

    In the above example, both the ``"image"`` and the ``"caption"`` has the same
    ``"batch"`` dimension so we know that if we loop over the batch-items we must loop
    over both the images and captions.
    """

    def is_leaf(self, tree: Optional[Tree] = "_NOT_GIVEN") -> bool:
        """Return True if this spec object represents the dimensions of an array
        (i.e.: not a nested data-structure of arrays).

        May optionally be called as a static method where ``self`` is a tree, or with
        an explicitly given tree.

        See also:
            func:`._is_spec_leaf`).
        """
        if tree == "_NOT_GIVEN":  # `None` has a semantic meaning
            if isinstance(self, Spec):
                return _is_spec_leaf(self.tree)
            else:
                # is_leaf was called as a static method
                return _is_spec_leaf(self)
        else:
            return _is_spec_leaf(tree)

    def remove_dimension(
        self,
        dimension: Union[str, Sequence[str]],
        path: Sequence = (),
    ) -> "Spec":
        """Remove the given dimension from everywhere in the spec.

        >>> spec = Spec({"signal": ["transmits", "receivers"],
        ...              "receiver": {"position": ["receivers"], "direction": []}})
        >>> spec.remove_dimension("receivers")
        Spec({'signal': ['transmits'], 'receiver': {'position': [], 'direction': []}})

        You can also remove multiple dimensions at once:

        >>> spec.remove_dimension(["transmits", "receivers"])
        Spec({'signal': [], 'receiver': {'position': [], 'direction': []}})
        """
        state = self.get(path)

        if isinstance(dimension, (list, tuple, set)):
            for dim in dimension:
                state = state.remove_dimension(dim, path)
            return state

        for leaf in leaves(state.tree, self.is_leaf):
            if dimension in leaf.value:
                state = state.set([x for x in leaf.value if x != dimension], leaf.path)
        return state

    def index_for(self, dimension: str, path: Sequence = ()) -> Tree:
        """Return the indices of the given dimension in the spec with the same
        structure as the spec.

        >>> spec = Spec({"signal": ["transmits", "receivers"],
        ...              "receiver": {"position": ["receivers"], "direction": []}})
        >>> spec.index_for("receivers")
        {'signal': 1, 'receiver': {'position': 0, 'direction': None}}
        """
        state = self.get(path).tree
        for leaf in leaves(state, self.is_leaf):
            index = (
                leaf.value.index(dimension)
                if (leaf.value is not None and dimension in leaf.value)
                else None
            )
            state = trees.set(state, index, leaf.path)
        return state

    @property
    def dimensions(self) -> Set[str]:
        """Return all dimensions in the spec.

        >>> spec = Spec({"signal": ["transmits", "receivers"],
        ...              "receiver": {"position": ["receivers"], "direction": []},
        ...              "point_position": ["transmits", "points"]})
        >>> sorted(spec.dimensions)
        ['points', 'receivers', 'transmits']
        """
        return reduce(
            lambda dims, leaf: dims.union(leaf.value if leaf.value is not None else []),
            leaves(self.tree, self.is_leaf),
            set(),
        )

    def has_dimension(self, *dimensions: str) -> bool:
        """Return True if the spec has the given dimension(s).

        >>> spec = Spec({"signal": ["transmits", "receivers"],
        ...              "receiver": {"position": ["receivers"], "direction": []}})
        >>> spec.has_dimension("transmits", "receivers")
        True
        >>> spec.has_dimension("frames", "transmits", "receivers")
        False
        """
        return all(dim in self.dimensions for dim in dimensions)

    def add_dimension(
        self, dimension: str, path: Sequence = (), index: int = 0
    ) -> "Spec":
        """Add the dimension to the list of dimensions at the specified path and at the
        specified index in the list.

        >>> spec = Spec({"foo": {"baz": ["a", "b"]}, "bar": ["b"]})
        >>> spec.add_dimension("c", ["foo", "baz"], 0)
        Spec({'foo': {'baz': ['c', 'a', 'b']}, 'bar': ['b']})
        >>> spec.add_dimension("c", ["foo", "baz"], 1)
        Spec({'foo': {'baz': ['a', 'c', 'b']}, 'bar': ['b']})
        >>> spec.add_dimension("c", ["bar"], 0)
        Spec({'foo': {'baz': ['a', 'b']}, 'bar': ['c', 'b']})
        """
        current_dims = self.get(path)
        current_dims = current_dims.tree if current_dims is not None else []
        if not self.is_leaf(current_dims):
            raise ValueError(
                f"The provided path does not lead to a dimensions definition. \
Dimensions must be a list of strings, but got {current_dims} at the path {path}."
            )
        new_dims = [*current_dims[:index], dimension, *current_dims[index:]]
        return self.set(new_dims, path)

    def replace(self, replacements: Tree) -> "Spec":
        """Update the spec by replacing subtrees with corresponding subtrees in the
        replacements tree.

        * A value of None in the replacements tree removes the subtree at the
          corresponding path.
        * A leaf (list of dimensions, see Spec.is_leaf) in the replacements tree always
          replaces the leaf (or subtree) at the corresponding path.
        * Keys present in the replacements tree but not in the spec are added to the
          spec at the corresponding path.

        >>> spec = Spec({"foo": {"baz": ["a", "b"]}, "bar": ["b"]})

        Replacing a path with None removes the subtree at that path:

        >>> spec.replace({"foo": None})
        Spec({'bar': ['b']})

        Removing a subtree such that its parent becomes an empty collection also
        removes the parent:

        >>> spec.replace({"foo": {"baz": None}})
        Spec({'bar': ['b']})

        Replacing an existing path with a list of dimensions overwrites the path:

        >>> spec.replace({"foo": ["c"]})
        Spec({'foo': ['c'], 'bar': ['b']})

        Other than that, it is assumed that the ``replacements`` tree structure mirrors
        the spec structure:

        >>> spec.replace({"foo": {"baz": ["c"]}})
        Spec({'foo': {'baz': ['c']}, 'bar': ['b']})
        """
        state = self.tree
        for replacement in traverse(replacements, self.is_leaf):
            if replacement.value is None:
                state = trees.remove(state, replacement.path)
            elif replacement.is_leaf or self.is_leaf(
                trees.get(state, replacement.path, None)
            ):
                state = trees.set(state, replacement.value, replacement.path)
            else:
                replacement_value = trees.filter(
                    replacement.value,
                    self.is_leaf,
                    lambda tree: tree is not None,
                )
                state = trees.update(
                    state,
                    # Current value takes presedence over replacement value in order to
                    # preserve replace semantics.
                    lambda current_value: trees.merge(
                        replacement_value, current_value, "last"
                    ),
                    replacement.path,
                )
        return Spec(state).prune_empty_branches()

    def validate(self, data: Tree):
        """Validate that the data conforms to the spec, raising a
        :class:`~spekk.validation.ValidationError` if not.

        See also:
            :func:`~spekk.validation.validate`
        """
        from spekk.util.validation import validate

        validate(self, data)

    def size(
        self,
        data: Tree,
        dimension: Optional[str] = None,
    ) -> Union[int, Dict[str, int]]:
        """Get the size of dimensions (or a single dimension) in the data.

        >>> import numpy as np
        >>> spec = Spec({"signal": ["transmits", "receivers"],
        ...              "receiver": {"position": ["receivers"], "direction": []}})
        >>> data = {"signal": np.random.randn(10, 20),
        ...         "receiver": {"position": np.random.randn(20, 3),
        ...                      "direction": np.random.randn(20, 3)}}
        >>> spec.size(data) == {'transmits': 10, 'receivers': 20}
        True
        >>> spec.size(data, "transmits")
        10
        >>> spec.size(data, "receivers")
        20
        """
        from spekk import util

        if dimension is None:
            return {dim: self.size(data, dim) for dim in self.dimensions}
        if not self.has_dimension(dimension):
            raise ValueError(f"Spec does not contain the dimension {dimension}.")

        indices_tree = self.index_for(dimension)
        for leaf in leaves(indices_tree, lambda x: isinstance(x, int) or x is None):
            if leaf.value is not None and trees.has_path(data, leaf.path):
                # Assume that all data with the same dimension has the same size, so we
                # just return the first one we find.
                return util.shape(trees.get(data, leaf.path))[leaf.value]

    def __fastmath_keys__(self):
        return trees.treedef(self.tree).keys()

    def __fastmath_children__(self):
        return trees.treedef(self.tree).values()

    def __fastmath_create__(self, keys: Sequence, children: Sequence):
        return Spec(trees.treedef(self.tree).create(keys, children))

    def __hash__(self):
        return hash(self.tree)

    def __eq__(self, other) -> bool:
        """Return True if the specs are equal.

        >>> spec = Spec({"signal": ["transmits", "receivers"]})
        >>> spec == Spec({"signal": ["transmits", "receivers"]})
        True
        >>> spec == Spec({"signal": ["frames", "transmits", "receivers"]})
        False
        >>> spec == Spec({"signal": ["transmits", "receivers"], "foo": ["bar"]})
        False
        >>> spec == Spec({"foo": ["bar"]})
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
                        if isinstance(d2, Spec):
                            d2 = d2.tree
                        if d1 != d2:
                            return False
                else:
                    if (
                        treedef(self.get(subtree.path)).keys()
                        != treedef(subtree.value).keys()
                    ):
                        return False
            return True

    def __len__(self):
        return len(self.tree)

    def __repr__(self):
        if self.tree is None:
            return "Spec()"
        return f"Spec({self.tree})"


register_dispatch_fn(lambda t: treedef(t.tree) if isinstance(t, Spec) else None)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
