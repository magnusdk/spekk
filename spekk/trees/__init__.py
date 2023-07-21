"""Classes and functions for working with tree-like structures."""

from spekk.trees.core import (
    are_equal,
    get,
    leaves,
    merge,
    remove,
    set,
    traverse,
    update,
    update_leaves,
)
from spekk.trees.registry import (
    Tree,
    TreeDef,
    has_treedef,
    register_dispatch_fn,
    register_type,
    treedef,
)
from spekk.trees.treelens import TreeLens

__all__ = [
    "are_equal",
    "get",
    "leaves",
    "merge",
    "remove",
    "set",
    "traverse",
    "update",
    "update_leaves",
    "Tree",
    "TreeDef",
    "has_treedef",
    "register_dispatch_fn",
    "register_type",
    "treedef",
    "TreeLens",
]
