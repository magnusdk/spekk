import dataclasses
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator

Destructure = Callable[
    [Any],
    tuple[
        Iterable,  # Children
        Callable[[tuple], "TreeDef"],  # Create the TreeDef from destructured children
    ],
]

# Registry: list of (predicate, destructure) tuples
_registry: list[tuple[Callable[[Any], bool], Destructure]] = []


def register(predicate: Callable[[Any], bool], destructure: Destructure):
    """Register a type for flattening."""
    _registry.append((predicate, destructure))


def get_destructure(obj: Any) -> Destructure | None:
    """Get the destructure function for an object, or None if it's a leaf."""
    for predicate, destructure in _registry:
        if predicate(obj):
            return destructure
    return None


@dataclasses.dataclass(frozen=True)
class Leaf:
    """Represents a leaf node (non-container value)."""

    @property
    def num_leaves(self) -> int:
        return 1

    @property
    def static_values(self) -> tuple:
        return ()

    def unflatten(self, leaves: Iterable = ()) -> Any:
        """Reconstruct a leaf."""
        leaves_list = list(leaves)
        if len(leaves_list) != 1:
            raise ValueError(f"Expected 1 leaf, got {len(leaves_list)} for {self!r}.")
        return leaves_list[0]

    def __repr__(self) -> str:
        return "*"


@dataclasses.dataclass(frozen=True)
class StaticLeaf:
    """Represents a leaf that was marked as static."""

    value: Any

    @property
    def num_leaves(self) -> int:
        return 0

    @property
    def static_values(self) -> tuple:
        return (self.value,)

    def unflatten(self, leaves: Iterable = ()) -> Any:
        """Reconstruct a static leaf."""
        leaves_list = list(leaves)
        if len(leaves_list) != 0:
            raise ValueError(f"Expected 0 leaves, got {len(leaves_list)} for {self!r}.")
        return self.value

    def __repr__(self) -> str:
        return f"StaticLeaf({self.value!r})"


@dataclasses.dataclass(frozen=True)
class TreeDef(ABC):
    """Base class for tree structure definitions."""

    children: tuple["TreeDef | Leaf | Any", ...]

    @property
    def num_leaves(self) -> int:
        def count(node: "TreeDef | Leaf | Any") -> int:
            if isinstance(node, Leaf):
                return 1
            if not isinstance(node, TreeDef):
                return 0  # Static value
            return sum(count(child) for child in node.children)

        return count(self)

    @property
    def static_values(self) -> tuple:
        """Return static values from this node and its children."""
        result = []
        for child in self.children:
            if isinstance(child, (Leaf, TreeDef)):
                result.extend(child.static_values)
            else:
                result.append(child)  # Static value
        return tuple(result)

    def unflatten(self, leaves: Iterable = ()) -> Any:
        """Reconstruct an object from leaves."""
        leaves_list = list(leaves)
        if len(leaves_list) != self.num_leaves:
            raise ValueError(
                f"Expected {self.num_leaves} leaves, got {len(leaves_list)} for {self!r}."
            )
        return self._unflatten(iter(leaves_list))

    def _unflatten(self, leaves_iter: Iterator) -> Any:
        """Recursively unflatten using leaves from the iterator."""
        unflattened = []
        for child in self.children:
            if isinstance(child, Leaf):
                unflattened.append(next(leaves_iter))
            elif isinstance(child, StaticLeaf):
                unflattened.append(child.value)
            elif isinstance(child, TreeDef):
                unflattened.append(child._unflatten(leaves_iter))
            else:
                unflattened.append(child)  # Static value
        return self._construct(unflattened)

    @abstractmethod
    def _construct(self, children: list) -> Any:
        """Construct the object from already-unflattened children."""

    def __repr__(self) -> str:
        def child_repr_fn(x: Any) -> str:
            if isinstance(x, TreeDef):
                return x._inner_repr(child_repr_fn)
            elif isinstance(x, StaticLeaf):
                # Keep representation of static values as-is for a cleaner output.
                return repr(x.value)
            return repr(x)

        return f"TreeDef({self._inner_repr(child_repr_fn)})"

    @abstractmethod
    def _inner_repr(self, child_repr_fn: Callable[[Any], str]) -> str:
        """This function takes child_repr_fn as an argument just to make it more
        obvious to potential new implementers that the _inner_repr should not just call
        repr on the children."""


@dataclasses.dataclass(frozen=True, repr=False)
class TupleTreeDef(TreeDef):
    def _construct(self, children: list) -> tuple:
        return tuple(children)

    def _inner_repr(self, child_repr_fn: Callable[[Any], str]) -> str:
        inner = ", ".join(child_repr_fn(c) for c in self.children)
        return f"({inner})"


@dataclasses.dataclass(frozen=True, repr=False)
class ListTreeDef(TreeDef):
    def _construct(self, children: list) -> list:
        return children

    def _inner_repr(self, child_repr_fn: Callable[[Any], str]) -> str:
        inner = ", ".join(child_repr_fn(c) for c in self.children)
        return f"[{inner}]"


@dataclasses.dataclass(frozen=True, repr=False)
class DictTreeDef(TreeDef):
    keys: tuple

    def _construct(self, children: list) -> dict:
        return dict(zip(self.keys, children))

    def _inner_repr(self, child_repr_fn: Callable[[Any], str]) -> str:
        items = ", ".join(
            f"{k!r}: {child_repr_fn(c)}" for k, c in zip(self.keys, self.children)
        )
        return "{" + items + "}"


@dataclasses.dataclass(frozen=True, repr=False)
class SetTreeDef(TreeDef):
    def _construct(self, children: list) -> set:
        return set(children)

    def _inner_repr(self, child_repr_fn: Callable[[Any], str]) -> str:
        inner = ", ".join(child_repr_fn(c) for c in self.children)
        return "{" + inner + "}"


def _destructure_tuple(obj: tuple):
    return obj, TupleTreeDef


def _destructure_list(obj: list):
    return obj, ListTreeDef


def _destructure_dict(obj: dict):
    keys = tuple(obj.keys())
    return obj.values(), functools.partial(DictTreeDef, keys=keys)


def _destructure_set(obj: set):
    return obj, SetTreeDef


# Register built-in types
register(lambda x: type(x) is tuple, _destructure_tuple)
register(lambda x: type(x) is list, _destructure_list)
register(lambda x: type(x) is dict, _destructure_dict)
register(lambda x: type(x) is set, _destructure_set)
