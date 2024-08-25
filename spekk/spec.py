import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

from spekk import common
from spekk.trees import Tree, TreeLens, traverse, traverse_iter


class SpecLens(TreeLens):
    def add_dimension(self, dim: str, index: int = 0) -> "Spec":
        return self.update(Spec.add_dimension, dim, index)

    def remove_dimension(self, *removed_dims: str) -> "Spec":
        return self.update(Spec.remove_dimension, *removed_dims)._prune_empty_branches()


def _inferring_staticmethod(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        # Convert object to Tree if the method is called as a staticmethod
        if not isinstance(self, Spec):
            self = Spec(self)
        return f(self, *args, **kwargs)

    return wrapped


def is_spec_dimensions(obj: Any) -> bool:
    "Spec dimensions is always either a list/tuple of strings or None."
    while isinstance(obj, Spec):
        obj = obj.data
    is_list_of_strings = isinstance(obj, (list, tuple)) and all(
        isinstance(x, str) for x in obj
    )
    return obj is None or is_list_of_strings


def index_spec_dimensions(dims: List[str], dim: str) -> Union[int, None]:
    if dims is None or dim not in dims:
        return None
    return dims.index(dim)


def _unwrap_spec(data: Union["Spec", Any]) -> Any:
    return traverse(
        lambda obj: obj.data if isinstance(obj, Spec) else obj,
        data,
        should_stop=is_spec_dimensions,
    )


class Spec:
    def __init__(self, data: Any):
        self._data = _unwrap_spec(data)

    @property
    def data(self) -> Any:
        return self._data

    @_inferring_staticmethod
    def at(spec: "Spec", *path: Any) -> SpecLens:
        return SpecLens(spec).at(*path)

    @_inferring_staticmethod
    def add_dimension(spec: "Spec", dim: str, index: int = 0) -> "Spec":
        return Spec(
            traverse(
                lambda obj: (
                    common.insert(obj, index, dim) if is_spec_dimensions(obj) else obj
                ),
                spec.data,
            )
        )

    @_inferring_staticmethod
    def remove_dimension(spec: "Spec", *removed_dims: str) -> "Spec":
        return Spec(
            traverse(
                lambda obj: (
                    None
                    if obj is None
                    else [dim for dim in obj if dim not in removed_dims]
                    if is_spec_dimensions(obj)
                    else Spec._prune_empty_branches(obj)
                ),
                spec.data,
                should_stop=is_spec_dimensions,
            )
        )

    @_inferring_staticmethod
    def conform(spec: "Spec", data: Any) -> "Spec":
        """Return a new spec guaranteed to have the same structure as the given data.
        This is especially helpful when we want to flatten spec and data
        simultaneously."""
        # Return spec as-is, if the given data is a leaf (base case).
        if not Tree.is_leaf(data):
            data_keys = Tree.keys(data)

            # Recreate as a dict when data is something other than list or tuple. This
            # is to support arbitrary objects.
            recreation_template = data if isinstance(data, (list, tuple)) else {}

            # Construct a new spec with the same structure as the data.
            if is_spec_dimensions(spec):
                # We just repeat the spec for each key in the data.
                spec_values = [spec] * len(data_keys)
                spec = Tree.recreate(recreation_template, data_keys, spec_values)

            # Recursively conform the values for each sub-tree of the data.
            expanded_spec_values = [
                Spec.conform(
                    Tree.get(spec, key, default=None),
                    Tree.get(data, key),
                )
                for key in data_keys
            ]
            # Recreate spec again with the same structure as the data.
            spec = Tree.recreate(recreation_template, data_keys, expanded_spec_values)
        return Spec(spec)

    @_inferring_staticmethod
    def indices_for(spec: "Spec", dim: str, *, conform: Optional[Any] = None):
        if conform is not None:
            spec = spec.conform(conform)
        return traverse(
            lambda obj: (
                index_spec_dimensions(obj, dim) if is_spec_dimensions(obj) else obj
            ),
            spec.data,
            should_stop=is_spec_dimensions,
        )

    @_inferring_staticmethod
    def index_data(spec: "Spec", data: Any, i: int, dim: str) -> Any:
        def f(data_item: Any, dimensions: Union[Sequence[str], Any]):
            if not is_spec_dimensions(dimensions):
                return data_item
            axis = index_spec_dimensions(dimensions, dim)
            if axis is not None:
                data_item = common.index_at(data_item, i, axis)
            return data_item

        spec = spec.conform(data)
        return traverse(
            f,
            data,
            spec.data,
            should_stop=lambda _, dims: is_spec_dimensions(dims),
        )

    @_inferring_staticmethod
    def size(spec: "Spec", data: Any, *dims: str) -> Union[int, Dict[str, int]]:
        if len(dims) == 1:
            dim = dims[0]
            spec_indices = spec.indices_for(dim, conform=data)
            sizes = set(
                x.shape[index]
                for x, index in traverse_iter(data, spec_indices)
                if index is not None and Tree.is_leaf(x)
            )
            if len(sizes) == 0:
                raise ValueError(f"Dimension '{dim}' not found in the spec: {spec}")
            if len(sizes) > 1:
                raise ValueError(
                    f"Inconsistent sizes found for dimension '{dim}'. Sizes: {sizes}"
                )
            # len(sizes) == 1. Return the only value.
            return list(sizes)[0]
        else:
            if len(dims) == 0:
                dims = spec.dimensions
            return {dim: spec.size(data, dim) for dim in dims}

    @property
    def dimensions(self) -> Set[str]:
        dims = set()
        for x in traverse_iter(self.data, should_stop=is_spec_dimensions):
            if is_spec_dimensions(x) and x is not None:
                dims = dims.union(x)
        return dims

    @_inferring_staticmethod
    def _prune_empty_branches(spec: "Spec") -> "Spec":
        if is_spec_dimensions(spec.data):
            return spec
        pruned_keys, pruned_children = [], []
        for k, v in zip(Tree.keys(spec.data), Tree.children(spec.data)):
            if hasattr(v, "__len__") and len(v) > 0:
                pruned_keys.append(k)
                pruned_children.append(v)
        return Tree.recreate(spec, pruned_keys, pruned_children)

    def __spekk_keys__(self) -> Callable[[], Iterable[Any]]:
        return Tree.keys(self.data)

    def __spekk_children__(self) -> Callable[[], Iterable[Any]]:
        return [Spec(node) for node in Tree.children(self.data)]

    def __spekk_is_leaf__(self) -> Callable[[], Iterable[Any]]:
        return is_spec_dimensions

    def __spekk_recreate__(
        self, keys: Sequence[Any], children: Sequence[Any]
    ) -> "Spec":
        return Spec(Tree.recreate(self.data, keys, children))

    def __iter__(self):
        return iter(self.data)

    def __eq__(self, value: Any) -> bool:
        return isinstance(value, Spec) and value._data == self._data

    def __hash__(self) -> int:
        return hash(self._data)

    def __repr__(self) -> str:
        return f"Spec({self._data})"
