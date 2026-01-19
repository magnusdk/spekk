import abc
import dataclasses
import functools
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    cast,
)

from typing_extensions import dataclass_transform

if TYPE_CHECKING:
    from spekk.ops._types import Dim
    from spekk.ops.array_object import array


@functools.wraps(dataclasses.field)
def field(*, static: bool = False, **kwargs):
    "Return an object to identify dataclass fields, and optionally mark them as static."
    metadata = dict(kwargs.pop("metadata", {}))  # Copy to new dict
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    if static:
        metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


@dataclasses.dataclass
class _static_value:
    """NOTE: this is an internal class; do not use it. Marker for letting spekk know
    that a value is considered static and should trigger JIT-recompilation when changed.
    """

    value: Any

    def __repr__(self):
        return f"static_value({self.value})"


@dataclass_transform(
    eq_default=False,
    field_specifiers=(dataclasses.field, cast(Callable, field)),
)
class Module(abc.ABC):
    """Base class for custom classes that are understood by spekk.

    Classes that inherit from Module are dataclasses:
    >>> from spekk import ops, Module
    >>> class MyClass(Module):
    ...     foo: float
    ...     bar: ops.array
    ...     quiz: int = field(static=True)

    Fields that are marked as static are filtered out during backend function
    compilation, similar to Equinox.
    """

    def __init_subclass__(cls, **kwargs):
        "Convert all subclasses of Module to dataclass."
        # NOTE: naively passing kwargs to dataclass can break if user passes additional
        # kwargs not meant for the dataclass. The fix is to filter the kwargs here
        # before passing them to the dataclass.
        dataclasses.dataclass(cls, **kwargs)

    @property
    def dim_sizes(self) -> dict["Dim", int | set[int]]:
        """Return a dictionary of the dimensions and corresponding sizes of all arrays
        in self. If a dimension has different sizes for different arrays, a set of
        sizes is returned for that dimension and a warning is created.
        """
        return dim_sizes(self)

    @property
    def at(self):
        """Helper for updating slices of _all_ arrays in the object that have the
        sliced dimension(s)."""
        return at(self)

    def __eq__(self, other):
        if self is other:
            return True
        if self.__class__ is other.__class__:
            from spekk import ops

            for _field in dataclasses.fields(self):
                a = getattr(self, _field.name)
                b = getattr(other, _field.name)
                if isinstance(a, ops.array) and isinstance(b, ops.array):
                    if a._id != b._id:
                        return False
                elif a != b:
                    return False
            return True
        return NotImplemented


####
# Helpers for slicing arrays in Module objects:


def at(obj):
    return _ModuleAtHelper(obj)


def dim_sizes(obj):
    from spekk import ops
    from spekk.ops._types import _UndefinedDim

    undefined_dim_key = _UndefinedDim()

    # Gather the size(s) for each dimension for each array in self. Store the sizes
    # for each dimension in a set.
    dim_size_sets: dict[str | _UndefinedDim, set[int]] = defaultdict(set)

    def add_size(x):
        if isinstance(x, ops.array):
            for dim, size in x.dim_sizes.items():
                if isinstance(dim, _UndefinedDim):
                    dim = undefined_dim_key
                dim_size_sets[dim].add(size)
        return x

    traverse(obj, map_leaf=add_size, map_static_field=add_size)

    # Check if any dimensions has inconsistent sizes. Warn if they do. If not, get
    # the single size for that dimension instead of the set of one element.
    dim_sizes: dict[Dim | _UndefinedDim, int | set[int]] = {}
    for dim, size in dim_size_sets.items():
        if len(size) == 1:
            # Get the single size from the set of sizes.
            size = next(iter(size))
        else:
            warnings.warn(f"Inconsistent sizes for dimension {dim}. Sizes: {size}.")
        dim_sizes[dim] = size
    return dim_sizes


class _ModuleAtHelper[M: Module]:
    def __init__(self, module_obj: M):
        self.module_obj = module_obj

    def __getitem__(self, indexing_object: Any):
        return _ModuleAtUpdateRef(self.module_obj, indexing_object)


class _ModuleAtUpdateRef[M: Module]:
    def __init__(self, module_obj: M, indexing_object: Any):
        self.module_obj = module_obj
        self.indexing_object = indexing_object

    def _get_map_leaf_fn(self, name: str, *args, **kwargs):
        from spekk import ops
        from spekk.ops._indexing import IndexingBehavior

        def map_leaf(leaf):
            if isinstance(leaf, ops.array):
                _at = leaf.at[self.indexing_object]
                _at.indexing_behavior = IndexingBehavior(leaf.dims)
                leaf = getattr(_at, name)(*args, **kwargs)
            return leaf

        return map_leaf

    def get(self) -> M:
        return traverse(self.module_obj, map_leaf=self._get_map_leaf_fn("get"))

    def set(self, value: "array") -> M:
        return traverse(self.module_obj, map_leaf=self._get_map_leaf_fn("set", value))

    def update(self, f: Callable[["array"], "array"], *args, **kwargs) -> M:
        return traverse(
            self.module_obj,
            map_leaf=self._get_map_leaf_fn("update", f, *args, **kwargs),
        )  # type: ignore


####
# Helper functions for updating Modules in an immutably fashion:

replace = dataclasses.replace


def replace_at[M: Module](obj: M, path: Sequence[str], new_value: Any) -> M:
    if not path:
        return new_value
    current, *rest = path
    return dataclasses.replace(
        obj, **{current: replace_at(getattr(obj, current), rest, new_value)}
    )


def update_at[M: Module](
    obj: M, path: Sequence[str], f: Callable, *args, **kwargs
) -> M:
    if not path:
        return f(obj, *args, **kwargs)
    current, *rest = path
    return dataclasses.replace(
        obj, **{current: update_at(getattr(obj, current), rest, f, *args, **kwargs)}
    )


def get_at(obj: Module, path: Sequence[str]):
    for step in path:
        obj = getattr(obj, step)
    return obj


def _noop(x):
    return x


def _recreate_container[M: TContainer](
    constructor: Callable[[Type[M], *tuple[Any, ...]], M], *args
) -> M:
    return constructor(*args)


type TLeaf = array | bool | int | float | complex
type TContainer = Module | list | tuple | dict


def traverse(
    obj: TLeaf | TContainer,
    *,
    map_leaf: Callable[[TLeaf], Any] = _noop,
    map_static_field: Optional[Callable[[Any], Any]] = _noop,
    recreate_container: Callable[..., TContainer] = _recreate_container,
) -> TLeaf | TContainer:
    recur = functools.partial(
        traverse,
        map_leaf=map_leaf,
        recreate_container=recreate_container,
        map_static_field=map_static_field,
    )

    # Handle basic Python container types
    if isinstance(obj, (list, tuple)):
        traversed_elements = tuple(recur(element) for element in obj)
        return recreate_container(type(obj), traversed_elements)
    elif isinstance(obj, dict):
        traversed_values = tuple(recur(value) for value in obj.values())
        return recreate_container(dict, tuple(zip(obj.keys(), traversed_values)))

    # Handle custom Module types
    elif isinstance(obj, Module):
        fields = dataclasses.fields(obj)
        traversed_values = []
        for _field in fields:
            value = getattr(obj, _field.name)
            if _field.metadata.get("static", False):
                traversed_values.append(map_static_field(value))
            else:
                traversed_values.append(recur(value))
        return recreate_container(type(obj), *traversed_values)

    # The rest are considered leaves
    return map_leaf(obj)


def _partial_at(func: Callable, fixed: dict[int, object], n_args: int) -> Callable:
    """Like functools.partial but for arbitrary positional slots.
    >>> def func(a, b, c):
    ...     return [a, b, c]
    >>> p_func = _partial_at(func, {1: "arg_1"}, 3)
    >>> p_func(0, 1)
    [0, 'arg_1', 1]
    """

    @functools.wraps(func)
    def wrapper(*args):
        it = iter(args)
        return func(*[fixed[i] if i in fixed else next(it) for i in range(n_args)])

    return wrapper


def traverse_multiple(
    *objs: TLeaf | TContainer,
    f: Callable[..., TLeaf],
    recreate_container: Callable = _recreate_container,
) -> TLeaf | TContainer:
    traversable_objs = []
    untraversable_objs = {}
    for i, obj in enumerate(objs):
        if isinstance(obj, (list, tuple, dict, Module)):
            traversable_objs.append(obj)
        else:
            untraversable_objs[i] = obj
    if not traversable_objs:
        return f(*objs)
    f = _partial_at(f, untraversable_objs, len(objs))
    objs = traversable_objs

    obj, *rest = objs
    assert all(type(x) is type(obj) for x in rest)

    recur = functools.partial(
        traverse_multiple,
        f=f,
        recreate_container=recreate_container,
    )

    # Handle basic Python container types
    if isinstance(obj, (list, tuple)):
        assert all(len(x) == len(obj) for x in rest)
        traversed_elements = [recur(*elements) for elements in zip(*objs)]
        constructor = lambda *args: type(obj)(args)
        return recreate_container(constructor, *traversed_elements)
    elif isinstance(obj, dict):
        assert all(x.keys() == obj.keys() for x in rest)
        traversed_values = [
            recur(*values) for values in zip(*[x.values() for x in objs])
        ]
        constructor = lambda *values: dict(zip(obj.keys(), values))
        return recreate_container(constructor, *traversed_values)

    # Handle custom Module types
    elif isinstance(obj, Module):
        fields = dataclasses.fields(obj)
        traversed_values = []
        for _field in fields:
            values = [getattr(x, _field.name) for x in objs]
            if _field.metadata.get("static", False):
                traversed_values.append(f(*values))
            else:
                traversed_values.append(recur(*values))
        field_names = [_field.name for _field in fields]
        constructor = lambda *values: type(obj)(**dict(zip(field_names, values)))
        return recreate_container(constructor, *traversed_values)

    # The rest are considered leaves
    return f(*objs)


####
# Functions and helpers for flattening a Module into dynamic and static components:

type _TContainerRecreator = Callable[..., TContainer]


class _DynamicArg:
    def __repr__(self):
        return "*"


class TreeDef[M: TContainer]:
    "A tree-like structure representing a tree-like object that was flattened."

    def __init__(
        self,
        constructor: Callable[..., M] | None,
        leaves: tuple[_DynamicArg | Any, ...],
    ):
        self.constructor = constructor
        self.leaves = leaves

    def unflatten(self, dynamic_args: Iterable):
        "Undo the flattening using the dynamic_args."
        # Ensure dynamic_args is an iterator.
        dynamic_args = iter(dynamic_args)

        leaves = []
        for leaf in self.leaves:

            def map_leaf(leaf):
                if isinstance(leaf, _DynamicArg):
                    return next(dynamic_args)
                elif isinstance(leaf, TreeDef):
                    return leaf.unflatten(dynamic_args)
                else:
                    return leaf

            leaf = traverse(leaf, map_leaf=map_leaf)
            leaves.append(leaf)

        # Return the original object.
        return self.constructor(*leaves)

    def __hash__(self) -> int:
        # Don't create unique hash-values for placeholder args.
        args = [None if isinstance(leaf, _DynamicArg) else leaf for leaf in self.leaves]
        return hash((self.constructor, *args))

    def __eq__(self, other) -> bool:
        if not isinstance(other, TreeDef):
            return False
        args = [None if isinstance(leaf, _DynamicArg) else leaf for leaf in self.leaves]
        args_other = [
            None if isinstance(leaf, _DynamicArg) else leaf for leaf in other.leaves
        ]
        return (args == args_other) and (self.constructor == other.constructor)

    def _get_repr(self):
        constructor_name = getattr(self.constructor, "__name__", repr(self.constructor))
        leaf_reprs = []
        for leaf in self.leaves:
            leaf_reprs.append(
                leaf._get_repr() if isinstance(leaf, TreeDef) else repr(leaf)
            )
        return f"{constructor_name}({", ".join(leaf_reprs)})"

    def __repr__(self):
        return f"<TreeDef {self._get_repr()}>"


class FlattenedTree(NamedTuple):
    dynamic: tuple[TLeaf, ...]
    treedef: TreeDef

    def filter_dynamic(self, predicate: Callable, *args, **kwargs) -> "FlattenedTree":
        """Return a new FlattenedTree object where all dynamic fields for which
        predicate returns False are made static instead."""
        dynamic = []
        leaves = []
        for arg in self.dynamic:
            if predicate(arg, *args, **kwargs):
                dynamic.append(arg)
                leaves.append(_DynamicArg())
            else:
                leaves.append(arg)

        treedef = TreeDef(lambda *leaves: self.treedef.unflatten(leaves), tuple(leaves))
        return FlattenedTree(tuple(dynamic), treedef)


def _is_array_like(x) -> bool:
    import spekk.ops as ops

    return isinstance(
        x, (int, float, complex, ops.array)
    ) or ops.backend._is_backend_array(x)


def flatten(obj, *, flatten_spekk_arrays: bool = False) -> FlattenedTree:
    """Flatten a tree-like structure recursively.

    This is useful when we want to pass custom Module objects into traced functions,
    e.g.: functions that have been wrapped with jax.jit. Flattening the object first
    lets us input only the tuple of dynamic fields and the backend doesn't have to know
    about our classes. Static fields are meant to be baked into the computation and if
    they change then it is assumed that the computation also meaningfully changes.

    Args:
        obj (TModule): A tree-like object, i.e. one of Module, list, tuple or dict.
        flatten_spekk_arrays (bool): Whether to also flatten arrays into "data"
            (dynamic) and "dims" (static). If False (default), then arrays are
            considered as leaves and as dynamic attributes.
    """
    from spekk import ops

    dynamic = []

    def map_leaf(leaf: TLeaf | _static_value):
        if isinstance(leaf, _static_value):
            return leaf.value
        elif not _is_array_like(leaf):
            return leaf

        # Else, leaf is an array or number:

        if flatten_spekk_arrays and isinstance(leaf, ops.array):
            dynamic.append(leaf.data)
            return TreeDef(ops.array, (_DynamicArg(), tuple(leaf.dims)))
        else:
            dynamic.append(leaf)
            return _DynamicArg()

    def as_treedef(type, *args):
        return TreeDef(type, args)

    treedef = traverse(obj, map_leaf=map_leaf, recreate_container=as_treedef)
    if not isinstance(treedef, TreeDef):
        treedef = TreeDef(lambda value: value, (treedef,))
    return FlattenedTree(tuple(dynamic), treedef)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
