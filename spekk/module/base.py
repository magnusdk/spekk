import abc
import dataclasses
import functools
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    cast,
    overload,
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
    >>> class MyClass(Module):
    ...     foo: float
    ...     bar: ops.array
    ...     quiz: int = field(static=True)

    Fields that are marked as static are filtered out during backend function
    compilation, similar to Equinox.
    """

    def __init_subclass__(cls):
        "Convert all subclasses of Module to dataclass."
        dataclasses.dataclass(cls)

    @property
    def dim_sizes(self) -> Dict["Dim", int | set[int] | dict[int, int]]:
        """Return a dictionary of the dimensions and corresponding sizes of all arrays
        in self. If two or more arrays contain the same dimension, but the sizes
        differ, a warning is printed, and a set of the different sizes is returned for
        that dimension. Undefined dimensions are combined into a single dict from size
        to number of arrays sharing that size.
        """
        from spekk import ops
        from spekk.ops._types import _UndefinedDim

        # Gather the size(s) for each dimension for each array in self. Store the sizes
        # for each dimension in a set. For undefined dimension, store in a dict how
        # many times an array has an axis with a specific size.
        dim_size_sets = defaultdict(set)
        undefined_dim_sizes = defaultdict(int)
        flattened = flatten(self)
        for x in flattened.dynamic + flattened.static:
            if isinstance(x, ops.array):
                for dim, size in x.dim_sizes.items():
                    if isinstance(dim, _UndefinedDim):
                        undefined_dim_sizes[size] += 1
                    else:
                        dim_size_sets[dim].add(size)

        # Go through all dimensions and corresponding set of sizes and check which
        # (if any) dimensions have inconsistent sizes. Get the single size for al other
        # dimensions.
        dim_sizes = {}
        inconsistent_dim_sizes = {}
        for dim, sizes in dim_size_sets.items():
            if len(sizes) > 1:
                dim_sizes[dim] = sizes
                inconsistent_dim_sizes[dim] = sizes
            else:
                # Get the single size from the set of sizes.
                size = next(iter(sizes))
                dim_sizes[dim] = size

        # Handle undefined dimensions or dimensions with inconsistent sizes.
        if undefined_dim_sizes:
            dim_sizes[_UndefinedDim()] = dict(undefined_dim_sizes)
        if inconsistent_dim_sizes:
            warnings.warn(
                f"Got inconsistent sizes for dimensions: {inconsistent_dim_sizes}.",
                category=UserWarning,
                stacklevel=1,
            )
        return dim_sizes

    @property
    def at(self):
        """Helper for updating slices of _all_ arrays in the object that have the
        sliced dimension(s)."""
        return _ModuleAtHelper(self)

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
    recreate_container: Callable[..., TContainer] = _recreate_container,
    map_static_field: Optional[Callable[[Any], Any]] = None,
) -> TContainer:
    recur = functools.partial(
        traverse,
        map_leaf=map_leaf,
        recreate_container=recreate_container,
        map_static_field=map_static_field,
    )

    # Handle basic Python container types
    if isinstance(obj, (list, tuple)):
        traversed_elements = [recur(element) for element in obj]
        constructor = lambda *args: type(obj)(args)  # noqa: E731
        return recreate_container(constructor, *traversed_elements)
    elif isinstance(obj, dict):
        traversed_values = [recur(value) for value in obj.values()]
        constructor = lambda *values: dict(zip(obj.keys(), values))
        return recreate_container(constructor, *traversed_values)

    # Handle custom Module types
    elif isinstance(obj, Module):
        fields = dataclasses.fields(obj)
        traversed_values = []
        for _field in fields:
            value = getattr(obj, _field.name)
            if map_static_field is not None and _field.metadata.get("static", False):
                traversed_values.append(map_static_field(value))
            else:
                traversed_values.append(recur(value))
        field_names = [_field.name for _field in fields]
        constructor = lambda *values: type(obj)(**dict(zip(field_names, values)))
        return recreate_container(constructor, *traversed_values)

    # The rest are considered leaves
    return map_leaf(obj)


####
# Functions and helpers for flattening a Module into dynamic and static components:


class _Arg:
    is_dynamic: bool

    @staticmethod
    def dynamic():
        marker = _Arg()
        marker.is_dynamic = True
        return marker

    @staticmethod
    def static():
        marker = _Arg()
        marker.is_dynamic = False
        return marker

    def __repr__(self):
        return "*" if self.is_dynamic else "_"


type _TContainerRecreator = Callable[..., TContainer]


@dataclasses.dataclass
class _Flattened[M: TContainer]:
    dynamic: tuple[TLeaf]
    static: tuple
    unflatten_ops: list[_TContainerRecreator | _Arg]

    def unflatten(self, dynamic: Iterable[TLeaf]) -> M:
        dynamic = iter(dynamic)
        static = iter(self.static)

        def _eval(op: _Arg | Sequence[_TContainerRecreator | _Arg]):
            if isinstance(op, _Arg):
                return next(dynamic) if op.is_dynamic else next(static)
            f, *args = op
            return f(*(_eval(arg) for arg in args))  # type: ignore

        return _eval(self.unflatten_ops)  # type: ignore

    def filter_dynamic(self, predicate: Callable, *args, **kwargs) -> "_Flattened[M]":
        """Return a new _Flattened object where all dynamic fields for which predicate
        returns False are made static instead."""
        dynamic = []
        static = []
        unflatten_ops: list[_TContainerRecreator | _Arg] = [
            lambda *args: self.unflatten(args)
        ]

        for obj in self.dynamic:
            if predicate(obj, *args, **kwargs):
                dynamic.append(obj)
                unflatten_ops.append(_Arg.dynamic())
            else:
                static.append(obj)
                unflatten_ops.append(_Arg.static())

        # Add the existing static fields to the end of the new tuple of static fields.
        static.extend(self.static)
        return _Flattened(tuple(dynamic), tuple(static), unflatten_ops)


def _is_array_like(x) -> bool:
    import spekk.ops as ops

    return isinstance(
        x, (int, float, complex, ops.array)
    ) or ops.backend._is_backend_array(x)


def flatten[M: TContainer](
    obj: M, *, flatten_spekk_arrays: bool = False
) -> _Flattened[M]:
    """Flatten the Module recursively and put dynamic and static attributes into
    separate tuples.

    This is useful when we want to pass custom Module objects into traced functions,
    e.g.: functions that have been wrapped with jax.jit. Flattening the object first
    lets us input only the tuple of dynamic fields and the backend doesn't have to know
    about our classes. Static fields are meant to be baked into the computation and if
    they change then it is assumed that the computation also meaningfully changes.

    Args:
        obj (TModule): An instance of a class that inherits from Module.
        flatten_spekk_arrays (bool): Whether to also flatten arrays into "data"
            (dynamic) and "dims" (static). If False (default), then arrays are
            considered as leaves and as dynamic attributes.

    Returns:
        An object containing the dynamic and static fields as separate tuples and that
        has an unflatten method for getting back the original object. See
        :class:`~spekk.module.base._Flattened` for more information.
    """
    from spekk import ops

    dynamic = []
    static = []

    def map_leaf(leaf: TLeaf | _static_value):
        if isinstance(leaf, _static_value):
            static.append(leaf.value)  # Unwrap the value
            return _Arg.static()
        elif _is_array_like(leaf):
            if flatten_spekk_arrays and isinstance(leaf, ops.array):
                dynamic.append(leaf.data)
                static.append(tuple(leaf.dims))
                return [ops.array, _Arg.dynamic(), _Arg.static()]
            else:
                dynamic.append(leaf)
                return _Arg.dynamic()
        else:
            static.append(leaf)
            return _Arg.static()

    def map_static_field(leaf):
        static.append(leaf)
        return _Arg.static()

    def as_sexpr_ops(recreate_fn, *args):
        return [recreate_fn, *args]

    unflatten_ops = traverse(
        obj,
        map_leaf=map_leaf,
        recreate_container=as_sexpr_ops,
        map_static_field=map_static_field,
    )
    return _Flattened(tuple(dynamic), tuple(static), unflatten_ops)  # type: ignore
