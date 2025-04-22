import abc
import dataclasses
import functools
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from typing_extensions import dataclass_transform

if TYPE_CHECKING:
    from spekk.ops._types import Dim
    from spekk.ops.array_object import array


TModule = TypeVar("TModule", bound="Module")
TContainer = Union[list, tuple, dict, "Module"]
T = TypeVar("T")
V = TypeVar("V")


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
    """Internal class, do not use it (unless you really want to; noone can stop you).
    Marker for letting spekk know that a value is considered static, and should trigger
    JIT-recompilation when changed."""

    value: Any

    def __repr__(self):
        return f"static_value({self.value})"


@dataclass_transform(eq_default=False, field_specifiers=(dataclasses.field, field))
class _ModuleMeta(abc.ABCMeta):
    """Metaclass for the Module base class.

    Wraps classes with dataclass during class creation and ensures that all backend
    arrays are converted to spekk arrays during object instantiation.
    """

    # __new__ is called whenever a class that inherits from Module is defined.
    def __new__(mcls, name: str, bases, namespace: Dict[str, Any], **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        # Wrap the new class in dataclass.
        cls = dataclasses.dataclass(cls, eq=False, init=True)
        return cls

    # __call__ is called whenever we create a new instance of a class that uses
    # _ModuleMeta as the metaclass or inherits from Module (defined below).
    def __call__(cls, *args, **kwargs):
        import spekk.ops as ops

        # Ensure that the arguments are arrays and not backend arrays.
        args = [
            (ops.array(arg) if ops.backend._is_backend_array(arg) else arg)
            for arg in args
        ]
        kwargs = {
            key: (ops.array(value) if ops.backend._is_backend_array(value) else value)
            for key, value in kwargs.items()
        }
        return super(_ModuleMeta, cls).__call__(*args, **kwargs)


class Module(metaclass=_ModuleMeta):
    """Base class for custom classes that are understood by spekk.

    Classes that inherit from Module are dataclasses:
    >>> class MyClass(Module):
    ...     foo: float
    ...     bar: ops.array
    ...     quiz: int = field(static=True)

    Fields that are marked as static are filtered out during backend function
    compilation, similar to Equinox.
    """

    @property
    def dim_sizes(self) -> Dict["Dim", int]:
        """Return a dictionary of the dimensions and corresponding sizes of all arrays
        in self.
        """
        from spekk import ops
        from spekk.ops._types import _UndefinedDim

        # Gather the size(s) for each dimension for each array in self. Store the sizes
        # for each dimension in a set.
        dim_size_sets = defaultdict(set)
        flattened = flatten(self)
        for x in flattened.dynamic + flattened.static:
            if isinstance(x, ops.array):
                for dim, size in x.dim_sizes.items():
                    if isinstance(dim, _UndefinedDim):
                        raise ValueError(
                            "Can not calculated dimension sizes when some arrays have "
                            "undefined dimensions."
                        )
                    dim_size_sets[dim].add(size)

        # Go through all dimensions and correpsonding set of sizes. Raise a ValueError
        # if a dimension has more than one corresponding size. If no error was raised,
        # return a dictionary from dim to the size from the set.
        dim_sizes = {}
        for dim, sizes in dim_size_sets.items():
            if len(sizes) > 1:
                raise ValueError(
                    f"Inconsistent sizes for dimension {dim}. Sizes: {sizes}."
                )
            # Get the single size from the set of sizes.
            size = next(iter(sizes))
            dim_sizes[dim] = size
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
                    return a._id == b._id
                elif a != b:
                    return False
            return True
        return NotImplemented


####
# Helpers for slicing arrays in Module objects:


class _ModuleAtHelper:
    def __init__(self, module_obj: "array"):
        self.module_obj = module_obj

    def __getitem__(self, slices):
        return _ModuleAtUpdateRef(self.module_obj, slices)


class _ModuleAtUpdateRef:
    def __init__(self, module_obj: "Module", slices: tuple):
        self.module_obj = module_obj
        self.slices = slices

    def _get_map_leaf_fn(self, name: str, *args, **kwargs):
        from spekk import ops

        def map_leaf(leaf):
            if isinstance(leaf, ops.array):
                indexing_ref = leaf.at[self.slices]._with_indexing_behavior(
                    raise_if_slice_dim_not_in_x=False
                )
                leaf = getattr(indexing_ref, name)(*args, **kwargs)
            return leaf

        return map_leaf

    def get(self) -> "array":
        return traverse(self.module_obj, map_leaf=self._get_map_leaf_fn("get"))

    def set(self, value: "array") -> "array":
        return traverse(self.module_obj, map_leaf=self._get_map_leaf_fn("set", value))

    def update(self, f: Callable[["array"], "array"], *args, **kwargs) -> "array":
        return traverse(
            self.module_obj,
            map_leaf=self._get_map_leaf_fn("update", f, *args, **kwargs),
        )


####
# Helper functions for updating Modules in an immutably fashion:

replace = dataclasses.replace


def replace_at(obj: TModule, path: Sequence[str], new_value: T) -> TModule:
    if not path:
        return new_value
    current, *rest = path
    return dataclasses.replace(
        obj, **{current: replace_at(getattr(obj, current), rest, new_value)}
    )


def update_at(
    obj: TModule,
    path: Sequence[str],
    f: Callable[[T], T],
    *args,
    **kwargs,
) -> TModule:
    if not path:
        return f(obj, *args, **kwargs)
    current, *rest = path
    return dataclasses.replace(
        obj, **{current: update_at(getattr(obj, current), rest, f, *args, **kwargs)}
    )


def _noop(x):
    return x


def _recreate_container(constructor, *args):
    return constructor(*args)


def traverse(
    obj: T,
    *,
    map_leaf: Callable[[V], V] = _noop,
    recreate_container: Callable = _recreate_container,
    map_static_field: Optional[Callable[[V], V]] = None,
) -> T:
    recur = functools.partial(
        traverse,
        map_leaf=map_leaf,
        recreate_container=recreate_container,
        map_static_field=map_static_field,
    )

    # Handle basic Python container types
    if isinstance(obj, (list, tuple)):
        traversed_elements = [recur(element) for element in obj]
        constructor = lambda *args: type(obj)(args)
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


@dataclasses.dataclass
class _Flattened:
    dynamic: tuple
    static: tuple
    unflatten_ops: list

    def unflatten(self, dynamic: Iterable) -> Any:
        dynamic = iter(dynamic)
        static = iter(self.static)

        def _eval(op):
            if isinstance(op, _Arg):
                return next(dynamic) if op.is_dynamic else next(static)
            f, *args = op
            return f(*(_eval(arg) for arg in args))

        return _eval(self.unflatten_ops)

    def filter_dynamic(self, predicate: Callable, *args, **kwargs) -> "_Flattened":
        """Return a new _Flattened object where all dynamic fields for which predicate
        returns False are made static instead."""
        dynamic = []
        static = []
        unflatten_ops = [lambda *args: self.unflatten(args)]

        for obj in self.dynamic:
            if predicate(obj, *args, **kwargs):
                dynamic.append(obj)
                unflatten_ops.append(_Arg.dynamic())
            else:
                static.append(obj)
                unflatten_ops.append(_Arg.static())

        # Add the existing static fields to the end of the new tuple of static fields.
        static.extend(self.static)
        return _Flattened(dynamic, static, unflatten_ops)


def _is_array_like(x) -> bool:
    import spekk.ops as ops

    return isinstance(
        x, (int, float, complex, ops.array)
    ) or ops.backend._is_backend_array(x)


def flatten(obj: TModule, *, flatten_spekk_arrays: bool = False) -> _Flattened:
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

    def map_leaf(leaf):
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
    return _Flattened(tuple(dynamic), tuple(static), unflatten_ops)
