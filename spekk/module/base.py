import abc
import dataclasses
import functools
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Hashable, Sequence, cast

from typing_extensions import dataclass_transform

import spekk.tree as tree

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

    # Declare __dataclass_fields__ to satisfy DataclassInstance protocol for type checkers
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

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
            # By flattening we compare both the structure (treedefs) and values. If any
            # values (be they dynamic or static) are non-scalar arrays, this may raise
            # an exception.
            return tree.flatten(self) == tree.flatten(other)
        return NotImplemented


def dim_sizes(obj):
    from spekk import ops
    from spekk.ops._types import _UndefinedDim

    undefined_dim_key = _UndefinedDim()

    # Gather the size(s) for each dimension for each array in self. Store the sizes
    # for each dimension in a set.
    dim_size_sets: dict[str | _UndefinedDim, set[int]] = defaultdict(set)
    dynamic, treedef = tree.flatten(obj)
    # Loop over both dynamic and static values
    for leaf in [*dynamic, *treedef.static_values]:
        if isinstance(leaf, ops.array):
            for dim, size in leaf.dim_sizes.items():
                if isinstance(dim, _UndefinedDim):
                    dim = undefined_dim_key
                dim_size_sets[dim].add(size)

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


def at(obj):
    return _ModuleAtHelper(obj)


class _ModuleAtHelper[M]:
    def __init__(self, module_obj: M):
        self.module_obj = module_obj

    def __getitem__(self, indexing_object: Any):
        return _ModuleAtUpdateRef(self.module_obj, indexing_object)


class _ModuleAtUpdateRef[M]:
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
        return tree.map(self._get_map_leaf_fn("get"), self.module_obj)

    def set(self, value: "array") -> M:
        return tree.map(self._get_map_leaf_fn("set", value), self.module_obj)

    def update(self, f: Callable[["array"], "array"], *args, **kwargs) -> M:
        return tree.map(
            self._get_map_leaf_fn("update", f, *args, **kwargs),
            self.module_obj,
        )


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


# Register Module as a treedef


@dataclasses.dataclass(frozen=True, repr=False)
class ModuleTreeDef(tree.registry.TreeDef):
    cls: type
    dynamic_fields: tuple[str, ...]
    static_fields: tuple[tuple[str, tree.TreeDef | tree.StaticLeaf], ...]

    def _construct(self, children: list) -> Any:
        kwargs = {
            field_name: frozen_static_value.unflatten()
            for field_name, frozen_static_value in self.static_fields
        }
        kwargs.update(zip(self.dynamic_fields, children))
        return self.cls(**kwargs)

    @property
    def static_values(self) -> tuple:
        return super().static_values + tuple(value for _, value in self.static_fields)

    def _inner_repr(self, child_repr_fn: Callable[[Any], str]) -> str:
        parts = []
        child_iter = iter(self.children)
        for name in self.dynamic_fields:
            parts.append(f"{name}={child_repr_fn(next(child_iter))}")
        for name, value in self.static_fields:
            parts.append(f"{name}={value!r}")
        return f"{self.cls.__name__}({', '.join(parts)})"


def _destructure_module(obj: Module):
    dynamic_fields = []
    static_fields = []
    children_values = []

    for field in dataclasses.fields(type(obj)):  # type: ignore[arg-type]
        value = getattr(obj, field.name)
        is_static = field.metadata.get("static", False)

        if is_static:
            # Freeze static values to make them immutible and make the most common
            # container objects hashable.
            frozen_static_value = tree.freeze_as_treedef(value)
            static_fields.append((field.name, frozen_static_value))
        else:
            dynamic_fields.append(field.name)
            children_values.append(value)

    def make_treedef(children: tuple[tree.registry.TreeDef, ...]) -> ModuleTreeDef:
        return ModuleTreeDef(
            children,
            type(obj),
            tuple(dynamic_fields),
            tuple(static_fields),
        )

    return children_values, make_treedef


tree.registry.register(lambda x: isinstance(x, Module), _destructure_module)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
