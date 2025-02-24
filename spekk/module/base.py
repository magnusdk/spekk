import abc
import contextlib
import dataclasses
import functools
import operator
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


def is_array_like(x) -> bool:
    import spekk.ops as ops

    return isinstance(
        x, (int, float, complex, ops.array)
    ) or ops.backend._is_backend_array(x)


_MODULE_METHODS_CACHE = None


@contextlib.contextmanager
def cache_module_methods():
    global _MODULE_METHODS_CACHE
    previous_cache = _MODULE_METHODS_CACHE
    _MODULE_METHODS_CACHE = {}
    try:
        yield
    finally:
        if False:
            print(len(_MODULE_METHODS_CACHE))
            n_misses = 0
            n_hits = 0
            for a in _MODULE_METHODS_CACHE.values():
                n_misses += a.cache_info().misses
                n_hits += a.cache_info().hits
            print(f"{n_misses=}")
            print(f"{n_hits=}")
            a = {k: v.cache_info().hits for k, v in _MODULE_METHODS_CACHE.items()}
            a = sorted(list(a.items()), key=lambda item: item[1], reverse=True)
            for f, hits in a:
                print(hits, f)

        for cache in _MODULE_METHODS_CACHE.values():
            cache.cache_clear()
        _MODULE_METHODS_CACHE = previous_cache


@functools.wraps(dataclasses.field)
def field(*, static: bool = False, **kwargs):
    try:
        metadata = dict(kwargs.pop("metadata"))  # safety copy
    except KeyError:
        metadata = {}
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")

    if static:
        metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


def _wrap_method_as_cacheable(f):
    return f

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if _MODULE_METHODS_CACHE is None:
            return f(*args, **kwargs)
        if f not in _MODULE_METHODS_CACHE:
            _MODULE_METHODS_CACHE[f] = functools.lru_cache(maxsize=None, typed=True)(f)
        return _MODULE_METHODS_CACHE[f](*args, **kwargs)

    return wrapped


@dataclass_transform(
    frozen_default=True,
    eq_default=False,
    field_specifiers=(dataclasses.field, field),
)
class _ModuleMeta(abc.ABCMeta):
    def __new__(mcls, name: str, bases, namespace: Dict[str, Any], **kwargs):
        new_namespace = {}
        for attr_name, attr_value in namespace.items():
            if not attr_name.startswith("__"):
                if isinstance(attr_value, staticmethod):
                    attr_value = staticmethod(
                        _wrap_method_as_cacheable(attr_value.__func__)
                    )
                elif isinstance(attr_value, property):
                    attr_value = property(
                        _wrap_method_as_cacheable(attr_value.fget),
                        fset=attr_value.fset,
                        fdel=attr_value.fdel,
                        doc=attr_value.__doc__,
                    )
                elif callable(attr_value):
                    attr_value = _wrap_method_as_cacheable(attr_value)
            new_namespace[attr_name] = attr_value

        cls = super().__new__(mcls, name, bases, new_namespace, **kwargs)
        cls = dataclasses.dataclass(cls, frozen=True, eq=False, init=True)
        return cls

    def __call__(cls, *args, **kwargs):
        import spekk.ops as ops

        args = [
            ops.array(arg) if ops.backend._is_backend_array(arg) else arg
            for arg in args
        ]
        kwargs = {
            key: ops.array(value) if ops.backend._is_backend_array(value) else value
            for key, value in kwargs.items()
        }
        return super(_ModuleMeta, cls).__call__(*args, **kwargs)


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
                leaf = getattr(leaf.at[self.slices], name)(*args, **kwargs)
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


class Module(metaclass=_ModuleMeta):
    @property
    def dim_sizes(self) -> Dict["Dim", int]:
        from spekk import ops

        dim_sizes = {}
        flattened = flatten(self)
        for x in flattened.dynamic + flattened.static:
            if isinstance(x, ops.array):
                dim_sizes.update(x.dim_sizes)
        return dim_sizes

    @property
    def at(self):
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

    def __hash__(self):
        fields = []
        for _field in dataclasses.fields(self):
            value = getattr(self, _field.name)
            if isinstance(value, dict):
                value = tuple(value.keys()) + tuple(value.values())
            elif isinstance(value, list):
                value = tuple(value)
            fields.append(value)
        return hash((self.__class__, *fields))


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


def flatten(obj: TModule, *, flatten_spekk_arrays: bool = False) -> _Flattened:
    from spekk import ops

    dynamic = []
    static = []

    def map_leaf(leaf):
        if is_array_like(leaf):
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


if __name__ == "__main__":
    import spekk.ops as ops

    ops.backend.set_backend("numpy")

    class Foo(Module):
        a: ops.array
        b: str

        @abc.abstractmethod
        def bar(self): ...

    class Bar(Foo):
        c: int

        def bar(self):
            print("Hello, Method!")
            return 2

        @property
        def my_property(self):
            print("Hello, Property!")
            return 3

    class Quiz(Module):
        d: Bar
        e: int

    obj = Quiz(Bar(ops.ones((2, 5), dims=["rx", "tx"]), "abc", 2), 3)

    with cache_module_methods():
        print(obj.d.bar(), obj.d.bar(), obj.d.my_property, obj.d.my_property)
    print()
    print(obj.d.bar(), obj.d.bar(), obj.d.my_property, obj.d.my_property)
    print()

    print(obj)
    print(replace(obj, e=30_000))
    print(replace_at(obj, ["d", "a"], 1.1))
    print(update_at(obj, ["d", "a"], operator.add, 0.2))
    print(obj)
    flattened_obj = flatten(obj)
    print(flattened_obj)
    print(flattened_obj.unflatten((100, "foobar") + flattened_obj.dynamic[2:]))

    print()
    print()
    print(obj.dim_sizes)
    print(obj)
    print(obj.at["tx", ::2].set(10))
