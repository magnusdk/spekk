from typing import Any, Callable, List, Sequence, TypeAlias

Dims: TypeAlias = List[str]


class ArrayNamespace:
    _backend: str
    ...


class SpeccedArray:
    _data: ...
    _dims: Dims
    ...


class _FlattenedModule:
    dynamic: Sequence[SpeccedArray]
    static: Sequence[Any]

    def unflatten(self, data: Sequence[SpeccedArray]) -> "Module": ...
    def filter(self, pred: Callable[[SpeccedArray], bool]) -> "_FlattenedModule": ...


class Module:
    def flatten(self) -> _FlattenedModule: ...
    def flatten_with_dimension(self, dim: str) -> _FlattenedModule:
        return self.flatten().filter(lambda a: dim in a._dims)
