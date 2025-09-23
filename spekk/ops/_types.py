"""
Types for type annotations used in the array API standard.

The type variables should be replaced with the actual types for a given
library, e.g., for NumPy TypeVar('array') would be replaced with ndarray.
"""

from __future__ import annotations

__all__ = [
    "Any",
    "List",
    "Literal",
    "NestedSequence",
    "Optional",
    "PyCapsule",
    "SupportsBufferProtocol",
    "SupportsDLPack",
    "Tuple",
    "Union",
    "Sequence",
    "device",
    "dtype",
    "ellipsis",
    "finfo_object",
    "iinfo_object",
    "Enum",
    "DefaultDataTypes",
    "DataTypes",
    "Capabilities",
    "Info",
]

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    from spekk.ops import array
    from spekk.ops.data_types import DType


# DTypes
class BackendDtype(Protocol): ...


type InferableDType = BackendDtype | DType | np.dtype | str


type DTypeLike = DType | np.dtype | str
type DeviceLike = str


@runtime_checkable
class ArrayLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def size(self) -> int: ...
    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> DTypeLike: ...

    @property
    def device(self) -> DeviceLike: ...


type ArrayOrNumber = array | bool | int | float | complex


class _UndefinedDim:
    def __eq__(self, other):
        # Like NaN, an undefined dim can't be said to equal anything else, since we
        # can't know what it represents.
        return False

    def __repr__(self):
        return "?"

    __hash__ = object.__hash__


undefined_dim = _UndefinedDim()

Dim: TypeAlias = str
PossiblyUndefinedDim: TypeAlias = Dim | _UndefinedDim
Dims: TypeAlias = Sequence[Dim]


def is_undefined_dim(x) -> bool:
    return isinstance(x, _UndefinedDim)


class BackendArray(Protocol): ...


class BackendDevice(Protocol): ...


type device = BackendDevice
type dtype = DType
SupportsDLPack = TypeVar("SupportsDLPack")
PyCapsule = TypeVar("PyCapsule")
# ellipsis cannot actually be imported from anywhere, so include a dummy here
# to keep pyflakes happy. https://github.com/python/typeshed/issues/3556
ellipsis = TypeVar("ellipsis")


@dataclass
class finfo_object:
    """Dataclass returned by `finfo`."""

    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: dtype


@dataclass
class iinfo_object:
    """Dataclass returned by `iinfo`."""

    bits: int
    max: int
    min: int
    dtype: dtype


class NestedSequence[T](Protocol):
    def __getitem__(self, key: int, /) -> T | NestedSequence[T]: ...

    def __len__(self, /) -> int: ...


class Info(Protocol):
    """Namespace returned by `__array_namespace_info__`."""

    def capabilities(self) -> Capabilities: ...

    def default_device(self) -> device: ...

    def default_dtypes(self, *, device: Optional[device]) -> DefaultDataTypes: ...

    def devices(self) -> List[device]: ...

    def dtypes(
        self, *, device: Optional[device], kind: Optional[Union[str, Tuple[str, ...]]]
    ) -> DataTypes: ...


DefaultDataTypes = TypedDict(
    "DefaultDataTypes",
    {
        "real floating": dtype,
        "complex floating": dtype,
        "integral": dtype,
        "indexing": dtype,
    },
)
DataTypes = TypedDict(
    "DataTypes",
    {
        "bool": dtype,
        "float32": dtype,
        "float64": dtype,
        "complex64": dtype,
        "complex128": dtype,
        "int8": dtype,
        "int16": dtype,
        "int32": dtype,
        "int64": dtype,
        "uint8": dtype,
        "uint16": dtype,
        "uint32": dtype,
        "uint64": dtype,
    },
    total=False,
)
Capabilities = TypedDict(
    "Capabilities", {"boolean indexing": bool, "data-dependent shapes": bool}
)
