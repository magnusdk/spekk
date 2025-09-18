from typing import Any, Union

import numpy as np
from spekk.ops._backend import backend
from spekk.ops._types import BackendDtype


class DType:
    def __init__(self, dtype: Union[str, "DType", Any]):
        if isinstance(dtype, str):
            self.name = dtype
        elif isinstance(dtype, DType):
            self.name = dtype.name
        else:
            try:
                self.name = backend.get_dtype_name(dtype)
            except Exception:
                raise ValueError(f"Unrecognized {dtype=}.")

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return getattr(backend, self.name) == other

    def __repr__(self):
        return f"_DType('{self.name}')"

    def __hash__(self):
        return hash((DType, self.name))

    def __call__(self, x):
        from spekk import ops

        return ops.astype(x, self)

    @staticmethod
    def _to_backend_dtype(
        dtype: "DType | BackendDtype | np.dtype | str",
    ) -> BackendDtype:
        if isinstance(dtype, DType):
            return getattr(backend, dtype.name)
        elif isinstance(dtype, str):
            return getattr(backend, dtype)
        else:
            return dtype


int8 = DType("int8")
int16 = DType("int16")
int32 = DType("int32")
int64 = DType("int64")
uint8 = DType("uint8")
uint16 = DType("uint16")
uint32 = DType("uint32")
uint64 = DType("uint64")
float32 = DType("float32")
float64 = DType("float64")
complex64 = DType("complex64")
complex128 = DType("complex128")
bool = DType("bool")
