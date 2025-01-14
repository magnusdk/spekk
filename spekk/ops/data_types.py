from typing import Any, Union

from spekk.ops._backend import backend


class _DType:
    def __init__(self, dtype: Union[str, "_DType", Any]):
        if isinstance(dtype, str):
            self.name = dtype
        elif isinstance(dtype, _DType):
            self.name = dtype.name
        else:
            try:
                self.name = backend.get_dtype_name(dtype)
            except Exception:
                raise ValueError(f"Unrecognized {dtype=}.")

    def __eq__(self, other):
        if isinstance(other, _DType):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return getattr(backend, self.name) == other

    def __repr__(self):
        return f"_DType('{self.name}')"

    def __hash__(self):
        return hash(self.name)

    def __call__(self, x):
        from spekk import ops

        return ops.astype(x, self)

    def _to_backend_dtype(dtype):
        if isinstance(dtype, _DType):
            return getattr(backend, dtype.name)
        elif isinstance(dtype, str):
            return getattr(backend, dtype)
        else:
            return dtype


int8 = _DType("int8")
int16 = _DType("int16")
int32 = _DType("int32")
int64 = _DType("int64")
uint8 = _DType("uint8")
uint16 = _DType("uint16")
uint32 = _DType("uint32")
uint64 = _DType("uint64")
float32 = _DType("float32")
float64 = _DType("float64")
complex64 = _DType("complex64")
complex128 = _DType("complex128")
bool = _DType("bool")
