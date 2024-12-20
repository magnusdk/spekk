from spekk.ops._backend import backend


class _DType:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, _DType):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        try:
            other = _DType._from_backend_dtype(other)
            return self.name == other.name
        except Exception:
            pass
        return False
    
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

    @staticmethod
    def _from_backend_dtype(dtype) -> "_DType":
        return _DType(backend.dtype(dtype).name)


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
