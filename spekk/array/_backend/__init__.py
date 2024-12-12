from typing import Literal, Optional

_backend_priority = ["jax", "torch", "numpy"]


class Backend:
    def __init__(self, backend_name: Optional[Literal["numpy", "jax", "torch"]] = None):
        self.backend_name = backend_name
        if self.backend_name is None:
            for backend_name in _backend_priority:
                try:
                    self.set_backend(backend_name)
                    self.backend_name = backend_name
                    break
                except ImportError as e:
                    print(e)
                    continue
            else:
                raise ValueError(
                    f"No valid backends could be loaded. Please install one of {_backend_priority}."
                )

    def set_backend(self, backend_name: Literal["numpy", "jax", "torch"]):
        if backend_name not in ["numpy", "jax", "torch"]:
            raise ValueError(f"Unknown backend '{backend_name}'")
        old_backend_name = self.backend_name
        self.backend_name = backend_name

        # Ensure that the backend is properly installed by trying to get an arbitrary
        # attribute (modules are loaded lazily; see __getattr__ method of this class).
        try:
            self.pi
        except ImportError:
            self.backend_name = old_backend_name
            raise

    def _is_backend_array(self, x):
        return isinstance(x, type(self.empty(())))

    def _setitem_impl(self, x, key, value):
        if self.backend_name == "jax":
            return x.at.__getitem__(key).set(value)
        x.__setitem__(key, value)
        return x

    def __getattr__(self, name: str):
        if self.backend_name == "numpy":
            from array_api_compat import numpy as ops
        elif self.backend_name == "jax":
            import jax.numpy as ops
        elif self.backend_name == "torch":
            from array_api_compat import torch as ops
        return getattr(ops, name)

    def __repr__(self):
        return f"Backend('{self.backend_name}')"


backend = Backend()
