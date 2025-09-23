import contextlib
import os
from typing import Literal, Optional

import array_api_compat


# TODO: Remove me when next version of array_api_compat comes out and just use their
# version instead.
def _is_writeable_array(x) -> bool:
    """
    Return False if ``x.__setitem__`` is expected to raise; True otherwise.

    Warning
    -------
    As there is no standard way to check if an array is writeable without actually
    writing to it, this function blindly returns True for all unknown array types.
    """
    if array_api_compat.is_numpy_array(x):
        return x.flags.writeable
    if array_api_compat.is_jax_array(x) or array_api_compat.is_pydata_sparse_array(x):
        return False
    return True


class Backend:
    def __init__(
        self,
        backend_name: Literal["numpy", "mlx", "jax", "torch", "cupy"] | None,
    ):
        self.backend_name = backend_name
        if backend_name is None:
            self._set_initial_backend()

    def _set_initial_backend(self):
        _backend_priority = ["jax", "torch", "cupy", "numpy"]
        for backend_name in _backend_priority:
            try:
                self.set_backend(backend_name)
                self.backend_name = backend_name
                break
            except ImportError:
                continue
        else:
            raise ValueError(
                "No valid backends could be loaded. Please install one of "
                f"{_backend_priority}."
            )

    @property
    def device(self):
        return self.active_device

    def set_backend(
        self, backend_name: Literal["numpy", "mlx", "jax", "torch", "cupy"]
    ):
        if backend_name not in ["numpy", "mlx", "jax", "torch", "cupy"]:
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

    @contextlib.contextmanager
    def temporary_backend(
        self, backend_name: Literal["numpy", "mlx", "jax", "torch", "cupy"]
    ):
        original_backend = self.backend_name
        self.set_backend(backend_name)
        try:
            yield
        finally:
            self.set_backend(original_backend)

    def _setitem_impl(self, x, key, value):
        if self.backend_name == "jax":
            return x.at.__getitem__(key).set(value)
        if not _is_writeable_array(x):
            x = self.asarray(x, copy=True)
        x.__setitem__(key, value)
        return x

    def _is_backend_array(self, x) -> bool:
        return self._get_active_backend_module()._is_backend_array(x)

    def _get_active_backend_module(self):
        if self.backend_name == "numpy":
            import spekk.ops._backend.included_backends.numpy as ops
        elif self.backend_name == "jax":
            import spekk.ops._backend.included_backends.jax as ops
        elif self.backend_name == "mlx":
            import spekk.ops._backend.included_backends.mlx as ops
        elif self.backend_name == "torch":
            import spekk.ops._backend.included_backends.torch as ops
        elif self.backend_name == "cupy":
            import spekk.ops._backend.included_backends.cupy as ops
        else:
            raise ValueError(f"Invalid active backend: '{self.backend_name}'.")
        return ops

    def __getattr__(self, name: str):
        return getattr(self._get_active_backend_module(), name)

    def __repr__(self):
        return f"Backend('{self.backend_name}')"


_env_backend = os.environ.get("SPEKK_BACKEND", None)
backend = Backend(_env_backend)
