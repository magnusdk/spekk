from spekk.backends.protocol import Backend
from spekk.backends.numpy import NumpyBackend


class ProxyBackend(Backend):
    def __getattribute__(self, attr):
        return getattr(_active_backend, attr)


_active_backend = NumpyBackend()
backend = ProxyBackend()


def set_backend(new_backend: Backend):
    global _active_backend
    _active_backend = new_backend
