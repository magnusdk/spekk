from typing import Protocol


class Backend(Protocol):
    def vmap():
        ...

    def array():
        ...

    def take():
        ...

    def transpose():
        ...
