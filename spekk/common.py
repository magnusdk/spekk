from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Specable(Protocol):
    @property
    def shape(self) -> Sequence[int]:
        ...


class ValidationError(Exception):
    ...


class InvalidDimensionError(Exception):
    ...
