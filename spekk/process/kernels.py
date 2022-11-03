from dataclasses import dataclass
from typing import Callable

from spekk import Shape, Spec


@dataclass
class Kernel:
    f: Callable
    required_spec: Spec
    returned_shape: Shape

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def validate(self, spec: Spec):
        if spec != self.required_spec:
            raise ValueError(
                f"The required spec for the kernel does not match the given spec. Did you forget to vmap an axis?\n  Required spec: {repr(self.required_spec)}\n  Given spec: {repr(spec)}"
            )

    def __repr__(self) -> str:
        return f"Kernel({self.f})"

    def __hash__(self) -> int:
        return hash((self.f, self.required_spec, self.returned_shape))
