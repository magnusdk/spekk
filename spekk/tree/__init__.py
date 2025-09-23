import functools
from typing import Any, Callable

from spekk.module.base import traverse_multiple


class SENTINEL: ...


_SENTINEL = SENTINEL()


def map(f, tree: Any = _SENTINEL, /, *rest):
    if isinstance(tree, SENTINEL):
        return functools.partial(map, f)
    return traverse_multiple(tree, *rest, f=f)
