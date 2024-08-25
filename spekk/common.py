import operator
from typing import Any, Sequence


def index_at(x, i: int, axis: int):
    """

    Examples:
    >>> import numpy as np
    >>> x = np.ones((2,3,4))
    >>> index_at(x, 0, axis=0).shape
    (3, 4)
    >>> index_at(x, 0, axis=1).shape
    (2, 4)
    >>> index_at(x, 0, axis=2).shape
    (2, 3)

    >>> x = np.array([[10], [20], [30]])
    >>> x.shape
    (3, 1)
    >>> index_at(x, 0, axis=0)
    array([10])
    >>> index_at(x, 1, axis=0)
    array([20])
    >>> index_at(x, 0, axis=1)
    array([10, 20, 30])
    """
    g = tuple([slice(None)] * axis + [i])
    return x.__getitem__(g)


def canonicalize_index(i: int, n: int) -> int:
    """Canonicalize an index in [-n, n) to [0, n)."""
    i = operator.index(i)
    if not -n <= i < n:
        raise ValueError(f"Index {i} is out of bounds for array with axis size {n}")
    if i < 0:
        i = i + n + 1
    return i


def insert(xs: Sequence, index: int, x: Any) -> list:
    "Return a new copy of xs with x inserted at the given index."
    index = canonicalize_index(index, len(xs) + 1)
    xs = list(xs)
    return xs[:index] + [x] + xs[index:]


def get_fn_name(f) -> str:
    if hasattr(f, "__qualname__"):
        return f.__qualname__
    if hasattr(f, "__name__"):
        return f.__name__
    return repr(f)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
