"Generalized :func:`np.shape` that works on nested Python sequences."

from typing import Sequence


def shape(x) -> Sequence[int]:
    """Get the shape of an array, number, or a nested sequence of numbers.

    >>> shape(1.0)
    ()
    >>> shape([0, 1, 2])
    (3,)
    >>> shape([[0, 1, 2], [3, 4, 5]])
    (2, 3)

    >>> import numpy as np
    >>> shape(np.ones((2, 3)))
    (2, 3)
    """
    if isinstance(x, (int, float, complex)):
        return ()
    elif hasattr(x, "shape"):
        return x.shape
    elif isinstance(x, (list, tuple, range)):
        # Assume each item in x has the same shape.
        return (len(x), *shape(x[0]))
    raise ValueError(f"Cannot get shape of object with type {type(x)}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
