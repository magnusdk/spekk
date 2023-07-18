from typing import Any, Sequence, Union

import numpy as np

from spekk import trees
from spekk.transformations import common


def compose(x, *wrapping_functions):
    """Apply each f in fs to x.

    Let's say we have some functions:
    >>> f = lambda x: x+1
    >>> g = lambda x: x*2
    >>> h = lambda x: x**2

    We can use compose to apply each function in order:
    >>> compose(1, f, g, h)
    16

    This would be the same as calling:
    >>> h(g(f(1)))
    16

    In situations with a lot of nested function calls, compose may be more readable.
    Also notice that when using compose, functions are evaluated in the order that they
    are passed in (left-to-right), while with the nested function calls, the functions
    are evaluated in the reverse order (right-to-left).

    Compose can also be used to build up a function from smaller function
    transformations:

    >>> wrap_double = lambda f: (lambda x: 2*f(x))
    >>> f = compose(
    ...   lambda x: x+1,
    ...   wrap_double,
    ...   wrap_double,
    ... )
    >>> f(1)
    8
    """
    for wrap in wrapping_functions:
        x = wrap(x)
    return x


def identity(x):
    "Return the input unchanged."
    return x


def get_fn_name(f) -> str:
    if hasattr(f, "__qualname__"):
        return f.__qualname__
    if hasattr(f, "__name__"):
        return f.__name__
    return repr(f)


def getitem_along_axis(x, axis: int, i: int):
    slice_ = tuple([slice(None)] * axis + [i])
    try:
        return x.__getitem__(slice_)
    except TypeError:
        try:
            return np.array(x).__getitem__(slice_)
        except Exception:
            raise ValueError(
                f"Cannot get item at index {i} along axis {axis} of object with type {type(x)}"
            )


def map_1_flattened(
    map_f: callable,
    flattened_args: Sequence[Any],
    in_axes: Sequence[Union[int, None]],
    unflatten: callable,
    i: int,
):
    args = []
    for arg, axis in zip(flattened_args, in_axes):
        # If axis is None then we leave the argument as is.
        if axis is not None:
            # If axis is not None, then get the item at index i along the given axis.
            arg = trees.update_leaves(
                arg,
                lambda x: not trees.has_treedef(x),
                lambda x: common.getitem_along_axis(x, axis, i),
            )
        args.append(arg)

    kwargs = unflatten(args)
    return map_f(**kwargs)


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
    elif hasattr(x, "__len__") and hasattr(x, "__getitem__"):
        # Assume each item in x has the same shape.
        return (len(x), *shape(x[0]))
    raise ValueError(f"Cannot get shape of object with type {type(x)}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
