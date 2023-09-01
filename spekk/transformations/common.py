"Some common utility functions used by :mod:`spekk.transformations`."

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

    We can use :func:`compose` to apply each function in order:

    >>> compose(1, f, g, h)  # ((1 + 1) * 2) ** 2 = 16
    16

    This would be the same as calling:

    >>> h(g(f(1)))  # ((1 + 1) * 2) ** 2 = 16
    16

    In situations with a lot of nested function calls, :func:`compose` may be more
    readable. Also notice that when using compose, functions are evaluated in the order
    that they are passed in (left-to-right), while with the nested function calls, the
    functions are evaluated in the reverse order (right-to-left).

    :func:`compose` can also be used to build up a function from smaller function
    transformations:

    >>> wrap_f_double = lambda f: (lambda x: 2*f(x))
    >>> wrap_f_square = lambda f: (lambda x: f(x)**2)
    >>> f = compose(
    ...   lambda x: x+1,
    ...   wrap_f_double,
    ...   wrap_f_square,
    ... )
    >>> f(1)  # ((1 + 1) * 2) ** 2 = 16
    16
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
                f"Cannot get item at index {i} along axis {axis} for {x!r}"
            )


def get_args_for_index(
    args: Sequence, in_axes: Sequence[Union[int, None]], i: int
) -> Sequence:
    return [
        common.getitem_along_axis(arg, a, i) if a is not None else arg
        for arg, a in zip(args, in_axes)
    ]


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
                lambda x: (
                    common.getitem_along_axis(x, axis, i)
                    if hasattr(x, "__getitem__")
                    else x
                ),
            )
        args.append(arg)

    kwargs = unflatten(args)
    return map_f(**kwargs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
