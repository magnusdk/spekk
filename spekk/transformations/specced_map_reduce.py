from typing import Any, Callable, Iterable, Optional, TypeAlias, TypeVar

from spekk import Spec

T: TypeAlias = TypeVar("T")
T_map_f_result: TypeAlias = TypeVar("T_map_f_result")
T_reduction_result: TypeAlias = TypeVar("T_reduction_result")
T_reduce_f: TypeAlias = Callable[
    [T_reduction_result, T_map_f_result], T_reduction_result
]
T_reduce: TypeAlias = Callable[
    [T_reduce_f, Iterable, T_reduction_result], T_reduction_result
]


def specced_map_reduce(
    map_f: Callable[[Any], T_map_f_result],
    reduce_f: T_reduce_f,
    spec: Spec,
    dim: str,
    initial_value: Optional[T_reduction_result] = None,
    enumerate: bool = False,
    reduce_impl: Optional[T_reduce] = None,
):
    """Return a function that applies a map-reduce operation over a specified dimension
    of input data.

    This function returns a new function that performs a map operation followed by a
    reduce operation on the input data along a specified dimension. The data along the
    dimension is processed iteratively. This may reduce memory usage because only one
    item is kept in memory at a time."""
    # Default `reduce_impl` is Python's built-in `functools.reduce`. Anything that has
    # the same signature as that can be used as the `reduce_impl`.
    if reduce_impl is None:
        import functools

        reduce_impl = functools.reduce

    def map_reduce(*args, **kwargs):
        if args:
            raise ValueError("Positional args not supported. Use kwargs instead.")

        size = spec.size(kwargs, dim)
        # Special case: there are no elements to reduce over.
        if size == 0:
            return initial_value

        # Get the first value.
        x_0 = map_f(**spec.index_data(kwargs, 0, dim))
        if enumerate:
            x_0 = (0, x_0)

        # Apply the first reduction-step to the value
        result_0 = reduce_f(initial_value, x_0)

        # Special case: don't reduce further if there was only one value
        if size == 1:
            return result_0

        # `wrapped_inner` is the actual reducing function. It runs for each index in
        # the data along the given dimension, applies `map_f` to the data at that
        # index, before applying `reduce_f` with the result.
        def wrapped_inner(carry: T_reduction_result, i: int):
            result_i = map_f(**spec.index_data(kwargs, i, dim))
            result_i = (i, result_i) if enumerate else result_i
            return reduce_f(carry, result_i)

        # Reduce over the remaining elements. Note that `carry` is the first element,
        # so we start at index 1 in the `range(1, size)`.
        return reduce_impl(wrapped_inner, range(1, size), result_0)

    return map_reduce
