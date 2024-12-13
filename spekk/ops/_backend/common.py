import functools
from typing import Any, Sequence, Union

import numpy as np

from spekk.module import trees


def _get_key(static_xs: Sequence[Any]):
    from fastmath import ops

    key = []
    for x in static_xs:
        if ops.is_array(x):
            # Use the memory address (id for CPython) as key for static arrays
            key.append(id(x))
        else:
            try:
                x = trees.Tree(x)
                key.append((hash(tuple(x.keys())), _get_key(x.children())))
            except trees.NotTreeLikeError:
                key.append(hash(x))
    return tuple(key)


def get_vmap_fn(vmap_impl):

    @functools.wraps(vmap_impl)
    def vmap(f, in_axes):
        cache = {}

        @functools.wraps(f)
        def wrapped_outer(*original_positional_args, **original_kwargs):
            if original_positional_args and original_kwargs:
                raise ValueError(
                    "You may pass in only positional arguments or only keyword arguments, "
                    "but not both at the same time to a vmapped function. This is because "
                    "the jax.vmap behavior keyword arguments are quite limited; i.e., they "
                    "are always vmapped over leading axes.\n"
                    "Our vmap allows passing a dictionary as the in_axes for mapping the "
                    "vectorized axes of functions that accept keyword arguments, but this "
                    "leaves no room for mapping the positional argument axes."
                )
            elif not (original_positional_args or original_kwargs):
                raise ValueError(
                    "Calling a vmapped function with no arguments is not allowed. Did you "
                    "forget to pass in arguments or did you not mean to vmap the function?"
                )
            elif original_positional_args:
                positional_args = True
                original_args = original_positional_args
                if not isinstance(in_axes, Sequence):
                    raise ValueError(
                        "Passing positional arguments to a vmapped function, where the "
                        f"in_axes is of type {type(in_axes)} is not allowed. Try passing "
                        "in only keyword arguments instead."
                    )
            else:
                positional_args = False
                original_args = original_kwargs
                if not isinstance(in_axes, dict):
                    raise ValueError(
                        "Passing keyword arguments to a vmapped function, where the "
                        f"in_axes is of type {type(in_axes)} is not allowed. Try passing "
                        "in only positional arguments instead."
                    )

            flatten_result_outer = trees.flatten(original_args)
            cache_key = _get_key(flatten_result_outer.static)
            if cache_key in cache:
                wrapped_inner, flatten_result_inner = cache[cache_key]
                result_outer = wrapped_inner(*flatten_result_outer.dynamic)
                return flatten_result_inner.treedef.unflatten(result_outer)
            else:
                flatten_result_inner: trees.FlattenedResult = None

                flattened_in_axes = []
                for path in flatten_result_outer.paths:
                    a = in_axes
                    for step in path:
                        if isinstance(a, int):
                            break
                        try:
                            a = a[step]
                        except Exception:
                            a = None
                            break
                    flattened_in_axes.append(a)
                flattened_in_axes = tuple(flattened_in_axes)

                @functools.partial(vmap_impl, in_axes=flattened_in_axes)
                def wrapped_inner(*flattened_args):
                    nonlocal flatten_result_inner
                    args = flatten_result_outer.treedef.unflatten(flattened_args)
                    result_inner = f(*args) if positional_args else f(**args)
                    flatten_result_inner = trees.flatten(result_inner)
                    return flatten_result_inner.dynamic

                result_outer = wrapped_inner(*flatten_result_outer.dynamic)
                cache[cache_key] = wrapped_inner, flatten_result_inner
                return flatten_result_inner.treedef.unflatten(result_outer)

        return wrapped_outer

    return vmap


def get_scan_fn(scan_impl):
    def scan(f, init, xs):
        flattened_carry = trees.flatten(init)

        def wrapped_f(carry, x):
            nonlocal flattened_carry
            carry = flattened_carry.treedef.unflatten(carry)

            new_carry, y = f(carry, x)
            flattened_carry = trees.flatten(new_carry)
            return flattened_carry.dynamic, y

        carry, ys = scan_impl(wrapped_f, flattened_carry.dynamic, xs)
        return flattened_carry.treedef.unflatten(carry), ys

    return scan


def get_jit_fn(jit_impl):
    @functools.wraps(jit_impl)
    def jit(f):
        "Our custom jit-function which filters out static fields."

        # We cache the jitted function (wrapped_inner) by the static fields. When the
        # static fields changes, the function is re-compiled.
        cache = {}

        @functools.wraps(f)
        def wrapped_outer(*original_args, **original_kwargs):
            # Flatten all args. flatten_result_outer knows which parts of the arguments
            # are static. The underlying jit_impl only ever sees non-static inputs; the
            # rest are baked into the function itself.
            #   This is why it is important to recompile the function when static
            # fields changes, otherwise the function runs with the old values for those
            # fields.
            flatten_result_outer = trees.flatten((original_args, original_kwargs))

            # Try to find an already-compiled version for the given static fields.
            cache_key = _get_key(flatten_result_outer.static)
            if cache_key in cache:
                # Cache hit!
                wrapped_inner, flatten_result_inner = cache[cache_key]
                result_outer = wrapped_inner(*flatten_result_outer.dynamic)
                return flatten_result_inner.treedef.unflatten(result_outer)
            else:
                # Cache miss! Now we have to compile it. This is simply done by
                # wrapping the function with the jit_impl.

                # flatten_result_inner will be set nonlocally inside wrapped_inner. We
                # flatten the result of calling f as well, such that backends only sees
                # arrays as outputs as well. We need to unflatten the result after, and
                # for that we use flatten_result_inner.
                flatten_result_inner: trees.FlattenedResult = None

                @jit_impl
                def wrapped_inner(*flattened_args):
                    nonlocal flatten_result_inner
                    args, kwargs = flatten_result_outer.treedef.unflatten(
                        flattened_args
                    )
                    result_inner = f(*args, **kwargs)
                    flatten_result_inner = trees.flatten(result_inner)
                    return flatten_result_inner.dynamic

                result_outer = wrapped_inner(*flatten_result_outer.dynamic)
                # Make sure to cache the function until next time. It is important to
                # cache flatten_result_inner AFTER calling wrapped_inner; otherwise it
                # will be stored as None.
                cache[cache_key] = wrapped_inner, flatten_result_inner
                return flatten_result_inner.treedef.unflatten(result_outer)

        return wrapped_outer

    return jit


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
    args: Sequence,
    in_axes: Sequence[Union[int, None]],
    i: int,
) -> Sequence:
    return [
        getitem_along_axis(arg, a, i) if a is not None else arg
        for arg, a in zip(args, in_axes)
    ]
