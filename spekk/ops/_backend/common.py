import functools
from typing import Any, List, Sequence, Union

import numpy as np

from spekk.module import trees
from spekk.ops._types import UndefinedDim


def _get_key(static_xs: Sequence[Any]):
    from spekk import ops

    key = []
    for x in static_xs:
        if isinstance(x, ops.array):
            # Use the memory address (id for CPython) as key for static arrays
            key.append(hash((id(x), id(x.data))))
        elif ops.backend._is_backend_array(x):
            # Use the memory address (id for CPython) as key for static arrays
            key.append(id(x))
        elif isinstance(x, UndefinedDim):
            key.append(hash(x))
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
            from spekk import Dim, ops

            if isinstance(in_axes, UndefinedDim):
                raise ValueError(f"Can not vmap over an UndefinedDim: {in_axes=}")
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
                if not isinstance(in_axes, (Sequence, str)):
                    raise ValueError(
                        "Passing positional arguments to a vmapped function, where the "
                        f"in_axes is of type {type(in_axes)} is not allowed. Try passing "
                        "in only keyword arguments instead."
                    )
            else:
                positional_args = False
                original_args = original_kwargs
                if not isinstance(in_axes, (dict, str)):
                    raise ValueError(
                        "Passing keyword arguments to a vmapped function, where the "
                        f"in_axes is of type {type(in_axes)} is not allowed. Try passing "
                        "in only positional arguments instead."
                    )

            # Flatten the arguments down to only arrays and numbers. We do this so that
            # the backend vmap implementation only need to handle arrays and numbers :)
            # It doesn't need to know about custom Module objects, or the like.
            def is_tree_like_but_not_array(x):
                return trees.Tree.is_tree_like(x) and not isinstance(x, ops.array)

            flattened_arguments = trees.flatten(
                original_args, is_tree_like=is_tree_like_but_not_array
            )

            # Calculate the cache key. It is used to recompile the vmapped function if
            # the static parts of the arguments have changed.
            cache_key = _get_key(flattened_arguments.static)

            # We have already compiled this function: just run the compiled version!
            if cache_key in cache:
                wrapped_inner, unflatten_result_inner = cache[cache_key]
                result_inner = wrapped_inner(*flattened_arguments.dynamic)
                return unflatten_result_inner(result_inner)

            # Else, we need to compile the function.
            else:
                # Flatten in_axes by traversing the flattened arguments. in_axes is
                # assumed to be a nested structure of sequences and dictionaries,
                # corresponding to the structure of the original (not flattened) input
                # arguments. We flatten in_axes this because the its elements have to
                # correspond to the flattened arguments when running the backend vmap
                # implementation.
                flattened_in_axes = []
                if isinstance(in_axes, Dim):
                    for x in flattened_arguments.dynamic:
                        if isinstance(x, ops.array) and in_axes in x.dims:
                            flattened_in_axes.append(x.dims.index(in_axes))
                        else:
                            flattened_in_axes.append(None)
                else:
                    for path in flattened_arguments.paths:
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

                # Get the (underlying backend-) data and dims separately from the
                # flattened arguments. We want to send in only the underlying backend
                # data to the vmapped function since the backend won't necessarily know
                # how to handle spekk.ops.array objects. We also need to update the
                # dimensions when inside the vmapped function, since it will have one
                # less dimension in this context.
                flattened_dynamic_data = []
                flattened_dynamic_dims_vmap_context = []
                vmapped_dims = set()
                for x, axis in zip(flattened_arguments.dynamic, flattened_in_axes):
                    if isinstance(x, ops.array):
                        flattened_dynamic_data.append(x.data)
                        dims_vmap_context = list(x.dims)
                        if axis is not None:
                            vmapped_dims.add(dims_vmap_context[axis])
                            # Remove the vmapped dimension
                            del dims_vmap_context[axis]
                        flattened_dynamic_dims_vmap_context.append(dims_vmap_context)
                    else:
                        # If the argument is not a spekk.ops.array, just pass it as-is.
                        # Use None as the dimension to represent that it is not a
                        # spekk.ops.array (this fact is used in wrapped_inner, defined
                        # below).
                        flattened_dynamic_data.append(x)
                        flattened_dynamic_dims_vmap_context.append(None)

                known_vmapped_dims = [
                    d for d in vmapped_dims if not isinstance(d, UndefinedDim)
                ]
                if len(known_vmapped_dims) > 1:
                    raise ValueError(
                        "Vmapping over axes with differing dimensions is not allowed."
                    )
                # Iterate again because the number of known vmapped dimensions may be
                # zero, aka they may all be UndefinedDim.
                for vmapped_dim in vmapped_dims:
                    if not isinstance(vmapped_dim, UndefinedDim):
                        break

                # Declare a variable named flatten_result_inner that will be set within
                # wrapped_inner (defined below).
                # If f returns a module, it must be flattened before being returned out
                # to the backend vmap implementation. This is so that the backend vmap
                # implementation only needs to know about arrays and numbers. Later,
                # the returned value needs to be unflattened, which
                # flatten_result_inner handles. Since the result of f is unknown until
                # it has actually been called, flatten_result_inner is set dynamically
                # inside wrapped_inner.
                flattened_result_inner: trees.FlattenedResult = None
                flattened_dynamic_dims_inner: List[ops.Dim]

                # wrapped_inner is a function that takes flattened arguments of arrays
                # or numbers and returns the arrays and numbers of the flattened result
                # of calling f. It handles all flattening and unflattening logic. Note
                # that it sets flatten_result_inner, which is defined outside the scope
                # of wrapped_inner. This is so that we can unflatten the result
                # afterwards.
                def wrapped_inner(*flattened_args):
                    nonlocal flattened_result_inner, flattened_dynamic_dims_inner

                    # Re-add the original dimensions to the arguments, minus the one
                    # being vmapped over.
                    flattened_args = [
                        # If dims is None (see comment above when creating
                        # flattened_arguments_dynamic_dims), then the argument was
                        # never an array to begin with.
                        ops.array(x, dims) if dims is not None else x
                        for x, dims in zip(
                            flattened_args,
                            flattened_dynamic_dims_vmap_context,
                        )
                    ]

                    # Unflatten args, call f, and flatten the result.
                    args = flattened_arguments.treedef.unflatten(flattened_args)
                    result_inner = f(*args) if positional_args else f(**args)
                    flattened_result_inner = trees.flatten(
                        result_inner, is_tree_like=is_tree_like_but_not_array
                    )

                    # Return the flattened result. It will be unflattened outside of
                    # this function using flattened_result_inner.
                    flattened_dynamic_data_inner = [
                        x.data if isinstance(x, ops.array) else x
                        for x in flattened_result_inner.dynamic
                    ]
                    flattened_dynamic_dims_inner = [
                        x.dims if isinstance(x, ops.array) else None
                        for x in flattened_result_inner.dynamic
                    ]
                    return flattened_dynamic_data_inner

                # Wrap wrapped_inner with the backend vmap implementation and call it.
                wrapped_inner = vmap_impl(wrapped_inner, in_axes=flattened_in_axes)
                result_inner = wrapped_inner(*flattened_dynamic_data)

                def unflatten_result_inner(result_inner):
                    nonlocal flattened_dynamic_dims_inner
                    result_inner = [
                        ops.array(x, [vmapped_dim, *dims]) if dims is not None else x
                        for x, dims in zip(result_inner, flattened_dynamic_dims_inner)
                    ]
                    return flattened_result_inner.treedef.unflatten(result_inner)

                # Cache the compiled function
                cache[cache_key] = wrapped_inner, unflatten_result_inner

                # Unflatten the flattened result of calling vmap(f).
                return unflatten_result_inner(result_inner)

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
                wrapped_inner, unflatten = cache[cache_key]
                result_outer = wrapped_inner(*flatten_result_outer.dynamic)
                return unflatten(result_outer)
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
                cache[cache_key] = wrapped_inner, flatten_result_inner.treedef.unflatten
                return flatten_result_inner.treedef.unflatten(result_outer)

        return wrapped_outer

    return jit


def getitem_along_axis(x, axis: int, i: int):
    slice_ = tuple([slice(None)] * axis + [i, ...])
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
