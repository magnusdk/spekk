import dataclasses
import functools
from typing import List, Sequence

from spekk.ops._types import undefined_dim


def _get_hashable_key(x):
    from spekk import Module, ops

    if isinstance(x, (list, tuple)):
        return (type(x), *(_get_hashable_key(element) for element in x))
    elif isinstance(x, dict):
        return (
            dict,
            *x.keys(),
            *(_get_hashable_key(element) for element in x.values()),
        )
    elif isinstance(x, Module):
        field_names = [field.name for field in dataclasses.fields(x)]
        return (
            type(x),
            *field_names,
            *(_get_hashable_key(getattr(x, name)) for name in field_names),
        )
    elif isinstance(x, ops.array):
        return hash(x._id)
    else:
        return x


def get_vmap_fn(vmap_impl):
    @functools.wraps(vmap_impl)
    def vmap(f, in_axes):
        from spekk import Dim, ops
        from spekk.module import flatten

        if not isinstance(in_axes, Dim):
            raise NotImplementedError()

        CACHE = {}

        @functools.wraps(f)
        def wrapped_outer(*original_positional_args, **original_kwargs):
            if in_axes is undefined_dim:
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

            flattened_arguments = flatten(original_args)

            # Calculate the cache key. It is used to recompile the vmapped function if
            # the static parts of the arguments have changed.
            cache_key = _get_hashable_key(flattened_arguments.static)
            if cache_key in CACHE:
                return CACHE[cache_key](*flattened_arguments.dynamic)

            # Else, we need to compile the function.
            else:
                # Flatten in_axes by traversing the flattened arguments. in_axes is
                # assumed to be a nested structure of sequences and dictionaries,
                # corresponding to the structure of the original (not flattened) input
                # arguments. We flatten in_axes this because the its elements have to
                # correspond to the flattened arguments when running the backend vmap
                # implementation.
                flattened_in_axes = []
                for x in flattened_arguments.dynamic:
                    if isinstance(x, ops.array) and in_axes in x.dims:
                        flattened_in_axes.append(x.dims.index(in_axes))
                    else:
                        flattened_in_axes.append(None)
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

                known_vmapped_dims = [d for d in vmapped_dims if d is not undefined_dim]
                if len(known_vmapped_dims) > 1:
                    raise ValueError(
                        "Vmapping over axes with differing dimensions is not allowed."
                    )
                # Iterate again because the number of known vmapped dimensions may be
                # zero, aka they may all be UndefinedDim.
                for vmapped_dim in vmapped_dims:
                    if vmapped_dim is not undefined_dim:
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
                flattened_result_inner = None
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
                    args = flattened_arguments.unflatten(flattened_args)
                    result_inner = f(*args) if positional_args else f(**args)
                    flattened_result_inner = flatten(result_inner)

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
                    return flattened_result_inner.unflatten(result_inner)

                # Cache the compiled function
                CACHE[cache_key] = lambda *dynamic_args: unflatten_result_inner(
                    wrapped_inner(*dynamic_args)
                )

                # Unflatten the flattened result of calling vmap(f).
                return unflatten_result_inner(result_inner)

        return wrapped_outer

    return vmap


def get_scan_fn(scan_impl):
    def scan(f, init, xs, unroll):
        from spekk.module.base import _Flattened, flatten

        flattened_carry = flatten(init, flatten_spekk_arrays=True)
        flattened_y: _Flattened = None

        def wrapped_f(carry, x):
            nonlocal flattened_carry, flattened_y
            carry = flattened_carry.unflatten(carry)

            new_carry, y = f(carry, x)
            flattened_carry = flatten(new_carry, flatten_spekk_arrays=True)
            flattened_y = flatten(y, flatten_spekk_arrays=True)
            return flattened_carry.dynamic, flattened_y.dynamic

        carry, ys = scan_impl(wrapped_f, flattened_carry.dynamic, xs, unroll=unroll)
        return flattened_carry.unflatten(carry), flattened_y.unflatten(ys)

    return scan


def get_jit_fn(jit_impl):
    @functools.wraps(jit_impl)
    def jit(
        f,
        static_argnums: Sequence[int] = (),
        static_argnames: Sequence[str] = (),
    ):
        "Our custom jit-function which filters out static fields."
        from spekk.module.base import _Flattened, _static_value, flatten

        # We cache the jitted function (wrapped_inner) by the static fields. When the
        # static fields changes, the function is re-compiled.
        CACHE = {}

        @functools.wraps(f)
        def wrapped_outer(*original_args, **original_kwargs):
            # Handle arguments explicitly marked as static
            original_args = list(original_args)
            for static_argnum in static_argnums:
                original_args[static_argnum] = _static_value(
                    original_args[static_argnum]
                )
            for static_argname in static_argnames:
                original_kwargs[static_argname] = _static_value(
                    original_kwargs[static_argname]
                )

            # Flatten all args. flatten_result_outer knows which parts of the arguments
            # are static. The underlying jit_impl only ever sees non-static inputs; the
            # rest are baked into the function itself.
            #   This is why it is important to recompile the function when static
            # fields changes, otherwise the function runs with the old values for those
            # fields.
            flattened_args = flatten(
                (original_args, original_kwargs), flatten_spekk_arrays=True
            )

            # Try to find an already-compiled version for the given static fields.
            cache_key = _get_hashable_key(flattened_args.static)
            if cache_key in CACHE:
                return CACHE[cache_key](*flattened_args.dynamic)
            else:
                # Cache miss! Now we have to compile it. This is simply done by
                # wrapping the function with the jit_impl.

                # flatten_result_inner will be set nonlocally inside wrapped_inner. We
                # flatten the result of calling f as well, such that backends only sees
                # arrays as outputs as well. We need to unflatten the result after, and
                # for that we use flatten_result_inner.
                flatten_result_inner: _Flattened = None

                @jit_impl
                def wrapped_inner(*args):
                    nonlocal flatten_result_inner
                    args, kwargs = flattened_args.unflatten(args)
                    result_inner = f(*args, **kwargs)
                    flatten_result_inner = flatten(
                        result_inner, flatten_spekk_arrays=True
                    )
                    return flatten_result_inner.dynamic

                result_outer = wrapped_inner(*flattened_args.dynamic)
                # Make sure to cache the function until next time. It is important to
                # cache flatten_result_inner AFTER calling wrapped_inner; otherwise it
                # will be stored as None.
                CACHE[cache_key] = lambda *dynamic_args: flatten_result_inner.unflatten(
                    wrapped_inner(*dynamic_args)
                )
                return flatten_result_inner.unflatten(result_outer)

        return wrapped_outer

    return jit
