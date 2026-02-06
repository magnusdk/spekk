import functools
from typing import (
    Callable,
    Sequence,
    overload,
)

from spekk import ops, tree
from spekk.ops._backend import backend
from spekk.ops._types import Dim
from spekk.ops._util import as_backend_arrays
from spekk.ops.array_object import array


def _argf_over_dims(
    f: Callable[[ops.array], int | ops.array],
    x: ops.array,
    *,
    axis: tuple[Dim] | list[Dim] | Dim | None = None,
) -> dict[Dim, int | ops.array]:
    """Can be used like argmin or argmax, but allows applying over multiple dimensions
    at once and integrates better with named dimensions.

    f can be either argmin or argmax (or any other function that similarly returns an
    index).

    Example:
    >>> x = ops.array(np.random.randn(2, 3, 4), ["a", "b", "c"])
    >>> min_i = _argf_over_dims(ops.argmin, x, axis=["a", "b"])
    >>> assert all(x[min_i] == ops.min(x, axis=["a", "b"]))
    """

    if axis is None:
        axis = x.dims
    elif not isinstance(axis, (tuple, list)):
        axis = [axis]
    assert all(isinstance(dim, Dim) for dim in axis)

    dim, *dims = axis
    i = f(x, axis=dim)
    x = x[dim, i]
    result = {dim: i}
    for dim in dims:
        i = f(x, axis=dim)
        x = x[dim, i]
        result = {result_dim: indices[dim, i] for result_dim, indices in result.items()}
        result[dim] = i
    return result


argmin_over_dims = functools.partial(_argf_over_dims, ops.argmin)
argmax_over_dims = functools.partial(_argf_over_dims, ops.argmax)


@overload
def jit[TFunc: Callable](
    f: TFunc,
    /,
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
) -> TFunc: ...
@overload
def jit[TFunc: Callable](
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
) -> Callable[[TFunc], TFunc]: ...
def jit[TFunc: Callable](
    f: TFunc | None = None,
    /,
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
) -> Callable[[TFunc], TFunc] | TFunc:
    """Just-in-time (JIT) compile `f` if the active backend supports it. Optionally
    mark arguments as static via `static_argnums` and `static_argnames`. Marking
    arguments as static means that the function is recompiled every time they change.
    Static arguments have to be hashable or be a nested list/tuple/dict of hashable
    arguments, or a `spekk.array`.

    For backends that doesn't support JIT (like numpy), this is a no-op."""

    if f is None:

        def wrapper(f: TFunc) -> TFunc:
            return jit(
                f, static_argnums=static_argnums, static_argnames=static_argnames
            )

        return wrapper
    return backend.jit(
        f,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )


def scan(fn, init, xs, *, dim: str | None = None, unroll: int = 1):
    return ops.backend.scan(fn, init, xs, dim=dim, unroll=unroll)


def reduce_over_dim[TInitialValue, TCarry, TInputData](
    reduce_f: Callable[[TCarry | TInitialValue, TInputData], TCarry],
    data: TInputData,
    *,
    init: TInitialValue,
    dim: Dim,
    include_index: bool = False,
    unroll: int | bool = 1,
) -> TCarry:
    from spekk.module import dim_sizes as get_dim_sizes
    from spekk.ops._util import as_backend_arrays

    dim_sizes = get_dim_sizes(data)
    if dim not in dim_sizes:
        raise ValueError(f"Dimension {dim} not found in the data.")

    # Flatten the initial carry state.
    flat_carry_leaves, flat_carry_treedef = as_backend_arrays(tree.flatten(init))

    # Flatten the object that we want to reduce over into a sequence of arrays that has
    # dim in its dimensions.
    flat_outer = tree.flatten(
        data,
        is_static=lambda x: isinstance(x, ops.array) and (dim not in x.dims),
    )
    # Ensure that the dimension being reduced over is at the first axis.
    dynamic = [ops.moveaxis(x, dim, 0) for x in flat_outer.leaves]

    # Extract underlying data and dims
    dynamic_data = [x.data for x in dynamic]
    dynamic_dims = [x.dims for x in dynamic]

    def _scan_f(carry, dyn_data):
        nonlocal flat_carry_treedef
        # Reconstruct the carry from its flattened version.
        carry = flat_carry_treedef.unflatten(carry)

        # Remove index from dynamic data.
        if include_index:
            index = dyn_data[-1]
            dyn_data = dyn_data[:-1]

        # Rebuild the current dynamic arrays for this time step without the leading
        # dimension. It does not have the leading dimension because that is what we are
        # "iterating" over.
        current_dynamic = [
            ops.array(data, dims[1:]) for data, dims in zip(dyn_data, dynamic_dims)
        ]
        reduce_f_args = [carry, flat_outer.treedef.unflatten(current_dynamic)]
        # Add index as last argument to reduce_f.
        if include_index:
            reduce_f_args.append(index)

        # Unflatten the data, call reduce_f, and flatten the result
        flat_carry_leaves, flat_carry_treedef = as_backend_arrays(
            tree.flatten(reduce_f(*reduce_f_args))
        )
        return flat_carry_leaves, None

    # Run the backend scan implementation on the dynamic backend data.
    if include_index:
        # Add indices to the end of dynamic data. NB! We NEED to extract it out again
        # before calling unflatten on the dynamic data.
        indices = backend.arange(dim_sizes[dim])
        dynamic_data.append(indices)

    new_dynamic_0, _ = _scan_f(flat_carry_leaves, [x[0] for x in dynamic_data])
    new_dynamic, _ = ops.backend.scan(
        _scan_f,
        new_dynamic_0,
        [x[1:] for x in dynamic_data],
        unroll=unroll,
    )
    return flat_carry_treedef.unflatten(new_dynamic)


def map_reduce_over_dim[TInitialValue, TCarry, TInputData, TMappedInputData](
    map_f: Callable[[TInputData], TMappedInputData],
    reduce_f: Callable[[TCarry | TInitialValue, TMappedInputData], TCarry],
    data: TInputData,
    *,
    init: TInitialValue,
    dim: Dim,
    include_index_in_reduce: bool = False,
    unroll: int | bool = 1,
) -> TCarry:
    if include_index_in_reduce:

        def f(carry, part, index):
            return reduce_f(carry, map_f(part), index)
    else:

        def f(carry, part):
            return reduce_f(carry, map_f(part))

    return reduce_over_dim(
        f,
        data,
        init=init,
        dim=dim,
        include_index=include_index_in_reduce,
        unroll=unroll,
    )


def map_over_dim(
    map_f: Callable,
    data,
    *,
    dim: Dim | Sequence[Dim],
    include_index: bool = False,
    unroll: int | bool = 1,
):
    """
    Iterates over each element of `dim` in `data`, applies `map_f` to it, and returns
    a new data object of the results along the same dimension.
    """
    from spekk.module import dim_sizes as get_dim_sizes

    # Handle case where multiple dims are given. Then we map over each dimension
    # individually.
    if not isinstance(dim, Dim):
        if not isinstance(dim, Sequence):
            raise ValueError(
                f"dim must be either a Dim or a sequence of Dim. Got '{type(dim)}'."
            )
        dim, *remaining_dims = dim
        if remaining_dims:
            return map_over_dim(
                lambda data_1: map_over_dim(map_f, data_1, dim=remaining_dims),
                data,
                dim=dim,
            )

    # Validate dimension
    # dim_sizes = data.dim_sizes
    dim_sizes = get_dim_sizes(data)
    if dim not in dim_sizes:
        raise ValueError(f"Dimension {dim} not found in the data.")

    # Optionally include index array
    if include_index:
        data = (ops.arange(dim_sizes[dim], dim=dim), data)

    # Flatten input data trees for dynamic arrays along `dim`
    flat_in_leaves, flat_in_treedef = tree.flatten(
        data,
        is_static=lambda x: isinstance(x, ops.array) and (dim not in x.dims),
    )

    # Move `dim` to leading axis and extract raw data and dims
    moved = [ops.moveaxis(arr, dim, 0) for arr in flat_in_leaves]
    in_data = tuple(arr.data for arr in moved)
    in_dims = tuple(arr.dims for arr in moved)

    # Flatten output tree structure by running map_f on a dummy first to capture shape
    # Prepare scanning
    flat_out_template: tree.TreeDef | tree.Leaf | tree.StaticLeaf = None  # type: ignore
    out_dims = None

    def _scan_f(_, dyn_vals):  # dyn_vals is sequence of numpy arrays for this step
        nonlocal flat_out_template, out_dims
        # Reconstruct input arrays for this time-step
        current = [ops.array(val, dims[1:]) for val, dims in zip(dyn_vals, in_dims)]
        inp = flat_in_treedef.unflatten(current)
        # Flatten and cache template dims on first call
        flat_dynamic, flat_out_template = tree.flatten(map_f(inp))
        flat_dynamic = tuple(
            ops.array(x) if not isinstance(x, ops.array) else x for x in flat_dynamic
        )
        out_dims = [[dim, *x.dims] for x in flat_dynamic]
        flat_dynamic = tuple(
            x.data if isinstance(x, ops.array) else None for x in flat_dynamic
        )
        # Return unchanged carry and this step's output dynamics
        return None, flat_dynamic

    # Initialize dummy carry and capture first step
    first_vals = [arr[0] for arr in in_data]
    carry0, first_out = _scan_f(None, first_vals)

    # Run scan over the rest
    rest_vals = [arr[1:] for arr in in_data]
    _, rest_out = ops.backend.scan(_scan_f, carry0, rest_vals, unroll=unroll)

    # Combine first and rest outputs for each dynamic output array
    all_out = []
    for first_arr, rest_arr in zip(first_out, rest_out):
        all_out.append(ops.backend.concat([first_arr[None, ...], rest_arr], axis=0))

    # Rewrap into ops.array with leading dimension restored
    wrapped = [ops.array(arr, dims=dims) for arr, dims in zip(all_out, out_dims)]

    # Reconstruct tree of outputs
    return flat_out_template.unflatten(wrapped)


def vmap(f, in_axes):
    return backend.vmap(f, in_axes=in_axes)


def grad(f=None, /, argnums: int | Sequence[int] = 0):
    """Create a function that evaluates the gradient of ``f``.

      f: Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, standard Python containers or spekk
        Modules. Argument arrays in the positions specified by ``argnums`` must be of
        inexact (i.e., floating-point or complex) type. It should return a scalar (which
        includes arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).

    Returns:
      A function with the same arguments as ``f``, that evaluates the gradient
      of ``f``. If ``argnums`` is an integer then the gradient has the same
      shape and type as the positional argument indicated by that integer. If
      argnums is a tuple of integers, the gradient is a tuple of values with the
      same shapes and types as the corresponding arguments.
    """

    # Allow the following syntax:
    #   @grad(argnums=1)
    #   def f(a, b): ...
    # which is shorthand for:
    #   @functools.partial(grad, argnums=1)
    #   def f(a, b): ...
    if f is None:
        return functools.partial(grad, argnums=argnums)

    # Ensure that argnums is a list because it simplifies the code further down. We
    # still have to keep information about whether the argnums was an integer or a
    # sequence to know whether we should return a single item or a tuple of items. If
    # argnums is a sequence of integers, the returned gradient is a tuple of values
    # with the same shapes and types as the corresponding arguments.
    is_single_argnum = isinstance(argnums, int)
    argnums = [argnums] if isinstance(argnums, int) else list(argnums)

    def outer(*args):
        # Extract only the arguments that are to be differentiated.
        grad_args = [arg for i, arg in enumerate(args) if i in argnums]
        # Flatten it down to values that are understood by the backend.
        flattened_grad_args = as_backend_arrays(tree.flatten(grad_args))

        # Define the function that accepts the flattened values and wrap it with grad.
        # We pass only the values that are to be differentiated to this function, so
        # argnums is set to be all arguments.
        @functools.partial(
            ops.backend.grad,
            argnums=range(len(flattened_grad_args.leaves)),
        )
        def inner(*args_inner):
            # Unflatten the arguments used to calculate the gradient back to their
            # original types and structure.
            grad_args_inner = flattened_grad_args.treedef.unflatten(args_inner)

            # Get the full list of args by inserting the unflattened args at the right
            # positions.
            args_inner = list(args)
            for i, arg in zip(argnums, grad_args_inner):
                args_inner[i] = arg

            # Call the function that will be differentiated.
            result = f(*args_inner)

            # The result must be a scalar float value.
            if not isinstance(result, (float, array)):
                raise ValueError(
                    "Gradient only defined for scalar-output functions. Got output "
                    f"with type: {type(result)!r}."
                )
            if isinstance(result, array) and not result.ndim == 0:
                raise ValueError(
                    "Gradient only defined for scalar-output functions. Got output "
                    f"with shape: {result.shape!r}"
                )

            # Return the backend data of the result which must be a scalar value.
            if isinstance(result, array):
                result = result.data
            return result

        # Get the resulting gradient and unflatten to the original type and structure.
        # The call to unflatten is what allows us to take the gradient with regards to
        # an arbitrary structure of arguments; even Modules.
        result = inner(*flattened_grad_args.leaves)
        result = flattened_grad_args.treedef.unflatten(result)

        # If argnums was a sequence of integers, the returned gradient is a tuple of
        # values with the same shapes and types as the corresponding arguments.
        # Otherwise, the result is just a single item and we return it.
        if is_single_argnum:
            assert len(result) == 1
            result = result[0]
        return result

    return outer


def value_and_grad(f=None, /, argnums: int | Sequence[int] = 0):
    """Create a function that evaluates both `f` and the gradient of ``f``.

      f: Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, standard Python containers or spekk
        Modules. Argument arrays in the positions specified by ``argnums`` must be of
        inexact (i.e., floating-point or complex) type. It should return a scalar (which
        includes arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).

    Returns:
      A function with the same arguments as ``f``, that evaluates both ``f``and the
      gradient of ``f``. If ``argnums`` is an integer then the gradient has the same
      shape and type as the positional argument indicated by that integer. If argnums
      is a tuple of integers, the gradient is a tuple of values with the same shapes
      and types as the corresponding arguments.
    """

    # Allow the following syntax:
    #   @value_and_grad(argnums=1)
    #   def f(a, b): ...
    # which is shorthand for:
    #   @functools.partial(value_and_grad, argnums=1)
    #   def f(a, b): ...
    if f is None:
        return functools.partial(value_and_grad, argnums=argnums)

    # Ensure that argnums is a list because it simplifies the code further down. We
    # still have to keep information about whether the argnums was an integer or a
    # sequence to know whether we should return a single item or a tuple of items. If
    # argnums is a sequence of integers, the returned gradient is a tuple of values
    # with the same shapes and types as the corresponding arguments.
    is_single_argnum = isinstance(argnums, int)
    argnums = [argnums] if isinstance(argnums, int) else list(argnums)

    def outer(*args):
        # Extract only the arguments that are to be differentiated.
        grad_args = [arg for i, arg in enumerate(args) if i in argnums]
        # Flatten it down to values that are understood by the backend.
        flattened_grad_args = as_backend_arrays(tree.flatten(grad_args))

        # Define the function that accepts the flattened values and wrap it with grad.
        # We pass only the values that are to be differentiated to this function, so
        # argnums is set to be all arguments.
        @functools.partial(
            ops.backend.value_and_grad,
            argnums=range(len(flattened_grad_args.leaves)),
        )
        def inner(*args_inner):
            # Unflatten the arguments used to calculate the gradient back to their
            # original types and structure.
            grad_args_inner = flattened_grad_args.treedef.unflatten(args_inner)

            # Get the full list of args by inserting the unflattened args at the right
            # positions.
            args_inner = list(args)
            for i, arg in zip(argnums, grad_args_inner):
                args_inner[i] = arg

            # Call the function that will be differentiated.
            result = f(*args_inner)

            # The result must be a scalar float value.
            if not isinstance(result, (float, array)):
                raise ValueError(
                    "Gradient only defined for scalar-output functions. Got output "
                    f"with type: {type(result)!r}."
                )
            if isinstance(result, array) and not result.ndim == 0:
                raise ValueError(
                    "Gradient only defined for scalar-output functions. Got output "
                    f"with shape: {result.shape!r}"
                )

            # Return the backend data of the result which must be a scalar value.
            if isinstance(result, array):
                result = result.data
            return result

        # Get the resulting gradient and unflatten to the original type and structure.
        # The call to unflatten is what allows us to take the gradient with regards to
        # an arbitrary structure of arguments; even Modules.
        value, grad = inner(*flattened_grad_args.leaves)
        assert value.ndim == 0  # We check for this in the inner function as well
        value = ops.array(value)
        grad = flattened_grad_args.treedef.unflatten(grad)

        # If argnums was a sequence of integers, the returned gradient is a tuple of
        # values with the same shapes and types as the corresponding arguments.
        # Otherwise, the result is just a single item and we return it.
        if is_single_argnum:
            assert len(grad) == 1
            grad = grad[0]
        return value, grad

    return outer


def wrap_backend_decorator(decorator):
    """Wraps a backend decorator so that it accepts spekk arrays and Modules."""

    def new_decorator(f=None, /, **decorator_kwargs):
        # Allow the following syntax:
        #   wrapped_decorator = wrap_backend_decorator(my_decorator)
        #   @wrapped_decorator(argnums=1)
        #   def f(a, b): ...
        # which is shorthand for:
        #   @functools.partial(wrapped_decorator, argnums=1)
        #   def f(a, b): ...
        if f is None:
            return functools.partial(new_decorator, **decorator_kwargs)

        def outer(*args, **kwargs):
            # Flatten the arguments down to values that are understood by the backend.
            args_dynamic, args_treedef = as_backend_arrays(tree.flatten((args, kwargs)))
            # flattened_output gets defined inside inner function.
            output_treedef: tree.TreeDef | tree.Leaf | tree.StaticLeaf = None  # type: ignore

            @functools.partial(decorator, **decorator_kwargs)
            def inner(inner_args):
                nonlocal output_treedef
                inner_args, inner_kwargs = args_treedef.unflatten(inner_args)
                result = f(*inner_args, **inner_kwargs)
                output_dynamic, output_treedef = as_backend_arrays(tree.flatten(result))
                return output_dynamic

            inner_result = inner(args_dynamic)
            return output_treedef.unflatten(inner_result)

        return outer

    return new_decorator


checkpoint = wrap_backend_decorator(backend.checkpoint)
