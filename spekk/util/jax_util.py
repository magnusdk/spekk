import functools

from spekk import tree


def make_jaxpr(f):
    import jax

    @functools.wraps(f)
    def wrapped_outer(*original_args, **original_kwargs):
        from spekk.ops._util import as_backend_arrays

        outer_dynamic, outer_treedef = as_backend_arrays(
            *tree.flatten((original_args, original_kwargs))
        )

        @jax.make_jaxpr
        def wrapped_inner(*flattened_args):
            args, kwargs = outer_treedef.unflatten(flattened_args)
            result = f(*args, **kwargs)
            inner_dynamic, _ = as_backend_arrays(*tree.flatten(result))
            return inner_dynamic

        return wrapped_inner(*outer_dynamic)

    return wrapped_outer
