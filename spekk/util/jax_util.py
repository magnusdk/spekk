import functools

from spekk.module import trees


def make_jaxpr(f):
    import jax

    @functools.wraps(f)
    def wrapped_outer(*original_args, **original_kwargs):
        flattened_args_outer = trees.flatten((original_args, original_kwargs))

        @jax.make_jaxpr
        def wrapped_inner(*flattened_args):
            args, kwargs = flattened_args_outer.treedef.unflatten(flattened_args)
            result = f(*args, **kwargs)
            flattened_result = trees.flatten(result)
            return flattened_result.dynamic

        return wrapped_inner(*flattened_args_outer.dynamic)

    return wrapped_outer
