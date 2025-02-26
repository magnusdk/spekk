import functools

from spekk.module import flatten


def make_jaxpr(f):
    import jax

    @functools.wraps(f)
    def wrapped_outer(*original_args, **original_kwargs):
        flattened_args_outer = flatten(
            (original_args, original_kwargs),
            flatten_spekk_arrays=True,
        )

        @jax.make_jaxpr
        def wrapped_inner(*flattened_args):
            args, kwargs = flattened_args_outer.unflatten(flattened_args)
            result = f(*args, **kwargs)
            flattened_result = flatten(
                result,
                flatten_spekk_arrays=True,
            )
            return flattened_result.dynamic

        return wrapped_inner(*flattened_args_outer.dynamic)

    return wrapped_outer
