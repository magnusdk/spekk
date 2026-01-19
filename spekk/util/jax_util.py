import functools

from spekk.module import flatten


def make_jaxpr(f):
    import jax

    @functools.wraps(f)
    def wrapped_outer(*original_args, **original_kwargs):
        outer_dynamic, outer_treedef = flatten(
            (original_args, original_kwargs),
            flatten_spekk_arrays=True,
        )

        @jax.make_jaxpr
        def wrapped_inner(*flattened_args):
            args, kwargs = outer_treedef.unflatten(flattened_args)
            result = f(*args, **kwargs)
            inner_dynamic, _ = flatten(result, flatten_spekk_arrays=True)
            return inner_dynamic

        return wrapped_inner(*outer_dynamic)

    return wrapped_outer
