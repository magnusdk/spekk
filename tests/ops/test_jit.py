from spekk import ops


def test_static_args():
    with ops.backend.temporary_backend("jax"):
        n_compilations = 0

        @ops.jit(static_argnums=[0], static_argnames=["b"])
        def f(a: int, *, b: int):
            nonlocal n_compilations
            n_compilations += 1
            return ops.arange(a) + ops.sum(ops.arange(b))

        f(1, b=2)  # Triggers compilation
        f(1, b=2)
        assert n_compilations == 1

        f(2, b=2)  # Triggers compilation (a changed)
        f(2, b=2)
        assert n_compilations == 2

        f(2, b=3)  # Triggers compilation (b changed)
        f(2, b=3)
        assert n_compilations == 3
