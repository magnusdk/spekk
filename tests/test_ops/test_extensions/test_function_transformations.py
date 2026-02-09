import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from spekk import Module, field, ops
from spekk.ops.extensions.function_transformations import (
    _argf_over_dims,
    grad,
    jit,
    map_over_dim,
    map_reduce_over_dim,
    reduce_over_dim,
    scan,
    value_and_grad,
)

# -- Shared fixtures ----------------------------------------------------------

DIMS = ["a", "b", "c"]


def make_array(*shape, dims=DIMS):
    """Create a named float array with the given shape."""
    return ops.reshape(ops.arange(np.prod(shape), dtype=float), shape, dims=dims)


class Data(Module):
    a: ops.array
    b: ops.array


class Carry(Module):
    total: ops.array


# -- Hypothesis strategies -----------------------------------------------------

ALL_DIMS = ["a", "b", "c", "d", "e"]


@st.composite
def array_and_axes(draw):
    """Generate a random named array and a random non-empty subset of its dims."""
    n_dims = draw(st.integers(min_value=1, max_value=4))
    dims = list(draw(st.permutations(ALL_DIMS)))[:n_dims]
    shape = [draw(st.integers(min_value=2, max_value=5)) for _ in range(n_dims)]
    n_elems = int(np.prod(shape))
    flat = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=n_elems,
            max_size=n_elems,
        )
    )
    x = ops.array(np.array(flat, dtype=np.float32).reshape(shape), dims)
    n_axes = draw(st.integers(min_value=1, max_value=n_dims))
    axes = list(draw(st.permutations(dims)))[:n_axes]
    return x, axes


# -- Tests: _argf_over_dims ---------------------------------------------------


@given(data=array_and_axes())
@settings(max_examples=50, deadline=None)
def test_argf_over_dims_hypothesis(data):
    x, axes = data
    result = _argf_over_dims(ops.argmin, x, axis=axes)
    expected = ops.min(x, axis=axes)
    assert ops.all(x[result] == expected)


def test_argf_over_dims_all_axes():
    x = make_array(2, 3, 4)
    result = _argf_over_dims(ops.argmin, x, axis=None)
    assert float(x[result]) == float(ops.min(x))


# -- Tests: jit ---------------------------------------------------------------


def test_jit_basic():
    with ops.backend.temporary_backend("jax"):
        f = jit(lambda x: x * 2)
        x = ops.arange(6, dtype=float, dim="a")
        assert ops.all(f(x) == x * 2)


def test_jit_module_recompilation():
    with ops.backend.temporary_backend("jax"):

        class Config(Module):
            x: ops.array
            mode: str = field(static=True)

        n = 0

        @jit
        def f(c: Config):
            nonlocal n
            n += 1
            if c.mode == "double":
                return c.x * 2
            return c.x + 1

        x = ops.arange(4, dtype=float, dim="a")
        f(Config(x, "double"))
        f(Config(x, "double"))
        assert n == 1  # no recompilation

        f(Config(x, "add"))
        assert n == 2  # recompilation


def test_jit_decorator_forms():
    with ops.backend.temporary_backend("jax"):

        @jit
        def f(x):
            return x + 1

        @jit(static_argnums=[0])
        def g(n, x):
            return x + n

        x = ops.arange(3, dtype=float, dim="a")
        assert ops.all(f(x) == x + 1)
        assert ops.all(g(10, x) == x + 10)


# -- Tests: reduce_over_dim ---------------------------------------------------


def test_reduce_over_dim_basic():
    x = make_array(3, 4, dims=["a", "b"])
    result = reduce_over_dim(
        lambda carry, xi: carry + xi, x, init=ops.zeros((4,), dims=["b"]), dim="a"
    )
    assert ops.all(result == ops.sum(x, axis="a"))


def test_reduce_over_dim_module():
    a = ops.reshape(ops.arange(12, dtype=float), (3, 4), dims=["a", "b"])
    b = ops.reshape(ops.arange(12, 24, dtype=float), (3, 4), dims=["a", "b"])
    data = Data(a, b)
    init = Carry(ops.zeros((4,), dims=["b"]))

    def reduce_f(carry: Carry, d: Data) -> Carry:
        return Carry(carry.total + d.a + d.b)

    result = reduce_over_dim(reduce_f, data, init=init, dim="a")
    expected = ops.sum(a, axis="a") + ops.sum(b, axis="a")
    assert isinstance(result, Carry)
    assert ops.all(result.total == expected)


# -- Tests: map_reduce_over_dim -----------------------------------------------


def test_map_reduce_over_dim_basic():
    x = make_array(3, 4, dims=["a", "b"])
    result = map_reduce_over_dim(
        lambda xi: xi**2,
        lambda carry, xi: carry + xi,
        x,
        init=ops.zeros((4,), dims=["b"]),
        dim="a",
    )
    assert ops.all(result == ops.sum(x**2, axis="a"))


def test_map_reduce_over_dim_module():
    a = ops.reshape(ops.arange(12, dtype=float), (3, 4), dims=["a", "b"])
    b = ops.reshape(ops.arange(12, 24, dtype=float), (3, 4), dims=["a", "b"])
    data = Data(a, b)
    init = Carry(ops.zeros((4,), dims=["b"]))

    def map_f(d: Data) -> Carry:
        return Carry(d.a * d.b)

    def reduce_f(carry: Carry, mapped: Carry) -> Carry:
        return Carry(carry.total + mapped.total)

    result = map_reduce_over_dim(map_f, reduce_f, data, init=init, dim="a")
    expected = ops.sum(a * b, axis="a")
    assert isinstance(result, Carry)
    assert ops.all(result.total == expected)


# -- Tests: map_over_dim ------------------------------------------------------


def test_map_over_dim_basic():
    x = make_array(3, 4, dims=["a", "b"])
    result = map_over_dim(lambda xi: xi * 2, x, dim="a")
    assert ops.all(result == x * 2)


def test_map_over_dim_module():
    a = ops.reshape(ops.arange(12, dtype=float), (3, 4), dims=["a", "b"])
    b = ops.reshape(ops.arange(12, 24, dtype=float), (3, 4), dims=["a", "b"])
    data = Data(a, b)

    def map_f(d: Data) -> Carry:
        return Carry(d.a + d.b)

    result = map_over_dim(map_f, data, dim="a")
    assert isinstance(result, Carry)
    assert ops.all(result.total == a + b)


# -- Tests: scan --------------------------------------------------------------


def test_scan_basic():
    """Scan should accumulate carry and stack outputs."""
    xs = ops.arange(5, dtype=float, dim="t")

    def scan_f(carry, x):
        new_carry = carry + x
        output = new_carry * 2
        return new_carry, output

    final_carry, outputs = scan(scan_f, ops.array(0.0), xs)

    # Final carry should be sum of all xs: 0+1+2+3+4 = 10
    assert float(final_carry) == 10.0

    # Outputs should be cumsum * 2: [0, 2, 6, 12, 20]
    expected_outputs = ops.array([0.0, 2.0, 6.0, 12.0, 20.0], dims=["t"])
    assert ops.all(outputs == expected_outputs)


def test_scan_module_carry():
    """Scan should work with Module as carry state."""
    xs = ops.arange(4, dtype=float, dim="t")

    def scan_f(carry: Carry, x: ops.array) -> tuple[Carry, ops.array]:
        new_total = carry.total + x
        return Carry(new_total), new_total

    init = Carry(ops.array(0.0))
    final_carry, outputs = scan(scan_f, init, xs)

    assert isinstance(final_carry, Carry)
    assert float(final_carry.total) == 6.0  # 0+1+2+3

    expected = ops.array([0.0, 1.0, 3.0, 6.0], dims=["t"])
    assert ops.all(outputs == expected)


def test_scan_module_output():
    """Scan should work with Module as output."""
    xs = ops.arange(3, dtype=float, dim="t")

    def scan_f(carry: float, x: ops.array) -> tuple[float, Carry]:
        new_carry = carry + float(x)
        return new_carry, Carry(x * 2)

    final_carry, outputs = scan(scan_f, 0.0, xs)

    assert final_carry == 3.0  # 0+1+2
    assert isinstance(outputs, Carry)
    expected = ops.array([0.0, 2.0, 4.0], dims=["t"])
    assert ops.all(outputs.total == expected)


# -- Tests: grad --------------------------------------------------------------


def test_grad_basic():
    with ops.backend.temporary_backend("jax"):
        x = ops.arange(6, dtype=float, dim="a")
        g = grad(lambda x: ops.sum(x**2))(x)
        assert ops.all(g == 2 * x)


def test_grad_module():
    with ops.backend.temporary_backend("jax"):
        a = ops.arange(4, dtype=float, dim="a")
        b = ops.arange(4, dtype=float, dim="a") + 1
        data = Data(a, b)

        g = grad(lambda d: ops.sum(d.a**2 + d.b**2))(data)
        assert isinstance(g, Data)
        assert ops.all(g.a == 2 * a)
        assert ops.all(g.b == 2 * b)


# -- Tests: value_and_grad ----------------------------------------------------


def test_value_and_grad_basic():
    with ops.backend.temporary_backend("jax"):
        x = ops.arange(6, dtype=float, dim="a")
        val, g = value_and_grad(lambda x: ops.sum(x**2))(x)
        assert float(val) == float(ops.sum(x**2))
        assert ops.all(g == 2 * x)


def test_value_and_grad_module():
    with ops.backend.temporary_backend("jax"):
        a = ops.arange(4, dtype=float, dim="a")
        b = ops.arange(4, dtype=float, dim="a") + 1
        data = Data(a, b)

        val, g = value_and_grad(lambda d: ops.sum(d.a**2 + d.b**2))(data)
        assert float(val) == float(ops.sum(a**2 + b**2))
        assert isinstance(g, Data)
        assert ops.all(g.a == 2 * a)
        assert ops.all(g.b == 2 * b)


# -- Tests: jit static args --------------------------------------------------


def test_jit_static_args():
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


def test_jit_recompiles_for_different_module_types():
    """Regression: different Module types with identical field names must trigger recompilation."""
    with ops.backend.temporary_backend("jax"):

        class A(Module):
            a: float = field(static=True)

        class B(Module):
            a: float = field(static=True)

        n = 0

        @jit
        def f(obj):
            nonlocal n
            n += 1
            return ops.full((), obj.a)

        f(A(1.0))
        f(A(1.0))
        assert n == 1

        f(B(1.0))  # Different type, same field value — must recompile
        assert n == 2


def test_jit_with_unhashable_static_field():
    """Regression: static fields with unhashable types (e.g. dict) must not crash."""
    with ops.backend.temporary_backend("jax"):

        class C(Module):
            a: float
            b: dict = field(static=True)

        n = 0

        @jit
        def f(obj):
            nonlocal n
            n += 1
            return ops.full((), obj.a) + obj.b["offset"]

        f(C(1.0, {"offset": 2.0}))
        f(C(1.0, {"offset": 2.0}))
        assert n == 1  # no recompilation

        f(C(1.0, {"offset": 5.0}))  # static field changed — must recompile
        assert n == 2


def test_jit_string_arg_auto_static():
    """Strings are not arrays, so jit should automatically treat them as static."""
    with ops.backend.temporary_backend("jax"):
        n = 0

        @jit
        def f(mode: str, x):
            nonlocal n
            n += 1
            if mode == "double":
                return x * 2
            return x + 1

        x = ops.arange(4, dtype=float, dim="a")
        assert ops.all(f("double", x) == x * 2)
        f("double", x)
        assert n == 1

        assert ops.all(f("add", x) == x + 1)
        assert n == 2  # recompiled because string changed


# -- Tests: non-array leaves in function transformations ----------------------


class DataWithLabel(Module):
    values: ops.array
    label: str = field(static=True)


def test_reduce_over_dim_with_string_field():
    """reduce_over_dim should handle Modules with static string fields."""
    data = DataWithLabel(
        ops.reshape(ops.arange(12, dtype=float), (3, 4), dims=["a", "b"]),
        "test",
    )
    init = ops.zeros((4,), dims=["b"])
    result = reduce_over_dim(
        lambda carry, d: carry + d.values,
        data,
        init=init,
        dim="a",
    )
    assert ops.all(result == ops.sum(data.values, axis="a"))


def test_map_over_dim_with_string_field():
    """map_over_dim should handle Modules with static string fields."""
    data = DataWithLabel(
        ops.reshape(ops.arange(12, dtype=float), (3, 4), dims=["a", "b"]),
        "test",
    )
    result = map_over_dim(lambda d: d.values * 2, data, dim="a")
    assert ops.all(result == data.values * 2)


def test_map_reduce_over_dim_with_string_field():
    """map_reduce_over_dim should handle Modules with static string fields."""
    data = DataWithLabel(
        ops.reshape(ops.arange(12, dtype=float), (3, 4), dims=["a", "b"]),
        "test",
    )
    init = ops.zeros((4,), dims=["b"])
    result = map_reduce_over_dim(
        lambda d: d.values**2,
        lambda carry, mapped: carry + mapped,
        data,
        init=init,
        dim="a",
    )
    assert ops.all(result == ops.sum(data.values**2, axis="a"))


def test_scan_with_string_field():
    """scan should handle Modules with static string fields as carry."""
    xs = ops.arange(3, dtype=float, dim="a")
    init = DataWithLabel(ops.array(0.0), "test")

    def scan_f(carry, x):
        return DataWithLabel(carry.values + x, carry.label), carry.values + x

    final, outputs = scan(scan_f, init, xs)
    assert isinstance(final, DataWithLabel)
    assert float(final.values) == float(ops.sum(xs))


def test_grad_with_string_field():
    """grad should handle Modules with static string fields."""
    with ops.backend.temporary_backend("jax"):
        data = DataWithLabel(
            ops.arange(4, dtype=float, dim="a"),
            "test",
        )
        g = grad(lambda d: ops.sum(d.values**2))(data)
        assert isinstance(g, DataWithLabel)
        assert ops.all(g.values == 2 * data.values)


def test_value_and_grad_with_string_field():
    """value_and_grad should handle Modules with static string fields."""
    with ops.backend.temporary_backend("jax"):
        data = DataWithLabel(
            ops.arange(4, dtype=float, dim="a"),
            "test",
        )
        val, g = value_and_grad(lambda d: ops.sum(d.values**2))(data)
        assert float(val) == float(ops.sum(data.values**2))
        assert isinstance(g, DataWithLabel)
        assert ops.all(g.values == 2 * data.values)
