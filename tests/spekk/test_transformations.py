import pytest

from spekk.spec import Spec
from spekk.transformations import (
    Apply,
    Axis,
    ForAll,
    TransformedFunctionError,
    compose,
)


def test_partial_transformations():
    def kernel(x):
        return x

    f1 = compose(kernel, ForAll("a"), ForAll("b"), ForAll("c"))
    partial_foralls = compose(ForAll("a"), ForAll("b"), ForAll("c"))
    f2 = compose(kernel, partial_foralls)
    partial_foralls = compose(ForAll("a"), compose(ForAll("b"), ForAll("c")))
    f3 = compose(kernel, partial_foralls)
    assert f1 == f2
    assert f2 == f3


def test_forall_specs():
    def kernel(x):
        return x

    spec = Spec({"x": ["b", "a"]})
    f = compose(
        kernel,
        ForAll("a"),
        ForAll("b"),
    ).build(spec)

    assert f.input_spec == Spec({"x": ["b", "a"]})
    assert f.passed_spec == Spec({"x": ["a"]})
    assert f.returned_spec == Spec(["a"])
    assert f.output_spec == Spec(["b", "a"])

    f_wrapped = f.original_f
    assert f_wrapped.input_spec == Spec({"x": ["a"]})
    assert f_wrapped.passed_spec == Spec({})
    assert f_wrapped.returned_spec == Spec([])
    assert f_wrapped.output_spec == Spec(["a"])


def test_building():
    def kernel(x):
        return x

    # We don't have to build the TransformedFunction if it doesn't use any spec info
    f = compose(kernel, Apply(lambda x: x + 1))
    assert f(1) == 2

    # However, if we try to use spec info, like Axis(dim), we have to build the
    # function first.
    with pytest.raises(TransformedFunctionError):
        f = compose(kernel, Apply(lambda x, _: x + 1, Axis("a")))
        f(1)

    # Providing the spec via build() fixes this.
    f = compose(kernel, Apply(lambda x, _: x + 1, Axis("a"))).build(Spec(["a"]))
    assert f(1) == 2


def sub1(x):
    return x - 1


def residual(x):
    return 1 / x


def add1(x):
    return x - 1


def test_raise_exception():
    kernel = lambda: 2
    with pytest.raises(TransformedFunctionError) as e:
        f = compose(
            kernel,
            Apply(sub1),
            Apply(sub1),
            Apply(residual),  # This should raise ZeroDivisionError
            Apply(add1),
            Apply(add1),
        )
        f()


if __name__ == "__main__":
    pytest.main([__file__])
