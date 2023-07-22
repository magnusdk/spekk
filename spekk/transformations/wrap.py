""":class:`Wrap` is a simple :class:`Transformation` that wraps a function with another 
function, such as :func:`jax.jit`."""

from spekk import Spec
from spekk.transformations import Transformation, common


class Wrap(Transformation):
    """Simply wraps a function with another function, but useful for keeping 
    information about the spec in a chain of :class:`Transformation`.
    
    Attributes:
        f: A wrapper function, for example :func:`jax.jit`.
        args: Optional extra positional arguments to pass to ``f``.
        kwargs: Optional extra keyword arguments to pass to ``f``.

    Example:
        >>> import jax
        >>> my_fn = lambda x: x**2
        >>> wrapped_fn1 = Wrap(jax.jit)(my_fn)
        >>> wrapped_fn2 = jax.jit(my_fn)

        `wrapped_fn1` and `wrapped_fn2` are equivalent, but `wrapped_fn1` will propagate 
        information about the spec (if applicable) to nested :class:`Transformation`.
    """
    def __init__(self, f: callable, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def transform_function(
        self, to_be_wrapped: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        return self.f(to_be_wrapped, *self.args, **self.kwargs)

    def transform_input_spec(self, spec: Spec) -> Spec:
        return spec

    def transform_output_spec(self, spec: Spec) -> Spec:
        return spec

    def __repr__(self) -> str:
        args_str = ", ".join([str(arg) for arg in self.args])
        kwargs_str = ", ".join([f"{k}={str(v)}" for k, v in self.kwargs.items()])
        repr_str = f"Wrap({common.get_fn_name(self.f)}"
        if self.args:
            repr_str += f", {args_str}"
        if self.kwargs:
            repr_str += f", {kwargs_str}"
        # Make sure the repr string is not too long
        if len(repr_str) > 140:
            repr_str = repr_str[: (140 - len("… <truncated>"))] + "… <truncated>"
        return repr_str + ")"
