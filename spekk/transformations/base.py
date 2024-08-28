"""The base classes and abstract classes for working with :class:`Transformation`."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

from spekk import Spec
from spekk.transformations.common import get_fn_name

TBuildableSelf = TypeVar("TBuildableSelf", bound="Buildable")


class Buildable(ABC):
    """An abstract class representing something that can be "built" with a
    :class:`Spec` to get information about how the spec changes when transforming
    functions.

    Args:
        input_spec (Spec): The spec that is passed into the function.
        passed_spec (Spec): The spec that is passed down into the wrapped function
            after the transformation has been applied.
        returned_spec (Spec): The spec that is returned up from the wrapped function.
        output_spec (Spec): The final spec that is returned after calling the
            transformed function.
    """

    input_spec: Optional[
        Spec
    ] = None  #: The spec for the input kwargs for the function.
    passed_spec: Optional[
        Spec
    ] = None  #: The spec for the input kwargs that are passed down into the wrapped function after the transformation has been applied.
    returned_spec: Optional[
        Spec
    ] = None  #: The spec of the returned value from the wrapped function.
    output_spec: Optional[
        Spec
    ] = None  #: The spec of the final returned value from the transformed function.

    @abstractmethod
    def build(self: TBuildableSelf, input_spec: Spec) -> TBuildableSelf:
        """Return a copy of this object with information about what happens to the spec
        when the function is called."""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + f"input_spec={self.input_spec}, "
            + f"passed_spec={self.passed_spec}, "
            + f"returned_spec={self.returned_spec}, "
            + f"output_spec={self.output_spec})"
        )


def _transformed_function_repr_fn(
    tf: "TransformedFunction",
    step_repr_fn: Optional[Callable[["TransformedFunction", str], str]] = None,
):
    "Format a TransformedFunction in a nice way."
    s = f"TransformedFunction(\n  <compose(\n"
    for step in tf.traverse(depth_first=True):
        step_repr = (
            repr(step.transformation)
            if isinstance(step, TransformedFunction)
            else get_fn_name(step)
        )
        if step_repr_fn is not None:
            step_repr = step_repr_fn(step, step_repr)
        else:
            step_repr = f"    {step_repr},\n"
        s += step_repr
    s += "  )>\n)"
    return s


@dataclass
class TransformedFunctionError(Exception):
    """An error that occurred while calling a :class:`TransformedFunction`.

    It will print out the nested steps vertically and highlight the step that raised an
    exception along with the original exception."""

    original_exception: Exception
    transformed_function: "TransformedFunction"
    error_step: Union[callable, "TransformedFunction"]

    def __post_init__(self):
        if isinstance(self.transformed_function, TransformedFunction):
            s = _transformed_function_repr_fn(
                self.transformed_function,
                lambda step, default_repr: f"⚠   {default_repr},\n"
                + f"⚠     ↳ This step raised {repr(self.original_exception)}\n"
                if step is self.error_step
                else f"    {default_repr},\n",
            )
            s = f"{str(self.original_exception)}\n" + s
        else:
            s = str(self.original_exception)
        super().__init__(s)


@dataclass
class _WrappedWithErrorHandling:
    """A wrapper around a function that catches errors and re-raises them as
    :class:`TransformedFunctionError`s.

    It is important to re-raise exceptions as :class:`TransformedFunctionError`s
    because then we can give users more information about where the error occurred in
    a sequence of transformations.

    See also:
        :class:`TransformedFunctionError`"""

    f: callable

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except Exception as e:
            raise TransformedFunctionError(e, self.f, self.f) from e


class _NoSpecGiven(Spec):
    """A dummy spec that raises an error when it is used.

    It is used to get better error messages when calling a :class:`TransformedFunction`
    that hasnot been built with a :class:`Spec`."""

    def __getattribute__(self, _: str) -> Any:
        raise ValueError(
            f"The Transformation tried to use the spec, but no spec was given. Did you \
forget to call build()?"
        )


@dataclass
class TransformedFunction(Buildable):
    wrapped_fn: Union[
        callable, "TransformedFunction"
    ]  #: The original function that was wrapped by the transformation.
    transformation: "Transformation"  #: The :class:`Transformation` that was applied to the wrapped function.

    def __call__(self, *args, **kwargs):
        try:
            # Handle the case where the function has not been built yet. If any
            # transformation requires/uses a spec, and this object has not been built
            # with a spec, it will raise an error.
            input_spec = self.input_spec
            returned_spec = self.returned_spec
            if input_spec is None:
                input_spec = _NoSpecGiven()
            if returned_spec is None:
                returned_spec = _NoSpecGiven()

            # Handle special case where the wrapped function is not a
            # TransformedFunction (e.g. it's the kernel function)
            wrapped_fn = self.wrapped_fn
            if not isinstance(wrapped_fn, TransformedFunction):
                wrapped_fn = _WrappedWithErrorHandling(wrapped_fn)

            # Perform the actual transformation
            transformed_wrapped_function = self.transformation.transform_function(
                wrapped_fn, input_spec, returned_spec
            )

            # Return the result of calling the transformed function
            return transformed_wrapped_function(*args, **kwargs)

        # Handle errors that can occur while running the transformed function
        except TransformedFunctionError as e:
            # A nested step raised an error. Let's add this step to the stack trace
            raise TransformedFunctionError(
                e.original_exception, self, e.error_step
            ) from e.original_exception
        except Exception as e:
            # An exception was raised in this step.
            raise TransformedFunctionError(e, self, self) from e

    def build(self, input_spec: Spec) -> "TransformedFunction":
        try:
            # The spec that will be passed into the wrapped function:
            passed_spec = self.transformation.transform_input_spec(input_spec)

            # Recursively build the wrapped function
            wrapped_fn = self.wrapped_fn
            if isinstance(wrapped_fn, Buildable):
                wrapped_fn = wrapped_fn.build(passed_spec)
                # The spec after the wrapped function has been called:
                returned_spec = wrapped_fn.output_spec
            else:
                # By default, we assume that the wrapped function returns a scalar.
                returned_spec = Spec(())

            # Lastly, transform the output spec according to the Transformation.
            output_spec = self.transformation.transform_output_spec(returned_spec)

            # Return a new copy with the updated specs
            copy = TransformedFunction(wrapped_fn, self.transformation)
            copy.input_spec = input_spec
            copy.passed_spec = passed_spec
            copy.returned_spec = returned_spec
            copy.output_spec = output_spec
            return copy

        # Handle errors that can occur while building
        except TransformedFunctionError as e:
            # A nested step raised an error. Let's reraise the exception.
            raise TransformedFunctionError(
                e.original_exception, self, e.error_step
            ) from e.original_exception
        except Exception as e:
            # An exception was raised in this step.
            raise TransformedFunctionError(e, self, self) from e

    def traverse(self, *, depth_first: bool = False):
        """Recursively yield the potentially nested :class:`TransformedFunction`.

        >>> from spekk.transformations import ForAll, compose, TransformedFunction
        >>> tf = compose(abs, ForAll("x"), ForAll("y"))
        >>> for t in tf.traverse():
        ...     if isinstance(t, TransformedFunction):
        ...         t = t.transformation  # Print transformation, not TransformedFunction
        ...     print(repr(t))
        ForAll("y")
        ForAll("x")
        <built-in function abs>
        """
        if not depth_first:
            yield self
        if isinstance(self.wrapped_fn, TransformedFunction):
            yield from self.wrapped_fn.traverse(depth_first=depth_first)
        else:
            yield self.wrapped_fn
        if depth_first:
            yield self

    def __repr__(self):
        return _transformed_function_repr_fn(self)

    def __hash__(self):
        return object.__hash__(self)


class Transformation(ABC):
    """A :class:`Transformation` takes a function and transforms/modifies it. It also
    keeps track of changes to the spec when transforming the function. See the
    schematic in :mod:`spekk.transformations`'s docstring for a visualization of how specs are processed.

    A :class:`Transformation` acts as a higher-order function that takes in a function
    and returns an instance of :class:`TransformedFunction`. It gets information about
    the dimensions of the data (through instances of :class:`Spec`) that the function
    operates on and it also defines what happens to the spec when the function is
    transformed.

    Attributes:
        transform_function: The main transformation function that returns a
            :class:`TransformedFunction`.
        transform_input_spec: A function that returns the spec that is passed down into
            the wrapped function after the transformation has been applied.
        transform_output_spec: A function that returns the final returned spec of the
            transformed function.
    """

    def __call__(self, wrapped_fn: callable) -> TransformedFunction:
        "Transform the wrapped function."
        if isinstance(wrapped_fn, PartialTransformation):
            return PartialTransformation([*wrapped_fn.partial_transformations, self])
        elif isinstance(wrapped_fn, Transformation):
            # Handle partial application of transformations.
            return PartialTransformation([wrapped_fn, self])
        return TransformedFunction(wrapped_fn, self)

    @abstractmethod
    def transform_function(
        self, wrapped_fn: callable, input_spec: Spec, output_spec: Spec
    ) -> TransformedFunction:
        """Transform the wrapped function given the spec of the input arguments and the
        spec of the returned value of the wrapped function.
        """

    @abstractmethod
    def transform_input_spec(self, spec: Spec) -> Spec:
        """Return a new spec that represent the input arguments that are passed down to
        the wrapped function after the transformation has been applied.

        For example, if the transformation vectorizes the wrapped function over a
        dimension, the wrapped function would only see single items of the dimension at
        a time. Therefore, the input spec will have one less dimension when passed down
        to the wrapped function.
        """

    @abstractmethod
    def transform_output_spec(self, spec: Spec) -> Spec:
        """Return a new spec that represent the returned value of the final transformed
        function.

        For example, if the transformation sums over a dimension of the result of
        calling the wrapped function, the output spec will have one less dimension.
        """


@dataclass
class PartialTransformation(Transformation):
    """A partially applied transformation.

    :class:`PartialTransformation` lets us compose transformations such that they can
    be re-used as building blocks. For example, in the below code, ``tf_partial`` and
    ``tf_full`` are equivalent:

    >>> import numpy as np
    >>> from spekk.transformations import ForAll, compose

    Let's create an example kernel function along with some data and a spec:

    >>> kernel = lambda x: x**2
    >>> data = {"x": np.ones((2, 3)) * 2}
    >>> spec = Spec({"x": ["a", "b"]})

    Create a partial transformation by composing two transformations:

    >>> forall_xy = compose(ForAll("a"), ForAll("b"))

    Create and build two equivalent transformed functions:

    >>> tf_partial = compose(kernel, forall_xy).build(spec)
    >>> tf_full = compose(kernel, ForAll("a"), ForAll("b")).build(spec)

    The two transformed functions are equivalent:

    >>> np.array_equal(tf_partial(**data), tf_full(**data))
    True
    """

    partial_transformations: Sequence[Transformation]

    def __call__(self, wrapped_fn: callable) -> TransformedFunction:
        for t in self.partial_transformations:
            if isinstance(t, PartialTransformation):
                # Makes sure that nested partial transformations are flattened
                wrapped_fn = t(wrapped_fn)
            else:
                wrapped_fn = TransformedFunction(wrapped_fn, t)
        return wrapped_fn

    def transform_function(
        self, wrapped_fn: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        "Transform the wrapped function by applying each partial transformation in turn."
        for t in self.partial_transformations:
            wrapped_fn = t.transform_function(wrapped_fn, input_spec, output_spec)
            input_spec = t.transform_input_spec(input_spec)
            output_spec = t.transform_output_spec(output_spec)
        return wrapped_fn

    def transform_input_spec(self, spec: Spec) -> Spec:
        "Transform the input spec using each partial transformation in turn."
        for t in self.partial_transformations:
            spec = t.transform_input_spec(spec)
        return spec

    def transform_output_spec(self, spec: Spec) -> Spec:
        "Transform the output spec using each partial transformation in turn."
        for t in self.partial_transformations:
            spec = t.transform_output_spec(spec)
        return spec

    def __repr__(self):
        return f"PartialTransformation({self.partial_transformations})"


@dataclass
class Specced(Buildable):
    """A wrapper around ``f`` that has information about what happens to the spec when
    the function is called."""

    f: callable  #: The function to wrap.
    transform_spec: Callable[
        [Spec], Spec
    ]  #: A function that transforms the spec, returning the spec of the return-value of ``f``.

    def __post_init__(self):
        if isinstance(self.transform_spec, Spec):
            self.transform_spec = lambda _: self.transform_spec

    def build(self, input_spec: Spec) -> "Specced":
        output_spec = self.transform_spec(input_spec)
        if not isinstance(output_spec, Spec):
            output_spec = Spec(output_spec)

        new_obj = Specced(self.f, self.transform_spec)
        new_obj.input_spec = input_spec
        new_obj.output_spec = output_spec
        return new_obj

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __hash__(self):
        return hash((self.f, self.transform_spec))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
