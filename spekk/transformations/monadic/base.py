from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Union

from spekk.spec import Spec


@dataclass
class Specced:
    f: Callable
    transform_spec: Callable[[Spec], Spec]

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __hash__(self):
        return hash((self.f, self.transform_spec))


class Transformation(ABC):
    @abstractmethod
    def transform_input_spec(self, spec: Spec):
        pass

    @abstractmethod
    def transform_output_spec(self, spec: Spec):
        pass

    @abstractmethod
    def transform_function(self, f: Callable, input_spec: Spec, returned_spec: Spec):
        pass

    def __call__(
        self, function: Union[Callable, "Transformation", "PartialTransformation"]
    ):
        if isinstance(function, (Transformation, PartialTransformation)):
            return PartialTransformation(self, function)
        return TransformedFunction(function, self)


@dataclass
class PartialTransformation:
    transformation: Transformation
    wrapped_transformation: Union[Transformation, "PartialTransformation"]

    def __call__(self, function):
        return self.transformation(self.wrapped_transformation(function))


class NotBuiltSpecError(Exception):
    pass


class _NotBuiltSpecGiven(Spec):
    def __init__(self):
        self._data = None

    @property
    def data(self) -> Any:
        raise NotBuiltSpecError(
            "The Transformation tried to use the spec, but no spec was given. "
            "Did you forget to call build()?"
        )


class TransformedFunctionError(Exception):
    def __init__(
        self, original_exception: Exception, raising_step: "TransformedFunction"
    ):
        self.original_exception = original_exception
        self.raising_step = raising_step
        self.transformed_function = raising_step

    def __str__(self):
        wrapped_fns = deque()
        for x in self.transformed_function._traverse():
            if x is self.raising_step:
                wrapped_fns.appendleft(
                    f"⚠   ↳ This step raised {repr(self.original_exception)}"
                )
                wrapped_fns.appendleft(
                    f"⚠   {repr(x.transformation if isinstance(x, TransformedFunction) else x)},"
                )
            else:
                wrapped_fns.appendleft(
                    f"    {repr(x.transformation if isinstance(x, TransformedFunction) else x)},"
                )
        steps_string = "\n".join(wrapped_fns)
        return f"TransformedFunction(\n  <compose(\n{steps_string}\n  )>\n)"


@dataclass
class TransformedFunction:
    original_f: Union["TransformedFunction", Specced, Callable]
    transformation: Transformation

    # Information about spec before and after transformation. Use the build() method to
    # populate the fields.
    input_spec: Spec = _NotBuiltSpecGiven()
    passed_spec: Spec = _NotBuiltSpecGiven()
    returned_spec: Spec = _NotBuiltSpecGiven()
    output_spec: Spec = _NotBuiltSpecGiven()

    def __call__(self, *args, **kwargs):
        try:
            transformed_function = self.transformation.transform_function(
                self.original_f, self.input_spec, self.returned_spec
            )
            return transformed_function(*args, **kwargs)
        except TransformedFunctionError as e:
            e.transformed_function = self
            e.__traceback__ = None
            raise
        except Exception as e:
            raise TransformedFunctionError(e, self) from e

    def build(self, input_spec: Spec) -> "TransformedFunction":
        self.input_spec = input_spec
        self.passed_spec = self.transformation.transform_input_spec(self.input_spec)
        if isinstance(self.original_f, TransformedFunction):
            self.original_f.build(self.passed_spec)
            self.returned_spec = self.original_f.output_spec
        elif isinstance(self.original_f, Specced):
            self.returned_spec = self.original_f.transform_spec(self.passed_spec)
        else:
            self.returned_spec = Spec([])
        self.output_spec = self.transformation.transform_output_spec(self.returned_spec)
        return self

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TransformedFunction):
            return False
        return all(
            [
                self.original_f == other.original_f,
                self.transformation == other.transformation,
                # transformed_function may have a different memory address, but — given
                # that transformation is assumed to be a pure function —
                # transformed_function should be semantically the same. Therefore, we
                # don't check that the transformed_functions are equal.
                # self.transformed_function == other.transformed_function,
                self.input_spec == other.input_spec,
                self.passed_spec == other.passed_spec,
                self.returned_spec == other.returned_spec,
                self.output_spec == other.output_spec,
            ]
        )

    def __hash__(self):
        return object.__hash__(self)

    def _traverse(self):
        state = self
        while isinstance(state.original_f, TransformedFunction):
            yield state
            state = state.original_f
        yield state
        yield state.original_f

    def __repr__(self) -> str:
        wrapped_fns = deque()
        for x in self._traverse():
            wrapped_fns.appendleft(
                f"    {repr(x.transformation if isinstance(x, TransformedFunction) else x)},"
            )
        steps_string = "\n".join(wrapped_fns)
        return f"TransformedFunction(\n  <compose(\n{steps_string}\n  )>\n)"
