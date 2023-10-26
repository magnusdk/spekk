""":class:`Reduce` transforms a function that works on scalar inputs such that it works 
on arrays instead and iteratively reduces the values of a dimension. For example, 
``Reduce.Sum("dimension")`` produces equivalent results as a ``ForAll("dimension")`` 
followed by an ``Apply(np.sum, "dimension")`` transformation, but will potentially use 
less memory because it sums the partial results iteratively isntead of trying to 
parallelize over the dimension first (in the case of using GPU-backends such as JAX).
"""
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, TypeVar

from spekk import Spec, trees, util
from spekk.transformations import Transformation, common
from spekk.transformations.axis import Axis, concretize_axes

T_reduce_cls = TypeVar("T_reduce_cls", bound="Reduce")
T_f_result = TypeVar("T_f_result")
T_reduction_result = TypeVar("T_reduction_result")
T_reduce_fn = Callable[[T_reduction_result, T_f_result], T_reduction_result]
T_reduce = Callable[[T_reduce_fn, Iterable, T_reduction_result], T_reduction_result]


@dataclass
class Reduce(Transformation):
    """Transform a function to make it reduce the values of a dimension iteratively.

    A :class:`Reduce` transformation is generally a :class:`ForAll` and :class:`Apply`
    transformation combined, if the :class:`Apply` transformation somehow aggregates
    the result (for example by summing over the vectorized axis).

    As a concrete example:
    ``ForAll("transmits")`` followed by ``Apply(np.sum, Axis("transmits")`` is
    equivalent to ``Reduce.Sum("transmits")``, but using ``Reduce.Sum`` will likely
    allocate a lot less memory, potentially at the cost of processing time.
    """

    dimension: str  #: The dimension to reduce over.
    reduce_fn: T_reduce_fn  #: The function to reduce with. For example operator.add for summation.
    initial_value: Optional[
        T_reduction_result
    ] = None  #: The initial value for the reduction. For example 0 for summation.
    enumerate: bool = False  #: If True, each item is a tuple of (index, value), similar to Python's built-in :func:`enumerate`.
    extra_args: Sequence[Any] = field(
        default_factory=tuple
    )  #: Extra arguments to pass to the reduce function.
    extra_kwargs: Dict[str, Any] = field(
        default_factory=dict
    )  #: Extra keyword arguments to pass to the reduce function.
    reduce_impl: Optional[
        T_reduce
    ] = None  #: The ``reduce`` implementation to use. Defaults to Python's built-in :func:`functools.reduce`.

    def __post_init__(self):
        "Sub-classes may override this method to perform additional initialization."

    def transform_function(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        if not input_spec.has_dimension(self.dimension):
            raise ValueError(f"Spec does not contain the dimension {self.dimension}.")

        def wrapped(*_unsupported_positional_args, **kwargs):
            if _unsupported_positional_args:
                raise ValueError(
                    "Positional arguments are not supported when using Reduce. Use keyword arguments when calling the transformed function instead."
                )
            extra_args, extra_kwargs = concretize_axes(
                output_spec, self.extra_args, self.extra_kwargs
            )
            reduce_fn = lambda *args, **kwargs: self.reduce_fn(
                *args, *extra_args, **kwargs, **extra_kwargs
            )
            return specced_map_reduce(
                to_be_transformed,
                reduce_fn,
                kwargs,
                input_spec,
                self.dimension,
                self.initial_value,
                self.enumerate,
                self.reduce_impl,
            )

        return wrapped

    def transform_input_spec(self, spec: Spec) -> Spec:
        return spec.remove_dimension(self.dimension)

    def transform_output_spec(self, spec: Spec) -> Spec:
        tree = (self.extra_args, self.extra_kwargs)
        for leaf in trees.leaves(
            tree, lambda x: isinstance(x, Axis) or not trees.has_treedef(x)
        ):
            if isinstance(leaf.value, Axis):
                spec = spec.update_leaves(leaf.value.new_dimensions)
        return spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + f'"{self.dimension}", '
            + f"{self.reduce_fn}, "
            + f"{self.initial_value})"
        )

    @classmethod
    def Sum(
        cls: T_reduce_cls,
        dimension: str,
        initial_value: Optional[T_reduction_result] = None,
        reduce_impl: T_reduce = None,
    ) -> T_reduce_cls:
        """Transformation that iteratively adds the results of the wrapped function for
        each item in the given dimension."""
        return cls(dimension, operator.add, initial_value, reduce_impl=reduce_impl)

    @classmethod
    def Product(
        cls: T_reduce_cls,
        dimension: str,
        initial_value: Optional[T_reduction_result] = None,
        reduce_impl: T_reduce = None,
    ) -> T_reduce_cls:
        """Transformation that iteratively multiplies the results of the wrapped
        function for each item in the given dimension."""
        return cls(dimension, operator.mul, initial_value, reduce_impl=reduce_impl)


def specced_map_reduce(
    map_f: Callable[[Any], Any],
    reduce_f: T_reduce_fn,
    data: Any,
    spec: Spec,
    dimension: str,
    initial_value: Optional[Any] = None,
    enumerate: bool = False,
    reduce_impl: Optional[T_reduce] = None,
):
    # Flatten the data so that we can iterate over it without worrying about the
    # potentially nested structure of the data.
    flattened_args, in_axes, unflatten = util.flatten(data, spec, dimension)

    # Get the size of the dimension that we are reducing over.
    size = spec.size(data, dimension)
    if size == 0:  # Return early if there are no elements to reduce over.
        return initial_value

    # Get the first mapped value.
    carry = common.map_1_flattened(map_f, flattened_args, in_axes, unflatten, 0)
    # Use the first mapped value as the initial value if no initial value was given.
    if initial_value is not None:
        x = (0, carry) if enumerate else carry
        carry = reduce_f(initial_value, x)
    # Don't reduce further if there was only one value
    if size == 1:
        return carry

    # `wrapped` puts everything together and makes it work with `reduce_impl`.
    # It gets the arguments indexed at `i` for the given dimension, and applies the
    # mapping function to them before performing a reduction step.
    def wrapped(carry, i):
        x = common.map_1_flattened(map_f, flattened_args, in_axes, unflatten, i)
        x = (i, x) if enumerate else x
        return reduce_f(carry, x)

    # Default `reduce_impl` is Python's built-in `functools.reduce`.
    if reduce_impl is None:
        import functools

        reduce_impl = functools.reduce
    # Reduce over the remaining elements. Note that `carry` is the first element, so we
    # start at index 1 in the `range(1, size)`.
    return reduce_impl(wrapped, range(1, size), carry)
