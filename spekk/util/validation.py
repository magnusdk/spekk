"Validate data according to a :class:`~spekk.spec.Spec`."

from dataclasses import dataclass
from typing import Generator, Sequence

from spekk import trees, util
from spekk.spec import Spec


class ValidationError(Exception):
    "Raised when the data does not conform to the spec."


@dataclass
class _DimensionSizeInfo:
    "Helper class for keeping track of the shape of data for a dimension."
    shape: Sequence[int]  # The shape of the value
    index: int  # The index in the shape for the dimension
    path: tuple  # The path to the value

    @property
    def size(self):
        "The size of the dimension in the data."
        return self.shape[self.index]


def _paths_and_indices(spec: Spec, dimension: str) -> Generator[tuple, None, None]:
    """Yield paths and indices for the given dimension.

    For example, given a spec as such:

    >>> spec = Spec({
    ...     "k1": {"k1_sub": ["dim1", "dim2"]},
    ...     "k2": ["dim2"]
    ... })

    This is what we get if we print out each path and index for the dimension "dim1":

    >>> for path, index in list(_paths_and_indices(spec, "dim1")):
    ...     print(path, index)
    ('k1', 'k1_sub') 0

    For dimension "dim2", both the keys "k1" and "k2" contain the dimension:

    >>> for path, index in list(_paths_and_indices(spec, "dim2")):
    ...     print(path, index)
    ('k1', 'k1_sub') 1
    ('k2',) 0
    """
    # An index can either be int or None
    is_leaf = lambda x: isinstance(x, int) or x is None
    indices = spec.index_for(dimension)
    for subtree in trees.leaves(indices, is_leaf):
        index = subtree.value
        if index is not None:
            yield subtree.path, index


def _check_path_present_in_data(data, path):
    """Return the value at the given path in the data, raising a
    :class:`ValidationError` if it is not present."""
    try:
        return trees.get(data, path)
    except KeyError:
        raise ValidationError(
            f"Path {list(path)} is not present in the data yet it is present in the spec."
        ) from None


def _check_value_has_shape_attribute(value, path):
    """Return the shape of the value, raising a :class:`ValidationError` if it does not
    have a ``__spekk_shape__`` or ``shape`` attribute."""
    try:
        return util.shape(value)
    except Exception as e:
        raise ValidationError(
            f"Unable to get the shape of value at path {list(path)} with type {type(value)}."
        ) from e


def _check_value_has_dimensions(spec, shape, path):
    """Raise a :class:`ValidationError` if the value has fewer dimensions than the spec.

    If the spec has more dimensions than the data then the data is not valid for the
    spec. The data may however have more dimensions than the spec, for example if the
    data has any additional dimensions that we don't care about, like the
    ``xyz``-dimension of a point coordinate. The additional dimensions are assumed to
    come last in the shape."""
    spec_dimensions_at_path = trees.get(spec, path)
    if len(shape) < len(spec_dimensions_at_path):
        raise ValidationError(
            f"The data has only {len(shape)} dimension{'s' if len(shape)>1 else ''} \
while the spec specifies dimensions {spec_dimensions_at_path} \
({len(spec_dimensions_at_path)} in total) at the path {list(path)}."
        )


def _check_consistent_dimension_sizes(
    dimension: str,
    dimension_sizes_info: Sequence[_DimensionSizeInfo],
):
    """Raise a :class:`ValidationError` if the dimension has inconsistent sizes in the
    data.

    This can happen if the data has arrays with different sizes for a given dimension.
    All array-sizes corresponding to the shape dimension must have the same size."""
    dimension_sizes = [info.size for info in dimension_sizes_info]
    if len(set(dimension_sizes)) > 1:  # If there are more than 1 distinct value
        path_sizes_str = "\n".join(
            f"    - Size={info.size} at path {list(info.path)}, shape={info.shape}, index={info.index}"
            for info in dimension_sizes_info
        )
        raise ValidationError(
            f"Dimension '{dimension}' has inconsistent sizes in the data:\n{path_sizes_str}"
        )


def validate(spec: Spec, data: trees.Tree):
    """Validate that the data conforms to the spec, raising a :class:`ValidationError`
    if not.

    Examples:

    >>> import numpy as np
    >>> spec = Spec({
    ...     "foo": {"bar": ["dim1", "dim2"]},
    ...     "baz": ["dim2"],
    ... })

    The following data is valid because ``"dim2"`` under path ``["foo", "bar"]`` has
    the same size as under path ``["baz"]`` and it has the same structure as the spec:

    >>> validate(spec, {
    ...     "foo": {"bar": np.ones((2, 3))},
    ...     "baz": np.ones((3,)),
    ... })

    If we try to validate against a spec with a path that is not present in the data, a
    :class:`ValidationError` is raised:

    >>> validate(spec, {
    ...     "foo": {"invalid_key": np.ones((2, 3))},
    ...     "baz": np.ones((3,)),
    ... })
    Traceback (most recent call last):
        ...
    ValidationError: Path ['foo', 'bar'] is not present in the data yet it is present in the spec.

    All specced values must have a ``shape`` attribute (or more generally; be supported
    by :func:`spekk.util.shape.shape`):

    >>> validate(spec, {
    ...     "foo": {"bar": np.ones((2, 3))},
    ...     "baz": object(),  # <- An object does not have a shape attribute
    ... })
    Traceback (most recent call last):
        ...
    ValidationError: Unable to get the shape of value at path ['baz'] with type <class 'object'>.

    The data must have at least as many dimensions as the spec:

    >>> validate(
    ...     Spec({"foo": ["dim1", "dim2"]}),
    ...     {"foo": np.ones((2,))},  # <- Too few dimensions!
    ... )
    Traceback (most recent call last):
        ...
    ValidationError: The data has only 1 dimension while the spec specifies dimensions ['dim1', 'dim2'] (2 in total) at the path ['foo'].

    It is OK if the data has more dimensions than the spec:

    >>> validate(
    ...     Spec({"foo": ["dim1", "dim2"]}),    # <- spec specifies 2 dimensions
    ...     {"foo": np.ones((2, 3, 4, 5, 6))},  # <- 5 dimensions is more than 2, which is OK!
    ... )

    The data must have the same size for a given dimension in all places where it is
    used:

    >>> validate(spec, {
    ...     "foo": {"bar": np.ones((2, 3))},
    ...     "baz": np.ones((4,)),  # <- Size of dim2 is 4 here, but 3 above!
    ... })
    Traceback (most recent call last):
        ...
    ValidationError: Dimension 'dim2' has inconsistent sizes in the data:
        - Size=3 at path ['foo', 'bar'], shape=(2, 3), index=1
        - Size=4 at path ['baz'], shape=(4,), index=0

    >>> validate(spec, {
    ...     "foo": {"bar": np.ones((2, 3))},
    ...     "baz": np.ones((3,)),  # <- This has the same size as the one above
    ... })

    The spec does not have to specify the dimensions for all paths in the data:

    >>> validate(spec, {
    ...     "foo": {"bar": np.ones((2, 3))},
    ...     "baz": np.ones((3,)),
    ...     "qux": np.ones((5, 6)),  # <- This is OK, the spec does not specify dimensions for "qux"
    ... })
    """
    for dimension in spec.dimensions:
        # A list is used to store information about the shape of the data for this
        # dimension, so we can check for consistency later.
        dimension_sizes_info = []
        for path, index in _paths_and_indices(spec, dimension):
            value = _check_path_present_in_data(data, path)
            shape = _check_value_has_shape_attribute(value, path)
            _check_value_has_dimensions(spec, shape, path)
            # Add the information to a list so we can check for consistency.
            dimension_sizes_info.append(_DimensionSizeInfo(shape, index, path))
        _check_consistent_dimension_sizes(dimension, dimension_sizes_info)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
