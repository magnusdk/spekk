__all__ = ["unique_all", "unique_counts", "unique_inverse", "unique_values"]


from spekk.ops._backend import backend
from spekk.ops.array_object import array


def unique_all(x: array, /) -> tuple[array, array, array, array]:
    """
    Returns the unique elements of ``x``, the first-occurrence indices for each unique element,
    the indices that reconstruct ``x`` from the unique elements, and the count of each unique element.

    The output shapes depend on the data values in the input array.

    Uniqueness is determined by value equality. For floating-point arrays:

    - ``nan`` values compare as ``False``, so each ``nan`` is treated as distinct and has a count of one.
    - ``-0`` and ``+0`` compare as ``True``, so signed zeros are not distinct; the retained unique
      element is backend-dependent and their counts are aggregated into a single count.
    - Because signed zeros are not distinct, using ``inverse_indices`` to reconstruct ``x`` is not
      guaranteed to return an array with the exact same values.

    Parameters
    ----------
    x: array
        Input array. If ``x`` has more than one dimension, it is flattened before finding unique elements.

    Returns
    -------
    out: tuple[array, array, array, array]
        A namedtuple ``(values, indices, inverse_indices, counts)`` where:

        - ``values``: a one-dimensional array of unique elements of ``x``, with the same data type as ``x``.
        - ``indices``: the indices of the first occurrences of each unique element in the flattened ``x``,
          with the same shape as ``values`` and the default index data type.
        - ``inverse_indices``: the indices into ``values`` that reconstruct ``x``,
          with the same shape as ``x`` and the default index data type.
        - ``counts``: the number of times each unique element occurs in ``x``,
          with the same shape as ``values`` and the default index data type.

        The order of unique elements is not specified and may vary between backends.
    """
    raise NotImplementedError("Please help me implement this!")


def unique_counts(x: array, /) -> tuple[array, array]:
    """
    Returns the unique elements of ``x`` and the count of each unique element.

    The output shapes depend on the data values in the input array.

    Uniqueness is determined by value equality. For floating-point arrays:

    - ``nan`` values compare as ``False``, so each ``nan`` is treated as distinct and has a count of one.
    - ``-0`` and ``+0`` compare as ``True``, so signed zeros are not distinct; the retained unique
      element is backend-dependent and their counts are aggregated into a single count.

    Parameters
    ----------
    x: array
        Input array. If ``x`` has more than one dimension, it is flattened before finding unique elements.

    Returns
    -------
    out: tuple[array, array]
        A namedtuple ``(values, counts)`` where:

        - ``values``: a one-dimensional array of unique elements of ``x``, with the same data type as ``x``.
        - ``counts``: the number of times each unique element occurs in ``x``,
          with the same shape as ``values`` and the default index data type.

        The order of unique elements is not specified and may vary between backends.
    """

    raise NotImplementedError("Please help me implement this!")


def unique_inverse(x: array, /) -> tuple[array, array]:
    """
    Returns the unique elements of ``x`` and the indices that reconstruct ``x`` from those unique elements.

    The output shapes depend on the data values in the input array.

    Uniqueness is determined by value equality. For floating-point arrays:

    - ``nan`` values compare as ``False``, so each ``nan`` is treated as distinct.
    - ``-0`` and ``+0`` compare as ``True``, so signed zeros are not distinct; the retained unique
      element is backend-dependent.
    - Because signed zeros are not distinct, using ``inverse_indices`` to reconstruct ``x`` is not
      guaranteed to return an array with the exact same values.

    Parameters
    ----------
    x: array
        Input array. If ``x`` has more than one dimension, it is flattened before finding unique elements.

    Returns
    -------
    out: tuple[array, array]
        A namedtuple ``(values, inverse_indices)`` where:

        - ``values``: a one-dimensional array of unique elements of ``x``, with the same data type as ``x``.
        - ``inverse_indices``: the indices into ``values`` that reconstruct ``x``,
          with the same shape as ``x`` and the default index data type.

        The order of unique elements is not specified and may vary between backends.
    """
    raise NotImplementedError("Please help me implement this!")


def unique_values(x: array, /) -> array:
    """
    Returns the unique elements of ``x``.

    The output shape depends on the data values in the input array.

    Uniqueness is determined by value equality. For floating-point arrays:

    - ``nan`` values compare as ``False``, so each ``nan`` is treated as distinct.
    - ``-0`` and ``+0`` compare as ``True``, so signed zeros are not considered distinct, and the
      retained unique element is backend-dependent.

    Parameters
    ----------
    x: array
        Input array. If ``x`` has more than one dimension, it is flattened before finding unique elements.

    Returns
    -------
    out: array
        A one-dimensional array of unique elements of ``x``, with the same data type as ``x``.
        The order of unique elements is not specified and may vary between backends.
    """
    return array(backend.unique_values(x.data))
