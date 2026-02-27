__all__ = [
    "__array_namespace_info__",
    "capabilities",
    "default_device",
    "default_dtypes",
    "devices",
    "dtypes",
]

from spekk.ops._backend import backend
from spekk.ops._types import Capabilities, DataTypes, DefaultDataTypes, Info, device


def __array_namespace_info__() -> Info:
    """
    Returns a namespace with Array API namespace inspection utilities.

    Returns
    -------
    out: Info
        An object containing Array API namespace inspection utilities.

    Examples
    --------
    ::

      info = xp.__array_namespace_info__()
      info.capabilities()
      info.devices()
      info.dtypes()
      info.default_dtypes()
    """
    return backend.__array_namespace_info__()


def capabilities() -> Capabilities:
    """
    Returns a dictionary of array library capabilities.

    The dictionary contains the following keys:

    -   ``"boolean indexing"``: boolean indicating whether the backend supports
        boolean indexing.
    -   ``"data-dependent shapes"``: boolean indicating whether the backend
        supports data-dependent output shapes.

    Returns
    -------
    out: Capabilities
        A dictionary of array library capabilities.
    """
    # NOTE: I don't think this is needed? It is part of the object returned from __array_namespace_info__() (I think).
    return backend.capabilities()


def default_device() -> device:
    """
    Returns the default device.

    Returns
    -------
    out: device
        The default device for the current backend.
    """
    # NOTE: I don't think this is needed? It is part of the object returned from __array_namespace_info__() (I think).
    return backend.default_device()


def default_dtypes(
    *,
    device: device | None = None,
) -> DefaultDataTypes:
    """
    Returns a dictionary containing default data types.

    The dictionary has the following keys:

    -   ``"real floating"``: default real floating-point data type.
    -   ``"complex floating"``: default complex floating-point data type.
    -   ``"integral"``: default integral data type.
    -   ``"indexing"``: default array index data type.

    Parameters
    ----------
    device: device | None
        Device for which to return default data types. If ``None``, returns
        the default data types for the current device. Default: ``None``.

    Returns
    -------
    out: DefaultDataTypes
        A dictionary containing the default data type for each data type kind.
    """
    # NOTE: I don't think this is needed? It is part of the object returned from __array_namespace_info__() (I think).
    return backend.default_dtypes(device=device)


def dtypes(
    *,
    device: device | None = None,
    kind: str | tuple[str, ...] | None = None,
) -> DataTypes:
    """
    Returns a dictionary of supported Array API data types.

    Parameters
    ----------
    kind: str | tuple[str, ...] | None
        Data type kind.

        -   If ``kind`` is ``None``, returns a dictionary containing all
            supported Array API data types.

        -   If ``kind`` is a string, returns a dictionary containing the data
            types belonging to the specified kind. Supported kinds:

            -   ``'bool'``: boolean data types (e.g., ``bool``).
            -   ``'signed integer'``: signed integer data types (e.g.,
                ``int8``, ``int16``, ``int32``, ``int64``).
            -   ``'unsigned integer'``: unsigned integer data types (e.g.,
                ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            -   ``'integral'``: integer data types. Shorthand for
                ``('signed integer', 'unsigned integer')``.
            -   ``'real floating'``: real-valued floating-point data types
                (e.g., ``float32``, ``float64``).
            -   ``'complex floating'``: complex floating-point data types
                (e.g., ``complex64``, ``complex128``).
            -   ``'numeric'``: numeric data types. Shorthand for
                ``('integral', 'real floating', 'complex floating')``.

        -   If ``kind`` is a tuple, returns a dictionary containing the data
            types belonging to at least one of the specified kinds.

        Default: ``None``.
    device: device | None
        Device for which to return supported data types. If ``None``, returns
        the supported data types for the current device. Default: ``None``.

    Returns
    -------
    out: DataTypes
        A dictionary containing supported data types.
    """
    # NOTE: I don't think this is needed? It is part of the object returned from __array_namespace_info__() (I think).
    return backend.dtypes(device=device, kind=kind)


def devices() -> list[device]:
    """
    Returns a list of supported devices available at runtime.

    Each returned device object can be passed as a ``device`` keyword argument
    to array creation functions.

    Returns
    -------
    out: list[device]
        A list of supported devices.
    """
    # NOTE: I don't think this is needed? It is part of the object returned from __array_namespace_info__() (I think).
    return backend.devices()
