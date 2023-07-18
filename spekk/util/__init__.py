"Utility functions for ``spekk``."

from spekk.util.flatten import flatten
from spekk.util.shape import shape
from spekk.util.slicing import slice_data, slice_spec
from spekk.util.validation import ValidationError, validate

__all__ = [
    "flatten",
    "shape",
    "slice_data",
    "slice_spec",
    "ValidationError",
    "validate",
]
