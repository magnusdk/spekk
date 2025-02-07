import uuid
from typing import Optional

from spekk.ops._types import Dim


def random_dim_name(obj, text: Optional[str] = None) -> Dim:
    """Generate a randomized dimension name that is presumably not meant to be read by
    humans. I.e.: it is presumably some temporary dimension created by a function that
    will be reduced over before returning.
    """
    random_id = str(uuid.uuid4())
    if text:
        random_id = f"{text}_{random_id}"
    return f"{obj.__module__}.{obj.__class__.__name__}_{random_id}"
