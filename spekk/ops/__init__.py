"""Function stubs and API documentation for the array API standard."""

from spekk.array import data_types as dtype
from spekk.array import fft, linalg
from spekk.ops._backend import backend
from spekk.ops.array_object import *
from spekk.ops.constants import *
from spekk.ops.creation_functions import *
from spekk.ops.data_type_functions import *
from spekk.ops.elementwise_functions import *
from spekk.ops.indexing_functions import *
from spekk.ops.info import __array_namespace_info__
from spekk.ops.linear_algebra_functions import *
from spekk.ops.manipulation_functions import *
from spekk.ops.searching_functions import *
from spekk.ops.set_functions import *
from spekk.ops.sorting_functions import *
from spekk.ops.statistical_functions import *
from spekk.ops.utility_functions import *
from spekk.ops.vbeam_extensions import *

__array_api_version__: str = "2023.12"
"""
String representing the version of the array API specification which the conforming implementation adheres to.
"""
