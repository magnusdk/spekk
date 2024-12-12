"""Function stubs and API documentation for the array API standard."""

from spekk.array import data_types as dtype
from spekk.array import fft, linalg
from spekk.array._backend import backend
from spekk.array.array_object import *
from spekk.array.constants import *
from spekk.array.creation_functions import *
from spekk.array.data_type_functions import *
from spekk.array.elementwise_functions import *
from spekk.array.indexing_functions import *
from spekk.array.info import __array_namespace_info__
from spekk.array.linear_algebra_functions import *
from spekk.array.manipulation_functions import *
from spekk.array.searching_functions import *
from spekk.array.set_functions import *
from spekk.array.sorting_functions import *
from spekk.array.statistical_functions import *
from spekk.array.utility_functions import *
from spekk.array.vbeam_extensions import *

__array_api_version__: str = "2023.12"
"""
String representing the version of the array API specification which the conforming implementation adheres to.
"""
