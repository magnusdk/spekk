import numpy as np

from spekk import Spec
from spekk.transformations import ForAll, compose

kernel = lambda x: x**2
data = {"x": np.ones((2, 3)) * 2}
spec = Spec({"x": ["b", "a"]})

forall_xy = compose(ForAll("a"), ForAll("b"))
tf_partial = compose(kernel, forall_xy).build(spec)
tf_full = compose(kernel, ForAll("a"), ForAll("b")).build(spec)

np.testing.assert_equal(tf_partial(**data), tf_full(**data))
print(np.array_equal(tf_partial(**data), tf_full(**data)))