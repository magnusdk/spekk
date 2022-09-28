"""A module for ad-hoc speccing array data and functions working on array data.

Speccing provides similar features as libraries such as xarray, allowing us to give names to the axes of an array. Unlike xarray, Spekk specs exist outside of, and independently of, the arrays that they describe. This puts more responsibility onto the users of Spekk, but also makes it completely agnostic to the underlying array backend.

There are two main components used for speccing data:
  Shape: a list of names for the dimensions of a single array.
  Spec: a list of shapes specifying the shape of each argument of a function, optionally
    with names for each argument.

A Shape is basically just a list of strings:
>>> Shape(["foo", "bar"])
Shape(dims=['foo', 'bar'])

A Spec consists of a list of shapes and optionally a list of corresponding names:
>>> my_spec = Spec([["foo"], ["bar", "foo"]], ["arg1", "arg2"])
>>> my_spec
Spec(
  arg1=['foo'],
  arg2=['bar', 'foo']
)

The above spec can be used to spec a function that takes two arguments, where the first 
dimension of the first argument and the second dimension of the second argument are part
of the same dimension. Use the method `indices_for` to get the indices of each argument 
for the given dimension.
>>> my_spec.indices_for("foo")
[0, 1]

If an argument does not contain that dimension, its index will be `None`.
>>> my_spec.indices_for("bar")
[None, 0]

You can validate some data given a spec:
>>> import numpy as np
>>> my_data = [np.ones((2,)), np.ones((2, 3))]
>>> my_spec.validate(my_data)
Traceback (most recent call last):
...
spekk.common.ValidationError: The size of a dimension must be the same for all arguments. The data has different sizes for dimension foo: arg1 has size 2, arg2 has size 3.
>>> my_valid_data = [np.ones((2,)), np.ones((3, 2))]
>>> my_spec.validate(my_valid_data)  # This does not raise a ValidationError
"""


from spekk.common import Specable, ValidationError
from spekk.ops import apply_across_dim
from spekk.shape import Shape
from spekk.slicer import Slicer
from spekk.spec import Spec
