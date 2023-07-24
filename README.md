# spekk is a tool for working with named dimensions for arrays
![spekk logo](docs/_static/spekk_logo.png)

`spekk` lets you declare specifications of the shapes of your arrays.

A common problem with multidimensional arrays is that it can be hard to keep track of the dimensions of the data over time. Additionally, dimensions may be "shared" across different arrays. For example, two different arguments to a function may share a dimension, like the positions of receiving elements in an ultrasound array and their corresponding recorded signals.

`spekk` attempts to solve this by providing a way to declare the dimensions of arrays using a class called `Spec`:

```python
import numpy as np
from spekk import Spec

# Anything that is a collection of key-value pairs can be used as a container of arrays:
spec = Spec({
    "receiving_element": { # <- You could also use your own custom class instead of dict
        "position": ["receivers", "xyz"],
        "weight": ["receivers"],
    },
    "signal": ["transmits", "receivers", "samples"],
})
# my_fn = lambda signal, receiving_element: ...
```

`spekk` exists independently of the underlying arrays and can thus be used to specify the dimensions of both NumPy and JAX arrays (or anything else that has a `shape` property). This is useful when working with code that need to support multiple array backends.

See [an overview of spekk](#overview) below. [Read the documentation](https://spekk.readthedocs.io/) for more details.


# Installation
```bash
python3 -m pip install spekk
```


# Overview
`spekk` lets you name the dimensions of an array as a sequence of strings:

```python
import numpy as np
from spekk import Spec

data = np.ones([4, 5, 6])  # A 3D array with shape [4, 5, 6].
spec = Spec(["transmits", "receivers", "samples"])  # <- The names of the dimensions, 
                                                    #    each name corresponding to an 
                                                    #    axis in the array ([4, 5, 6]).
```

It also lets you specify the dimensions of nested data structures of arrays:

```python
data = {
    "receiving_element": {
        "position": np.ones([5, 3]),
        "weight": np.ones([5]),
    },
    "signal": np.ones([4, 5, 6]),
}
# Note that the structure is the same as the data:
spec = Spec({
    "receiving_element": {
        "position": ["receivers", "xyz"],
        "weight": ["receivers"],
    },
    "signal": ["transmits", "receivers", "samples"],
})
```

You can spec what happens to data when you apply some function to it:

```python
from spekk.transformations import Specced

def f(x, y, c):
    """Return a dictionary of a circle and two hyperbolas (one for each axis) evaluated 
    at point (x, y) with radius/axis-width c."""
    return {
        "circle": x**2 + y**2 - c**2,
        "hyperbola": [
            x**2 - y**2 - c**2,
            x**2 - y**2 + c**2
        ],
    }

# Ignore input_spec and just return the output_spec:
specced_f = Specced(f, lambda input_spec: {"circle": [], "hyperbola": ["axes"]})
specced_f = specced_f.build(spec) # <- Let the function know about the spec

assert f(x=1, y=2, c=3) == specced_f(x=1, y=2, c=3)
assert specced_f.output_spec == Spec({"circle": [], "hyperbola": ["axes"]})
```

You can describe what happens to the spec of a function when you transform the function, for example when transforming it to loop over the arguments:

```python
from spekk.transformations import ForAll, compose
from spekk.util import shape

# The following spec represent the input kwargs to the function f:
spec = Spec({"x": ["x-values"], "y": ["y-values"], "c": ["c-values"]})

tf = compose(
    specced_f,
    ForAll("y-values"),  # Run it for all the y-values (all rows)
    ForAll("x-values"),  # Run that for all the x-values (all columns)
    ForAll("c-values"),  # And then run that for all values of c
).build(spec)  # <- Building the transformed function lets it know the spec of the data
               #    so that it also knows how to loop over it.
result = tf(x=np.linspace(-5, 5, 10), y=np.linspace(-5, 5, 11), c=np.arange(1, 6))

assert shape(result["circle"]) == (5, 10, 11)
assert tf.output_spec == Spec({
    "circle":    ["c-values", "x-values", "y-values"],
    "hyperbola": ["c-values", "x-values", "y-values", "axes"],
})
```

You may use more powerful frameworks when transforming functions using `ForAll`:

```python
from functools import partial
import jax

# Use JAX's vmap to vectorize the function in order to run in parallel on GPUs:
ForAll_jax = partial(ForAll, vmap_impl=jax.vmap)
```

In most cases, Numpy broadcasting will be enough to get the desired result when working with multidimensional data. However, broadcasting can sometimes be difficult or inefficient to get right, and it can be hard to keep track of the dimensions of the arrays over time. `ForAll` makes it easier to write code that loops over arbitrary dimensions and — if used in conjunction with for example JAX and `jax.vmap` — it can be very efficient as well.

[Read the documentation](https://spekk.readthedocs.io/) for more details.

