# spekk is a tool for working with named dimensions for arrays
`spekk` lets you declare specifications of the shapes of your arrays.

A common problem with array programming (i.e. working with libraries such as NumPy or JAX) is that an array can have many dimensions, and it can be easy to get them wrong. Additionally, dimensions may be "shared" across different arrays, for example a `"batch"` dimension may exist both for a set of images and their corresponding label tokens — each element in the `"batch"` dimension consist of one image and one label. `spekk` attempts to solve this by providing a way to declare the dimensions of arrays using a class called `Spec`.

`spekk` exists independently of the underlying arrays and can thus be used to specify the dimensions of both NumPy and JAX arrays (or anything else that has a `shape` property).


# How does it work
Anything that can be represented as a (nested) tree of arrays _(see `spekk.trees.registry.TreeDef`)_ can be specced. This means that we can easily spec custom classes, dictionaries, or the arguments to a function:

```python
from spekk import Spec
import numpy.random as random

def foo(images, labels):
    ...

spec = Spec(
    {
        "images": ["batch", "width", "height"],  # <-  Dimensions are always a sequence of strings
        "labels": ["batch", "label_tokens"],  # <- "labels" also have a "batch" dimension
        # Since both "images" and "labels" have a "batch" dimension, they should be iterated over together
    }
)
# The data could look like this:
data = {
    # batch-size = 32
    # width = height = 128 pixels
    # Number of tokens per label = 10
    "images": random.normal(size=(32, 128, 128)),
    "labels": random.normal(size=(32, 10)),
}
spec.validate(data)  # <- This is OK!

# We can get the indices in the data for a given dimension:
assert spec.index_for("batch") == {"images": 0, "labels": 0}
assert spec.index_for("width") == {"images": 1, "labels": None}
assert spec.index_for("height") == {"images": 2, "labels": None}
```

We can also spec more complex nested data structures:

```python
data = {
    "particles": {
        # The last dimension (with size = 2) represent the x- and y-components of the position and velocity.
        "position": random.random(size=(3, 2)),
        "velocity": random.random(size=(4, 3, 2)),
    },
    "attractors": {
        "position": random.random(size=(5, 2)),
        "strength": random.random(size=(5,)),
    },
}
spec = Spec(
    {
        "particles": {
            # We can ignore trailing dimensions such as the x- and y-components above
            "position": ["particles"],
            "velocity": ["starting_velocities", "particles"],
        },
        "attractors": {
            "position": ["attractors"],
            "strength": ["attractors"],
        },
    }
)
spec.validate(data)
assert spec.index_for("particles") == {
    "particles": {"position": 0, "velocity": 1},
    "attractors": {"position": None, "strength": None},
}
```


## Building up more complex specced functions using function transformations
`spekk` is quite verbose, but it gets more useful when used as a building block for describing what happens to data when it is passed as arguments to a function.

Let's say we have a function that takes a point at position (`x`, `y`) and a scalar value `c` and returns the value of a circle and a hyperbola at that point:

```python
def f(x, y, c):
    "Return a dictionary of a circle and a hyperbola evaluated at point (x, y)."
    return {
        "circle":    x**2 + y**2 - c**2,
        "hyperbola": x**2 - y**2 - c**2,
    }
```

We can spec this function by wrapping it in a `Specced` object. The `Specced` function acts the same as the original function when called, but it additionally describes what happens to the `input_spec` (aka the spec of the arguments to the function) after it has been called. In this case we assume (without actually checking) that all input arguments are scalars, and we return a dictionary with two numbers, one for the circle and one for the hyperbola:

```python
from spekk.transformations import Specced

specced_f = Specced(f, lambda input_spec: {"circle": (), "hyperbola": ()})  # An empty dimensions-sequence represent a 0-dimensional array, also called a scalar

# The returned values are the same before and after speccing the function:
assert f(x=1, y=2, c=3) == specced_f(x=1, y=2, c=3)
```

But what if we want a function that takes a list of `x`-values and a list of `y`-values and evaluate `f` on all the points in the grid? It would be nice if we could write it like this:

```python
from spekk.transformations import ForAll, compose

f_for_grid = compose(
    specced_f,
    ForAll("y-values"),  # Run it for all the y-values (all rows)
    ForAll("x-values"),  # Run it for all the x-values (all columns)
)
```

This is valid `spekk` code. `compose` wraps each function in the previous step with a transformation, so starting with `specced_f`, `ForAll("y-values")` transforms it into a function that automatically runs it for all the `"y-values"` (defined by a spec as shown below). `ForAll("x-values")` then takes the result of the function transformation and transforms it again to run for all the `"x-values"`.

Let's create some example data and define the corresponding spec:

```python
import numpy as np

# A grid of points with shape (20, 30)
x = np.linspace(-5, 5, 20)  # 20 columns
y = np.linspace(-5, 5, 30)  # 30 rows
c = 2

# The corresponding spec may look like this:
spec = Spec({"x": ["x-values"], "y": ["y-values"]})
```

We let the transformed function know about the dimensions of the data by building it with the spec:

```python
f_for_grid = f_for_grid.build(spec)  # Now it knows how to loop over the data :)

# We can call it now with the lists of x- and y-values:
result = f_for_grid(x=x, y=y, c=c)
# result is a dictionary with keys "circle" and "hyperbola", each value is a 2D list 
# with shape (20, 30), corresponding to the output_spec:
assert f_for_grid.output_spec == Spec({
    "circle":    ["x-values", "y-values"],
    "hyperbola": ["x-values", "y-values"],
})
```

We can further transform the function over arbitrary dimensions, as long as we also update the spec:

```python
f_for_all_xyc = compose(
    f_for_grid,
    ForAll("c-values"),
).build(spec.replace({"c": ["c-values"]}))  # <- Update the spec to include the "c-values" dimension for the c argument.

# We can now call it with multiple values for "c" as well:
result = f_for_grid(x=x, y=y, c=np.arange(0, 6))  # <- 6 values for "c"
# result is a dictionary with keys "circle" and "hyperbola", each value is a 2D list 
# with shape (20, 30), corresponding to the output_spec:
assert f_for_all_xyc.output_spec == Spec({
    "circle":    ["c-values", "x-values", "y-values"],
    "hyperbola": ["c-values", "x-values", "y-values"],
})

from spekk.util import shape
assert shape(result["circle"]) == (6, 20, 30)  # <- 6 values for "c"
```

In many cases, broadcasting in Numpy will be enough to get the desired result. However, broadcasting can sometimes be difficult or inefficient to get right, and it can be hard to keep track of the dimensions of the arrays. `ForAll` makes it easier to write code that loops over arbitrary dimensions, and if used in conjunction with JAX and `jax.vmap` it can be very efficient as well.
