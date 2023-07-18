# spekk is a tool for working with named dimensions for arrays
`spekk` lets you declare specifications of the shapes of your arrays. A common problem with array programming (i.e. working with libraries such as NumPy or JAX) is that an array can have many dimensions, and it can be easy to get them wrong. Additionally, some semantically equivalent dimensions may exist across different arrays, for example a `"batch"` dimension for a set of images and corresponding label tokens.

`spekk` exists independently of the underlying arrays and can thus be used to specify the dimensions of both NumPy and JAX arrays (or anything else that has a `shape` property).

Everything that can be represented as a tree of arrays _(see `spekk.trees.registry.TreeDef`)_ can be specced. This means that we can easily spec a dictionary, or arguments to a function:

```python
from spekk import Spec

def foo(images, labels):
    ...

spec = Spec(
    {
        "images": ["batch", "width", "height"],  # <-  Dimensions are always a sequence of strings
        "labels": ["batch", "label_tokens"],  # <- "labels" also have a "batch" dimension
        # Since both "images" and "labels" have a "batch" dimension, that dimension is assumed to be the same for both arrays
    }
)
# The data could look like this:
import numpy.random as random
data = {
    # batch-size = 32
    # width = height = 128 pixels
    # Number of tokens per label = 10
    "images": random.normal(size=(32, 128, 128)),
    "labels": random.normal(size=(32, 10)),
}
spec.validate(data)  # <- This is OK!

# We can get the indices in the data for a given dimension:
assert spec.index_for(spec, "batch") == {"images": 0, "labels": 0}
assert spec.index_for(spec, "width") == {"images": 1, "labels": None}
assert spec.index_for(spec, "height") == {"images": 2, "labels": None}
```


## This seems verbose and stupid — how can I use it?
