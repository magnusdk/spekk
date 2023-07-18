"""Transformations are higher-order functions used to wrap other functions in order to 
modify their behavior. :mod:`spekk.transformations` additionally allows you to specify
how the spec changes when a function is transformed.

Example:

    Let's say we have a function ``f`` that takes a single scalar argument ``x`` and 
    returns ``x**2``:

    >>> f = lambda x: x**2

    Let's say we also have some data and spec where ``x`` is a list of numbers:

    >>> from spekk import Spec
    >>> data = {"x": [1, 2, 3]}
    >>> spec = Spec({"x": ["numbers"]})

    We can use :class:`spekk.transformations.ForAll` to transform ``f`` into a function
    that takes a list of numbers and returns a list of numbers:

    >>> from spekk.transformations import ForAll
    >>> tf = ForAll("numbers")(f)
    >>> tf = tf.build(spec)  # Let the transformation know what the spec of the data is
    >>> tf(**data)
    [1, 4, 9]

    Now we have transformed ``f`` to take a list of integers instead of just a number!
    If we are using JAX, we could use :class:`spekk.transformations.ForAll` along with
    :func:`jax.vmap` to get the same result, except vectorized and running in parallel 
    on a GPU.

    This may still seem very verbose, but imagine that ``f`` is a more complex function 
    that takes in many more arguments, where some of the arguments have some of the 
    same dimensions. We can build up functionality by composing transformations:

    >>> from spekk.transformations import compose, Apply, Axis
    >>> import jax.numpy as jnp
    >>> import jax
    >>> some_complex_function = lambda *args, **kwargs: ...  # Just imagine it...
    >>> some_complex_spec = Spec()  # Just imagine it...
    >>> tf = compose(
    ...     some_complex_function,
    ...     ForAll("dimension0", vmap_impl=jax.vmap),
    ...     ForAll("dimension1", vmap_impl=jax.vmap),
    ...     Apply(jnp.sum, Axis("dimension1")),
    ...     ForAll("dimension2", vmap_impl=jax.vmap),
    ... ).build(some_complex_spec)
    
On a high level, a transformation takes in a function as an argument, the spec of the 
input-arguments, and the spec of the returned value from the function, and returns a 
new transformed function (of type :class:`spekk.transformations.TransformedFunction`).
It also gives information about how the spec of the original function changes when 
transformed.

A schematic of how :mod:`spekk.transformations` transforms a function is shown below. 
Note how it can be arbitrarily stacked to build up functionality.

.. code-block:: text

    input_spec         Transformed function         output_spec
        │                        ▲                        ▲
    ┌───┼────────────────────────┼────────────────────────┼───┐
    │   ├──────────────┐         │            ┌───────────┴─┐ │  
    │   │              │  ┌──────┴──────┐     │ Transform   │ │
    │ ┌─▼───────────┐  │  │ Transform   │     │ output spec │ │
    │ │ Transform   │  │  │ function    │     └──────▲──────┘ │
    │ │ input spec  │  │  └─▲────▲────▲─┘            │        │
    │ └──────┬──────┘  └────┘    │    └──────────────┤        │
    └────────┼───────────────────┼───────────────────┼────────┘
             │                   │                   │
        passed_spec              │              returned_spec
             │                   │                   │
           ┌╶▼╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶┴╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶┴╶┐
           ╎            Wrapped function               ╎
           └╶╶╶╶╶╶┬╶╶╶╶╶╶╶╶╶╶╶╶╶╶▲╶╶╶╶╶╶╶╶╶╶╶╶╶╶▲╶╶╶╶╶╶┘
                  ╎              ╎              ╎

For example, :class:`spekk.transformations.ForAll` transforms a function such that it 
runs for each element of the input data (along a dimension named by spec) individually. 
It can be used to transform a function such that it works on lists of numbers instead 
of just a single number, similarly to :func:`jax.vmap`.

Let's call the function that is transformed the "wrapped function.". Because the 
wrapped function runs on each element individually, we remove the looped-over/vecorized 
dimension from the spec before it is passed on down. Thus, 
:class:`spekk.transformations.ForAll` removes the dimension when transforming the input 
spec (visualied as the left-most inner rectangle in the above schematic).

The transformed input spec is named ``passed_spec`` in the schematic, and is passed 
down to the wrapped function. The wrapped function may be a 
:class:`TransformedFunction` itself, in which case it also has information about the 
spec of the returned value. The spec of the returned value is called ``returned_spec`` 
in the schematic.

In the case of :class:`spekk.transformations.ForAll`, the looped-over/vecorized 
dimension is re-added to the spec before being returned. The final returned spec is 
called ``output_spec`` in the schematic.

The actual transformation step takes the wrapped function, the original ``input_spec``, 
and the ``returned_spec`` to produce a new function.
"""

from spekk.transformations import common
from spekk.transformations.apply import Apply
from spekk.transformations.axis import Axis
from spekk.transformations.base import (
    PartialTransformation,
    Specced,
    Transformation,
    TransformedFunction,
)
from spekk.transformations.common import compose
from spekk.transformations.for_all import ForAll
from spekk.transformations.reduce import Reduce
from spekk.transformations.wrap import Wrap

__all__ = [
    "common",
    "Apply",
    "Axis",
    "PartialTransformation",
    "Specced",
    "Transformation",
    "TransformedFunction",
    "compose",
    "ForAll",
    "Reduce",
    "Wrap",
]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
