# spekk is a tool for working with named dimensions for arrays
`spekk` lets you declare specifications of the shapes of your arrays. A common problem with array programming (i.e. working with libraries such as NumPy or JAX) is that an array can have many dimensions, and it can be easy to get them wrong.

`spekk` exists independently of the underlying arrays and can thus be used to specify the dimensions of both NumPy and JAX arrays (or anything else that has a `shape` property).

`spekk` is designed to be simple at the cost of verbosity/ergonomy. It has two main concepts:
1. `Shape`: the named dimensions of an array.
2. `Spec`: the named dimensions of multiple objects (useful when a dimension is scattered across different arrays).


## Motivating problem
Let's say we have a dataset of ultrasound channel-data. The channel-data is for multiple frames of an ultrasound video, and each frame is created from multiple transmits, and for each transmit we have the reflected signal received at each element. The shape could thus be `(frames, transmits, receivers, signal_time)`. The shape of the actual array could be `(5, 75, 128, 1000)`, where there are 5 frames, 75 transmits, 128 elements, and 1000 samples of the received signal.

To sum over transmits using NumPy we'd run `np.sum(data, axis=1)`, with the knowledge that transmits are at `axis=1`. Then to sum over receivers we'd run `np.sum(data, axis=1)`, this time also with `axis=1` because after summing over transmits we are left with the dimensions `(frames, receivers, signal_time)`.

The problem with this is that we have to run a model in our head about which axis contains a given dimension. It would be better if we could reference the axes by names instead, for example: `np.sum(data, Axis("transmits"))` and `np.sum(data, Axis("receivers"))`.

Libraries such as [xarray](https://github.com/pydata/xarray) extends NumPy such that we can reference axes by name. However, xarray is deeply coupled with NumPy and doesn't work with JAX. JAX itself has some support for named axes through [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html).

`spekk` is not coupled with either NumPy or JAX — it exists independently of the underlying array.


## Speccing an array
Here's an example of how to use `spekk.Shape`:
```python
import numpy as np
from spekk import Shape

channel_data = np.random.normal(size=(5, 75, 128, 1000))
channel_data_shape = Shape("frames", "transmits", "receivers", "signal_time")

# Sum over transmits
channel_data = np.sum(channel_data, axis=channel_data_shape.index("transmits"))
channel_data_shape -= "transmits"  # Remove the transmits dimension from the shape

# Sum over receivers
channel_data = np.sum(channel_data, axis=channel_data_shape.index("receivers"))
channel_data_shape -= "receivers"  # Remove the receivers dimension from the shape
```

## Speccing a function
Here's an example of how to use `spekk.Spec`:
```python
import jax.numpy as jnp
import jax
from spekk import Spec

def beamform(channel_data, focus_points, receiver_positions):
    # Performs beamforming
    ...

beamform_spec = Spec(
    {
        "channel_data": ["frames", "transmits", "receivers", "signal_time"],
        "focus_points": ["transmits", "xyz"],
        "receiver_positions": ["receivers", "xyz"],
    }
)

# Vectorize over all transmits
beamform_transmits = jax.vmap(beamform, in_axes=beamform_spec.indices_for("transmits"))

channel_data = jnp.random.normal(size=(5, 75, 128, 1000))
focus_points = jnp.random.normal(size=(75, 3))
receiver_positions = jnp.random.normal(size=(128, 3))
result = beamform_transmits(channel_data, focus_points, receiver_positions)
```

In the `spekk.Spec` example we have multiple arrays that share some dimensions. For example, both the `channel_data` and `focus_points` have a `"transmits"` dimension.