import numpy as np
from spekk.backends.protocol import Backend


class NumpyBackend(Backend):
    array = np.array
    take = np.take
    transpose = np.transpose

    def vmap(self, fun, in_axes, out_axes=0):
        v_axes = [(i, ax) for i, ax in enumerate(in_axes) if ax is not None]

        def vectorized_fun(*args, **kwargs):
            v_sizes = [args[i].shape[ax] for i, ax in v_axes]
            v_ax_size = v_sizes[0]
            assert all(
                [v_size == v_ax_size for v_size in v_sizes]
            ), "All vectorized axes must have the same number of elements."

            results = []
            for i in range(v_ax_size):
                new_args = [
                    args[j] if ax is None else i_at(args[j], i, ax)
                    for j, ax in enumerate(in_axes)
                ]
                results.append(fun(*new_args, **kwargs))
            results = recombine(results)
            results = set_out_axes(results, out_axes)
            return results

        return vectorized_fun


def i_at(arr, i, ax):
    indices = (slice(None),) * ax + (i,)
    return arr[indices]


def recombine(results: list):
    if isinstance(results[0], tuple):
        assert all([isinstance(result, tuple) for result in results])
        n_items = len(results[0])
        return tuple([recombine([r[i] for r in results]) for i in range(n_items)])
    else:
        return np.array(results)


def set_out_axes(result: np.ndarray, out_axis: int):
    if isinstance(result, tuple):
        return tuple([set_out_axes(r, out_axis) for r in result])
    else:
        return np.moveaxis(result, 0, out_axis)
