from functools import partial
from typing import Union

from spekk import ops, util
from spekk.ops._types import Dim
from spekk.ops.array_object import array
from spekk.util.profiling import function_profiling


def windowed(
    f=None,
    window_sizes: dict[Dim, int] = None,
    pad_mode: Union[str, None] = None,
    reflect_type: Union[str, None] = None,
):
    if f is None:
        return partial(
            windowed,
            window_sizes=window_sizes,
            pad_mode=pad_mode,
            reflect_type=reflect_type,
        )

    def wrapped(data: ops.array, *args, **kwargs):
        if pad_mode is not None:
            pad_width = [(size // 2, size // 2) for size in window_sizes.values()]
            dims = list(window_sizes.keys())
            data = ops.pad(
                data,
                pad_width=pad_width,
                mode=pad_mode,
                reflect_type=reflect_type,
                dims=dims,
            )
            pad_half_width = {key: v // 2 for key, v in window_sizes.items()}
        else:
            pad_half_width = {key: 0 for key, v in window_sizes.items()}

        window_dim_names = [
            util.random_dim_name(text=dim) for dim in window_sizes.keys()
        ]
        window_index_offsets = {
            dim: ops.arange(-(size // 2), (size // 2) + 1, dim=dim_name)
            for dim_name, (dim, size) in zip(window_dim_names, window_sizes.items())
        }
        indices = {
            dim: ops.arange(
                pad_half_width[dim], data.dim_sizes[dim] - pad_half_width[dim], dim=dim
            )
            + offsets
            for dim, offsets in window_index_offsets.items()
        }
        return f(data[indices], *args, axis=tuple(window_dim_names), **kwargs)

    return wrapped


@function_profiling
def median_filter(
    image: array, window_sizes: dict[Dim, int], pad_mode: Union[str, None] = "edge"
) -> array:
    return windowed(ops.median, window_sizes, pad_mode)(image)


@function_profiling
# @ops.jit(static_argnames=("spatial_sigmas",))
def bilateral_filter(
    image: array, *, spatial_sigmas: dict[Dim, float], color_sigma: float
) -> array:
    f = partial(
        bilateral_filter_kernel, spatial_sigmas=spatial_sigmas, color_sigma=color_sigma
    )

    # window_sizes should be odd
    window_sizes = {key: 2 * int(2 * v) + 1 for key, v in spatial_sigmas.items()}

    return windowed(f, window_sizes, pad_mode="reflect", reflect_type="even")(image)


def bilateral_filter_kernel(
    img_pad: array,
    axis: tuple,
    spatial_sigmas: dict[Dim, float],
    color_sigma: float,
) -> array:
    # Pre-compute
    scaleFactor_color = 1 / (2 * color_sigma * color_sigma)

    # rename axis in "spatial_sigmas" to match new axis in "img_pad"
    spatial_sigmas = {
        img_pad.dims[img_pad.dims.index(key) + 1]: v
        for key, v in spatial_sigmas.items()
    }

    X = ops.meshgrid(
        *(
            ops.arange(
                -(img_pad.dim_sizes[ax] - 1) // 2,
                (img_pad.dim_sizes[ax] - 1) // 2 + 1,
                dim=ax,
            )
            / spatial_sigmas[ax]
            for ax in axis
        )
    )
    kernel_spatial = ops.exp(-0.5 * sum(x * x for x in X))

    roi_centers = {ax: (img_pad.dim_sizes[ax] - 1) // 2 for ax in axis}
    weight = (
        ops.exp(-((img_pad - img_pad[roi_centers]) ** 2) * scaleFactor_color)
        * kernel_spatial
    )
    out = ops.sum(weight * img_pad, axis=axis) / ops.sum(weight, axis=axis)

    return out


# @function_profiling
def convNd(image: array, kernel: array, pad_mode="edge") -> array:
    """
    Args:
        image: Input array
        kernel: The kernel used, note that if the kernel includes an axis not present in image,
                that axis will be seen as different set of convolution filters.

    """
    window_sizes = {}
    for dim in kernel.dims:
        if dim in image.dims:
            window_sizes[dim] = kernel.dim_sizes[dim]

    # window_sizes should be odd
    check_odd_items(window_sizes)

    f = partial(convNd_kernel, kernel=kernel)

    return windowed(f, window_sizes, pad_mode=pad_mode)(image)


def convNd_kernel(img_pad: array, axis: tuple, kernel: array) -> array:
    # rename axis in "kernel" to match new axis in "img_pad"
    for dim in kernel.dims:
        if dim in img_pad.dims:
            dim_index = img_pad.dims.index(dim) + 1
            kernel = kernel.rename_dim(dim, img_pad.dims[dim_index])

    return ops.sum(img_pad * kernel, axis=axis)


def check_odd_items(dictionary):
    for key, value in dictionary.items():
        if value % 2 == 0:
            raise ValueError(f"Item with key '{key}' has an even value: {value}")


if __name__ == "__main__":
    import time

    import cfm
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    # from cfm.utils import profiling
    # profiling.set_level( profiling.Level.Warning)

    np.random.seed(42)
    # img_gray = np.random.uniform(low=0.0, high=10.0, size=(25, 20))
    # img_gray = np.random.uniform(low=0.0, high=10.0, size=(36, 149))
    img_gray = np.random.uniform(low=0.0, high=10.0, size=(600, 600))
    img_gray[5:15, 5:10] = img_gray[5:15, 5:10] + 5
    img_gray = img_gray.astype(np.float32)

    # with ops.backend.temporary_backend("numpy"):

    img_gray_spekk = ops.array(img_gray, dims=["x", "y"])
    img_gray_spekk = ops.expand_dims(img_gray_spekk, axis="dummy")
    img_gray_spekk = ops.concat(
        (img_gray_spekk, img_gray_spekk, img_gray_spekk), axis="dummy"
    )

    # kernel = ops.array([-0.25, 0.5, 0.25])
    r = 3
    x, y = ops.meshgrid(ops.arange(-r, r + 1, dim="x"), ops.arange(-r, r + 1, dim="y"))
    kernel = ops.exp(-(x * x + y * y))
    kernel = kernel / ops.sum(kernel)

    print(img_gray_spekk.dim_sizes)
    print(kernel.dim_sizes)
    # tic = time.time()
    partial = ops.jit(partial(convNd, kernel=kernel))
    img_ref = convNd(img_gray_spekk, kernel)
    # print(f"bilateralfilter_ref = {time.time() - tic:0.5f}")

    # tic = time.time()
    # kernel = kernel
    img_ref = convNd(img_gray_spekk, kernel)
    img_ref = convNd(img_gray_spekk, kernel)
    img_ref = convNd(img_gray_spekk, kernel)
    # print(f"bilateralfilter_ref_jitted = {time.time() - tic:0.5f}")

    # tic = time.time()
    # r = 4
    # x, y = ops.meshgrid(ops.arange(-r, r + 1, dim="x"), ops.arange(-r, r + 1, dim="y"))
    # kernel = ops.exp(-(x * x + y * y))
    # kernel = kernel / ops.sum(kernel)

    # img_ref = convNd(img_gray_spekk+2, kernel)
    # print(f"bilateralfilter_ref_jitted = {time.time() - tic:0.5f}")

    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # vmin, vmax = img_gray.min(), img_gray.max()

    # im = ax[0].imshow(img_gray, vmin=vmin, vmax=vmax)
    # ax[0].set_aspect("auto")
    # ax[0].set_title("img_gray")
    # fig.colorbar(im, ax=ax[0])

    # im = ax[1].imshow(ops.to_numpy(img_ref), vmin=vmin, vmax=vmax)
    # ax[1].set_aspect("auto")
    # ax[1].set_title("img_filt")
    # fig.colorbar(im, ax=ax[1])

    # plt.show()
    # a = 1
