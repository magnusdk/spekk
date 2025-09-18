from typing import Callable, List

from spekk import ops

binary_operations: List[Callable[[ops.array, ops.array], ops.array]] = [
    ops.add,
    ops.atan2,
    ops.bitwise_and,
    ops.bitwise_left_shift,
    ops.bitwise_or,
    ops.bitwise_right_shift,
    ops.bitwise_xor,
    ops.copysign,
    ops.divide,
    ops.equal,
    ops.floor_divide,
    ops.greater,
    ops.greater_equal,
    ops.hypot,
    ops.less,
    ops.less_equal,
    ops.logaddexp,
    ops.logical_and,
    ops.logical_or,
    ops.logical_xor,
    ops.maximum,
    ops.minimum,
    ops.multiply,
    ops.not_equal,
    ops.pow,
    ops.remainder,
    ops.subtract,
]


def test_binary_operations():
    for op in binary_operations:
        result = op(
            ops.ones((3, 4), dims=["d3", "d4"], dtype="int32"),
            ops.ones((4, 5), dims=["d4", "d5"], dtype="int32"),
        )
        assert result.dim_sizes == {"d3": 3, "d4": 4, "d5": 5}


def test_linspace():
    result = ops.linspace(
        ops.ones((3, 4), dims=["d3", "d4"]),
        ops.ones((4, 5), dims=["d4", "d5"]),
        10,
        dim="d10",
    )
    assert result.dim_sizes == {"d3": 3, "d4": 4, "d5": 5, "d10": 10}


def test_stack():
    result = ops.stack(
        [
            ops.ones((3, 4), dims=["d3", "d4"]),
            ops.ones((4, 5), dims=["d4", "d5"]),
        ],
        axis="stacked_dim",
    )
    assert result.dim_sizes == {"d3": 3, "d4": 4, "d5": 5, "stacked_dim": 2}


def test_concat():
    result = ops.concat(
        [
            ops.ones((3, 10), dims=["d3", "concat_dim"]),
            ops.ones((2, 5), dims=["concat_dim", "d5"]),
        ],
        axis="concat_dim",
    )
    assert result.dim_sizes == {"d3": 3, "concat_dim": 12, "d5": 5}


def test_clip():
    result = ops.clip(
        ops.ones((3, 4), dims=["d3", "d4"]),
        ops.ones((4, 5), dims=["d4", "d5"]),
        ops.ones((2,), dims=["d2"]),
    )
    assert result.dim_sizes == {"d3": 3, "d4": 4, "d5": 5, "d2": 2}


def test_cross():
    result = ops.linalg.cross(
        ops.ones((3, 4), dims=["d3", "d4"]),
        ops.ones((4, 3, 5), dims=["d4", "d3", "d5"]),
        axis="d3",
    )
    assert result.dim_sizes == {"d3": 3, "d4": 4, "d5": 5}


def test_vecdot():
    result = ops.vecdot(
        ops.ones((3, 4), dims=["d3", "d4"]),
        ops.ones((4, 5), dims=["d4", "d5"]),
        axis="d4",
    )
    assert result.dim_sizes == {"d3": 3, "d5": 5}


def test_where():
    result = ops.where(
        ops.full((3, 4), True, dims=["d3", "d4"]),
        ops.ones((4, 5), dims=["d4", "d5"]),
        ops.ones((2,), dims=["d2"]),
    )
    assert result.dim_sizes == {"d3": 3, "d4": 4, "d5": 5, "d2": 2}
