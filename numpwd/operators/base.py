"""Abstract implementation of densities."""
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

from numbers import Number

from numpy import array, ndarray, allclose, zeros
from pandas import DataFrame, merge

try:
    import cupy as cp
except ImportError:
    cp = None

CHANNEL_COLUMNS = [
    "l_o",
    "l_i",
    "s_o",
    "s_i",
    "j_o",
    "j_i",
    "mj_o",
    "mj_i",
]


@dataclass(repr=None)
class Operator:
    """Implementation of external operators."""

    #: Returns operator matrix.
    #:  The first index represents the collective channel, the second the out momentum,
    #:  the third the in momentum and the last the external momentum.
    matrix: ndarray = field(default_factory=lambda: array([]))

    #: isospin matrix of operator for spin-0 and spin-1 systems.
    #   Keys correspond to (t_o, mt_o, ti, mt_i)
    isospin: Dict[Tuple[int, int, int, int], float] = field(default_factory=dict)

    #: Arguments of the opertor tensor (e.g., [("p_o", [1,2,3,4]), ...]).
    args: List[Tuple[str, ndarray]] = field(default_factory=list)

    #: In and out channels of the Density.
    #:  Must provide "l", "s", "j", "mj" with suffix '_i' and '_o'
    #:  for in and outgoing quantum numbers. Index is the index to the matrix first
    #:  dimension.
    channels: DataFrame = field(
        default_factory=lambda: DataFrame(data=[], columns=CHANNEL_COLUMNS)
    )

    #: Additional infos about all utilized meshes to reproduce them.
    mesh_info: Dict[str, float] = field(default_factory=dict)

    #: Additional information not directly needed for the computation
    misc: Dict[str, float] = field(default_factory=dict)

    def check(self):
        """Run checks if density was properly initialized."""
        if not isinstance(self.matrix, ndarray) or len(self.matrix) == 0:
            raise ValueError("Matrix was not initialized")

        if len(self.matrix.shape) != len(self.args) + 1:
            raise ValueError("Matrix of wrong shape")

        if not isinstance(self.channels, DataFrame) or self.channels.shape[0] == 0:
            raise ValueError("Channels not initialized.")

        if set(CHANNEL_COLUMNS) - set(self.channels.columns):
            raise KeyError("Channel columns do not contain correct keys.")

        if self.channels.shape[0] != self.matrix.shape[0]:
            raise ValueError("Channel shape does not match matrix shape.")

        for n, (key, val) in enumerate(self.args):
            if len(val) != self.matrix.shape[1 + n]:
                raise ValueError(
                    f"Operator argument {key} shape does not match matrix shape."
                )

        if self.isospin is None or len(self.isospin) == 0:
            raise KeyError("Isospin matrix is empty")

    def __eq__(self, other):
        """Check if all attributes are equal within numeric precision."""
        if not isinstance(other, Operator):
            return NotImplemented

        if self.mesh_info != other.mesh_info:
            return False

        if self.misc != other.misc:
            return False

        if len(self.args) == len(other.args):
            for (k1, v1), (k2, v2) in zip(self.args, other.args):
                if k1 != k2 or not allclose(v1, v2, rtol=1.0e-12, atol=0.0):
                    return False

        if not self.channels.equals(other.channels):
            return False

        if isinstance(self.isospin, DataFrame):
            if not self.isospin.equals(other.isospin):
                return False
        else:
            return self.isospin == other.isospin

        if not allclose(self.matrix, other.matrix, rtol=1.0e-12, atol=0.0):
            return False

        return True

    def to_gpu(self):
        """Move all matrix components to the GPU if possibe.

        Moves args and matrix attrubitue to gpu.
        Raises import error if cupy not present.
        """
        if cp is None:
            raise ImportError("Failed to import cupy")

        self.matrix = cp.array(self.matrix)
        new_args = []
        for arg in self.args:
            new_args.append((arg[0], cp.array(arg[1])))
        self.args = new_args

    def to_cpu(self):
        """Move all matrix components to the CPU if possibe.

        Moves args and matrix attrubitue to cpu.
        """
        if isinstance(self.matrix, cp.ndarray):
            self.matrix = self.matrix.get()
        new_args = []
        for arg in self.args:
            key, val = arg
            if isinstance(val, cp.ndarray):
                val = val.get()
            new_args.append((key, val))
        self.args = new_args

    def __add__(self, other: "Operator"):
        """Add two operators to form a new operator.

        This includes merging channels and reindexing the matrix as well as
        joining keywords.

        The user should make sure that operators are logically mergeable--this routine
        just checks matrix shapes.
        """
        return add(self, other, check=False)

    def copy(self):
        """Create a new operator copied from this instance."""
        op = Operator()
        for key in ["channels", "matrix", "isospin", "args", "misc", "mesh_info"]:
            setattr(op, key, getattr(self, key).copy())
        return op

    def __mul__(self, number: Number):
        """Multiply operator matrix with a factor."""
        if not isinstance(number, Number):
            raise TypeError(f"{number} must be a number")
        op = self.copy()
        op.matrix *= number
        op.misc["factor"] = number
        return op


def add(op1: Operator, op2: Operator, check: bool = False):
    """Add two operators to form a new operator.

    This includes merging channels and reindexing the matrix as well as joining keywords.

    The user should make sure that operators are logically mergeable--this routine
    just checks matrix shapes.
    """
    if check:
        op1.check()
        op2.check()

    assert isinstance(op1, Operator)
    assert isinstance(op2, Operator)

    shape1 = list(op1.matrix.shape[1:])
    shape2 = list(op2.matrix.shape[1:])

    assert shape1 == shape2
    assert op1.matrix.dtype == op2.matrix.dtype

    for (k1, v1), (k2, v2) in zip(op1.args, op2.args):
        assert k1 == k2
        assert allclose(v1, v2)

    if isinstance(op1.isospin, DataFrame):
        assert not op1.isospin.equals(op2.isospin)
    else:
        iso_equals = op1.isospin == op2.isospin
        if hasattr(iso_equals, "__iter__"):
            assert all(iso_equals)
        else:
            assert iso_equals

    new_channels = merge(
        op1.channels.reset_index(),
        op2.channels.reset_index(),
        how="outer",
        left_on=CHANNEL_COLUMNS,
        right_on=CHANNEL_COLUMNS,
        suffixes=("1", "2"),
    )
    idx1 = new_channels["index1"].fillna(-1).astype(int)
    idx2 = new_channels["index2"].fillna(-1).astype(int)
    new_channels = new_channels.drop(["index1", "index2"], axis=1)

    shape = tuple([len(new_channels)] + shape1)

    m1 = zeros(shape, dtype=op1.matrix.dtype)
    m1[idx1 > -1] = op1.matrix[idx1[idx1 > -1]]

    m2 = zeros(shape, dtype=op2.matrix.dtype)
    m2[idx2 > -1] = op2.matrix[idx2[idx2 > -1]]

    op = Operator()
    op.args = op1.args
    op.channels = new_channels
    op.matrix = m1 + m2
    op.isospin = op1.isospin
    op.misc["type"] = "Operator sum"
    op.misc["op left"] = op1.misc
    op.misc["op right"] = op2.misc
    op.mesh_info["op left"] = op1.mesh_info
    op.mesh_info["op right"] = op2.mesh_info

    if check:
        op.check()

    return op
