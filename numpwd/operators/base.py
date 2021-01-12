"""Abstract implementation of densities."""
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

from numpy import array, ndarray, allclose
from pandas import DataFrame

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

        if not self.isospin.equals(other.isospin):
            return False

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
        self.matrix = array(self.matrix)
        new_args = []
        for arg in self.args:
            new_args.append((arg[0], array(arg[1])))
        self.args = new_args
