"""Abstract implementation of densities."""
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

from numpy import array, ndarray
from pandas import DataFrame

CHANNEL_COLUMNS = [
    "l_o",
    "l_i",
    "s_o",
    "s_i",
    "j_o",
    "j_i",
    "mj_o",
    "mj_i",
    "ms_ex_o",
    "ms_ex_i",
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
    #:  Must provide "l", "s", "j", "mj", "ms_ex" with suffix '_i' and '_o'
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
        """Runs checks if density was properly initialized."""
        if not isinstance(self.matrix, ndarray) or len(self.matrix) == 0:
            raise ValueError("Matrix was not initialized")

        if len(self.matrix.shape) != len(self.args) + 1:
            raise ValueError("Matrix of wrong shape")

        if not isinstance(self.channels, DataFrame) or self.channels.shape[0] == 0:
            raise ValueError("Channels not initialized.")

        if set(self.channels.columns) != set(CHANNEL_COLUMNS):
            raise KeyError("Channel columns do not contain correct keys.")

        if self.channels.shape[0] != self.matrix.shape[0]:
            raise ValueError("Channel shape does not match matrix shape.")

        for n, (key, val) in enumerate(self.args):
            if len(val) != self.matrix.shape[1 + n]:
                raise ValueError(
                    f"Operator argument {key} shape does not match matrix shape."
                )

        if not self.isospin:
            raise KeyError("Isospin matrix is empty")
