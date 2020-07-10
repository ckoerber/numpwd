"""Abstract implementation of densities."""
from typing import Dict
from dataclasses import dataclass, field

from numpy import array, ndarray
from pandas import DataFrame

CHANNEL_COLUMNS = [
    "l_o",
    "s_o",
    "j_o",
    "t_o",
    "mt_o",
    "mj_o",
    "mjtotx2_o",
    "l_i",
    "s_i",
    "j_i",
    "t_i",
    "mt_i",
    "mj_i",
    "mjtotx2_i",
]


@dataclass
class Density:
    """Implementation of densities."""

    #: Returns total spin of the Density.
    jx2: int = None

    #: Returns operator matrix.
    #:  The first index represents the collective channel, the second the out momentum
    #:  and the third th in momentum.
    matrix: ndarray = field(repr=None, default_factory=lambda: array([]))

    #: Momentum mesh of the oparator (1-D).
    p: ndarray = field(repr=None, default_factory=lambda: array([]))

    #: Momentum mesh weigths of the oparator (1-D).
    wp: ndarray = field(repr=None, default_factory=lambda: array([]))

    #: In and out channels of the Density.
    #:  Must provide "l", "s", "j", "t", "mt", "mj", "mjtotx2" with suffix '_i' and '_o'
    #:  for in and outgoing quantum numbers. Index is the index to the matrix first
    #:  dimension.
    channels: DataFrame = field(
        repr=None, default_factory=lambda: DataFrame(data=[], columns=CHANNEL_COLUMNS)
    )

    #: Additional infos about the mesh to reproduce it.
    mesh_info: Dict[str, float] = field(default_factory=dict)

    #: Additional infos about the external current momentum.
    current_info: Dict[str, float] = field(default_factory=dict)

    #: Additional information
    misc: Dict[str, float] = field(default_factory=dict)

    def check(self):
        """Runs checks if density was properly initialized."""
        if self.jx2 is None:
            raise ValueError("jx2 was not set.")

        if not isinstance(self.matrix, ndarray) or len(self.matrix) == 0:
            raise ValueError("Matrix was not initialized")

        if len(self.matrix.shape) != 3:
            raise ValueError("Matrix of wrong shape")

        if not isinstance(self.p, ndarray) or len(self.p) != len(self.wp):
            raise ValueError("Momentum values and weights not of same shape.")

        if not len(self.p) == self.matrix.shape[1] == self.matrix.shape[2]:
            raise ValueError("Momenum vector shape does not match matrix.")

        if not isinstance(self.channels, DataFrame):
            raise ValueError("Channels not initialized.")

        if set(self.channels.columns) != set(CHANNEL_COLUMNS):
            raise KeyError("Channel columns do not contain correct keys.")

        if self.channels.shape[0] != self.matrix.shape[0]:
            raise ValueError("Channel shape does not match matrix shape.")
