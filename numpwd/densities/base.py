"""Abstract implementation of densities."""
from typing import Dict
from abc import ABCMeta

from numpy import ndarray
from pandas import DataFrame


class Density(metaclass=ABCMeta):
    """Abstract implementation of densities."""

    #: Additional information
    misc: Dict[str, float] = dict()

    #: Returns total spin of the Density.
    jx2: int = None

    #: Returns operator matrix.
    #:  The first index represents the collective channel, the second the out momentum
    #:  and the third th in momentum.
    matrix: ndarray = None

    #: Momentum mesh of the oparator (1-D).
    p: ndarray = None

    #: Momentum mesh weigths of the oparator (1-D).
    wp: ndarray = None

    #: Additional infos about the mesh to reproduce it.
    mesh_info: Dict[str, float] = None

    #: Additional infos about the external current momentum.
    current_info: Dict[str, float] = None

    #: In and out channels of the Density.
    #:  Must provide "l", "s", "j", "t", "mt", "mj", "mjtotx2" with suffix '_i' and '_o'
    #:  for in and outgoing quantum numbers. Index is the index to the matrix first
    #:  dimension.
    channels: DataFrame = None
