"""Abstract implementation of densities."""
from typing import Dict
from dataclasses import dataclass, field

from numpy import array, ndarray
from pandas import DataFrame


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
    channels: DataFrame = field(repr=None, default_factory=DataFrame)

    #: Additional infos about the mesh to reproduce it.
    mesh_info: Dict[str, float] = field(default_factory=dict)

    #: Additional infos about the external current momentum.
    current_info: Dict[str, float] = field(default_factory=dict)

    #: Additional information
    misc: Dict[str, float] = field(default_factory=dict)
