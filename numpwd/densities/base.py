"""Abstract implementation of densities."""
from typing import Dict
from abc import ABC, abstractmethod

from numpy import ndarray
from pandas import DataFrame


class Density(ABC):
    """Abstract implementation of densities."""

    misc: Dict[str, float] = dict()

    @property
    @abstractmethod
    def jx2(self) -> int:
        """Returns total spin of the Density."""
        pass

    @property
    @abstractmethod
    def matrix(self) -> ndarray:
        """Returns operator matrix.

        The first index represents the collective channel, the second the out momentum
        and the third th in momentum.
        """
        pass

    @property
    @abstractmethod
    def p(self) -> ndarray:
        """Momentum mesh of the oparator (1-D)."""
        pass

    @property
    @abstractmethod
    def wp(self) -> ndarray:
        """Momentum mesh weigths of the oparator (1-D)."""
        pass

    @property
    @abstractmethod
    def mesh_info(self) -> Dict[str, float]:
        """Additional infos about the mesh to reproduce it."""
        pass

    @property
    @abstractmethod
    def current_info(self) -> Dict[str, float]:
        """Additional infos about the external current momentum."""
        pass

    @property
    @abstractmethod
    def channels(self) -> DataFrame:
        """In and out channels of the Density.

        Must provide "l", "s", "j", "t", "mt", "mj", "mjtotx2" with suffix '_i' and '_o'
        for in and outgoing quantum numbers. Index is the index to the matrix first
        dimension.
        """
        pass
