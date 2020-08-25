"""Integration routines combined."""
from typing import Optional
from numpy import ndarray, array

from scipy.interpolate import RectBivariateSpline


def bi_interpolate(
    matrix: ndarray,
    p1: ndarray,
    p2: ndarray,
    index: Optional[ndarray] = None,
    kx: int = 3,
    ky: int = 3,
):
    """Runs bivariate interpolation on matrix second and third dimension.

    Interpolation runs for all entries in first dimension.
    """
    if not len(matrix.shape) == 3:
        raise ValueError("Routine requires matrix to be of shape (n, m, m)")

    if not len(p1) == matrix.shape[1] == matrix.shape[2]:
        raise ValueError(
            "Routine requires matrix to be of shape (n, m, m) where m = len(p1)"
        )

    iterator = (
        RectBivariateSpline(p1, p1, kernel, kx=kx, ky=ky)(p2, p2)
        for kernel in (matrix if index is None else matrix[index])
    )
    return array(list(iterator))
