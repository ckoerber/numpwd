"""Numerical integration mesh routines
"""
from typing import Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss


def get_trns_mesh(  # pylint: disable=C0103
    n1: int, n2: int, p1: float = 1.0, p2: float = 6.0, p3: float = 20.0
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Allocates transformed Gauss-Legendre mesh for numerical integration.

    The total interval spans from p0 = 0.0 to p3. The points are distributed as follows:

    * n1 / 2 points in (p0, p1)
    * n1 / 2 points in (p1, p2)
    * n2  points in (p2, p3)

    The (p0, p2) interval follows the hyperbolic transformation
    $$
    x\_1 \\to
    \\frac{1 + x\_1}
    {\\frac{1}{p\_1} - x\_1\\left(\\frac{1}{p\_1} -\\frac{1}{p\_2} \\right)}
    $$

    and the (p1, p2) interval follows the linear transformation
    $$ x\_2 \\to \\frac{p\_3 + p\_2}{2} + x\_2 \\frac{p\_3 - p\_2}{2}$$


    **Arguments**
        n1: int
            Number of meshpoints for first interval (hyperbolic transformation).

        n2: int
            Number of meshpoints for second interval (linear transformation).

        p1: float = 1.0
            Middle point (defined by number of points) of first interval.

        p2: float = 6.0
            End point of first interval.

        p3: float = 20.0
            End point of second interval.

    **Returns**
        p, wp: Tuple[np.ndarray, np.ndarray]
            The meshpoints and the integration weights.

    .. note:: This routine follows Andreas Nogga's implementation of the TRNS routine.
    """
    x = []
    wx = []

    xtmp, wtmp = leggauss(n1)
    for xi, wi in zip(xtmp, wtmp):
        xxi = 1 / p1 - (1 / p1 - 2 / p2) * xi
        x.append((1 + xi) / xxi)
        wx.append((2 / p1 - 2 / p2) * wi / xxi ** 2)

    xtmp, wtmp = leggauss(n2)
    p23 = (p3 - p2) / 2
    P23 = (p3 + p2) / 2
    for xi, wi in zip(xtmp, wtmp):
        x.append(P23 + p23 * xi)
        wx.append(p23 * wi)

    return np.array(x), np.array(wx)
