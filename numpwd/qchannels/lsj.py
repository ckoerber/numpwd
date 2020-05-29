"""Computations & help functions used for spin-orbit couplings
"""
from typing import Iterable, Generator, Dict, Optional

from itertools import product

from numpwd.qchannels.cg import get_cg, Number


def m_range(j: int) -> Iterable[int]:
    """Computes the range of allowed m quantum numbers for given j input

    Asserts j is an integer laerger equal zero
    """
    assert isinstance(j, int) and j >= 0
    return range(-j, j + 1)


def get_two_n_channels(
    l_max: int, s_max: int = 1, j_max: Optional[int] = None, numeric: bool = False
) -> Generator[Dict[str, Number]]:
    """Computes all non-zero (ls)j channels and corresponding clebsch gordan coefficients.

    Arguments:
        l_max: Maximal angular momentum quantum number
        s_max: Maximal spin qunatum number
        j_max: Maximal total spin quantum number. Defaults to l_max + s_max
        numeric: Return type of the CG; float if True, else sympy expression

    Asserts:
        Input quantum numbers are integers
    """
    j_max = l_max + s_max if j_max is None else j_max

    assert (
        l_max > 0
        and isinstance(l_max, int)
        and s_max > 0
        and isinstance(s_max, int)
        and j_max > 0
        and isinstance(j_max, int)
    )

    for l, s, j in product(*map(range, [l_max + 1, s_max + 1, j_max + 1])):
        for ml, ms, mj in product(*map(m_range, [l, s, j])):

            yield {
                "l": l,
                "ml": ml,
                "s": s,
                "ms": ms,
                "j": j,
                "mj": mj,
                "cg": get_cg(l, ml, s, ms, j, mj, numeric=numeric),
            }
