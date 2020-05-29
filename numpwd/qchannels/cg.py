"""Cached version of sympys cg functions
"""
from typing import Union
from functools import lru_cache

from sympy import Symbol
from sympy.physics.quantum.cg import CG

Number = Union[int, float, Symbol]


@lru_cache(maxsize=128)
def get_cg(  # pylint: disable=too-many-arguments
    j1: Number,
    m1: Number,
    j2: Number,
    m2: Number,
    j3: Number,
    m3: Number,
    numeric: bool = False,
    doit: bool = True,
) -> Number:
    """Wrapper for sympy's CG but wrapps input

    Defintions:
        cg = <j1 m1, j2 m2 | j3, m3>

    Arguments:
        quantum numbers: ...
        numeric: Converts evaluate cg to float
        doit: Evaluates the cg coeffient

    Asserts:
        * j's are larger equal zero
        * m's are smaller eqaul j's
        * m's and j's are integer or half integer
        * j's + m's are integer
    """
    assert (
        j1 >= 0
        and j2 >= 0
        and j3 >= 0
        and abs(m1) <= j1
        and abs(m2) <= j2
        and abs(m3) <= j3
        and isinstance(j1 * 2, int)
        and isinstance(j2 * 2, int)
        and isinstance(j3 * 2, int)
        and isinstance(m1 * 2, int)
        and isinstance(m2 * 2, int)
        and isinstance(m3 * 2, int)
        and isinstance(j1 + m1, int)
        and isinstance(j2 + m2, int)
        and isinstance(j3 + m3, int)
    )

    if (
        not abs(j1 - j2) <= j3 <= j1 + j2
        or m1 + m2 != m3
        or (m1 == m2 == m3 == 0 and (j1 + j2 - j3) % 2 == 1)
    ):
        # if all m signs flip, the CG is the same up to (-1)**(l+s-j)
        # thus if all m==0, l+s-j must be even
        cg = 0
    else:
        cg = CG(j1, m1, j2, m2, j3, m3)
        cg = cg.doit() if doit or numeric else cg
        cg = cg.evalf() if numeric else cg

    return cg
