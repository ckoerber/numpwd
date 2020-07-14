# pylint:disable=C0103
"""Cached version of sympys cg functions
"""
from typing import Union, Iterable
from functools import lru_cache

from numpy import issubdtype, integer

from sympy import Number as SympyNumber
from sympy.physics.quantum.cg import CG

Number = Union[int, float, SympyNumber]


def get_j_range(j1: int, j2: int) -> Iterable[int]:
    """Computes the range of allowed j quantum numbers for given j input.

    Asserts j1, j2 is an integer laerger equal zero
    """
    assert is_int(j1) and j1 >= 0
    assert is_int(j2) and j2 >= 0
    return range(abs(j1 - j2), j1 + j2 + 1)


class QuantumNumberError(Exception):
    """Custom exception for checking relations in quantum numbers
    """

    def __init__(self, message: str, **kwargs):
        self.data = kwargs
        self.message = message
        if self.data:
            self.message += "\nData:\n\t- " + "\n\t- ".join(
                [f"{key}={val}" for key, val in self.data.items()]
            )

        super().__init__(self.message)


def is_int(nn: Number) -> bool:
    """Checks if an expression is of integer type
    """
    return (
        isinstance(nn, int)  # Check generic int
        or (isinstance(nn, SympyNumber) and nn.is_integer)  # Check sympy
        or issubdtype(nn, integer)  # Check numpy
    )


def check_jm(j, m, **context):
    """Checks j and m qunatum numbers

    Asserts:
        * j's are larger equal zero
        * m's are smaller eqaul j's
        * m's and j's are integer or half integer
        * j's + m's are integer

    Raises:
        QuantumNumberError
    """
    if j < 0:
        raise QuantumNumberError("j-value smaller 0", j=j, m=m, **context)
    if abs(m) > j:
        raise QuantumNumberError(
            "Absolute of m-value larger than j", j=j, m=m, **context
        )
    if not is_int(j * 2):
        raise QuantumNumberError(
            "j*2 is not an integer", j=j, m=m, type_j=type(j), **context
        )
    if not is_int(m * 2):
        raise QuantumNumberError(
            "m*2 is not an integer", j=j, m=m, type_m=type(m), **context
        )
    if not is_int(j + m):
        raise QuantumNumberError(
            "j+m is not an integer", j=j, m=m, type_j=type(j), type_m=type(m), **context
        )


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
    for n, jm in enumerate([(j1, m1), (j2, m2), (j3, m3)]):
        check_jm(*jm, n=n)

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
        cg = float(cg) if numeric else cg

    return cg
