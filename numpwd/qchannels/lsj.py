# pylint:disable=C0103
"""Computations & help functions used for spin-orbit couplings
"""
from typing import Iterable, Generator, Dict, Optional, List, Callable

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
) -> List[Dict[str, Number]]:
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

    channels = []
    for l, s, j in product(*map(range, [l_max + 1, s_max + 1, j_max + 1])):
        for ml, ms, mj in product(*map(m_range, [l, s, j])):

            cg = get_cg(l, ml, s, ms, j, mj, numeric=numeric)
            if abs(cg) > 1.0e-7:
                channels.append(
                    {"l": l, "ml": ml, "s": s, "ms": ms, "j": j, "mj": mj, "cg": cg}
                )

    return channels


def generate_matrix_channels(
    two_n_channels: Dict[str, Number],
    allowed_transitions: Optional[
        List[Callable[[Dict[str, Number], Dict[str, Number]], bool]]
    ] = None,
) -> Generator[Dict[str, Number], None, None]:
    """Generates all in and out two-n channles for input two_n_channels

    Arguments:
        two_n_channels: The (ls)j coupled two-n channels generated by `get_two_n_channels`.
        allowed_transitions: List of callables which specify allowed transitions.
            Callable takes input and outpu channel dicts as input and returns a bool.
            For example for only 'l==0' transitions one has
                lambda c_i, c_o: c_i["l"] == 0 and c_o["l"] == 0
            Argument defaults to all channels allowed.

    Returns:
        Generator for each channel (dicts) where in channels contain same keys
        as in allowed_transitions with an addional '_i' or '_o' marking in or out.
    """

    allowed_transitions = allowed_transitions or [lambda ci, co: True]
    for c_i, c_o in product(two_n_channels, two_n_channels):
        if any(
            [transition_allowed(c_i, c_o) for transition_allowed in allowed_transitions]
        ):
            data = {"cg": c_i["cg"] * c_o["cg"]}
            data.update({key + "_i": val for key, val in c_i.items()})
            data.update({key + "_o": val for key, val in c_o.items()})
            yield data
