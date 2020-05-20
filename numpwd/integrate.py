"""Routines which simplify analytical integrations
"""
from typing import Tuple, Optional, Dict

from functools import lru_cache

from sympy import Symbol
from sympy import integrate as _integrate


def get_spherical_substitutions(
    vec: str, label: Optional[str] = None
) -> Dict[str, str]:
    """Retuns substitution from cartesian to spherical coordinates

    Arguemnts:
        vec: Name of the vector (e.g., `p`)
        label: Lable of the vector (e.g., `1`)
    """
    return {
        f"{vec}{label}1": f"{vec}{label} * x{label} * cos(phi{label})",
        f"{vec}{label}2": f"{vec}{label} * x{label} * sin(phi{label})",
        f"{vec}{label}3": f"{vec}{label} * sqrt(1 - x{label}**2)",
    }


def get_angular_substitutions(var1: str = "phi1", var2: str = "phi2") -> Dict[str, str]:
    """Returns cms substitutions solved for var1 and var2

    Phi = (var1 + var2)/2
    phi = var1 - var2
    """
    return {
        var1: "Phi + phi/2",
        var2: "Phi - phi/2",
    }


@lru_cache(maxsize=128)
def cached_integrate(*args, **kwargs) -> Symbol:
    """Wraps sympys integrated but caches calls.
    """
    return _integrate(*args, **kwargs)


def integrate(expr: Symbol, boundaries: Tuple(Symbol, Symbol, Symbol)) -> Symbol:
    """Wrapper for sympies integrate which integrates each summand of a given term and
    caches intermediate results. This speeds up integrations of sums of similiar terms.

    Arguments:
        expr: The kernel used for the integration
        boundaries: Integral boundaries specified as arg, start, end
    """
    var = boundaries[0]

    if not isinstance(var, Symbol):
        var = Symbol(var)

    summands, basis = expr.expand().as_terms()

    out = 0
    for term, (_, powers, _) in summands:
        kernel = 1
        for ee, pp in zip(basis, powers):
            if var in ee.free_symbols:
                kernel *= ee ** pp

        integrated = cached_integrate(kernel, boundaries)

        out += term / kernel * integrated

    return out
