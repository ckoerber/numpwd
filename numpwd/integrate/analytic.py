"""Routines which simplify analytical integrations
"""
from typing import Tuple, Optional, Dict

from functools import lru_cache

from sympy import Symbol
from sympy import integrate as _integrate


def get_spherical_substitutions(
    vec: str = "p", label: Optional[str] = None
) -> Dict[str, str]:
    """Retuns substitution from cartesian to spherical coordinates

    Arguemnts:
        vec: Name of the vector (e.g., `p`)
        label: Lable of the vector (e.g., `i`)
    """
    llabel = f"_{label}" if label else ""
    return {
        f"{vec}_{label}1": f"{vec}{llabel} * x{llabel} * cos(phi{llabel})",
        f"{vec}_{label}2": f"{vec}{llabel} * x{llabel} * sin(phi{llabel})",
        f"{vec}_{label}3": f"{vec}{llabel} * sqrt(1 - x{llabel}**2)",
    }


def get_angular_substitutions(
    var1: str = "phi_i", var2: str = "phi_o"
) -> Dict[str, str]:
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


def integrate(
    expr: Symbol, boundaries: Tuple[Symbol, Symbol, Symbol] = ("Phi", 0, "2*pi")
) -> Symbol:
    """Wrapper for sympies integrate which integrates each summand of a given term and
    caches intermediate results. This speeds up integrations of sums of similiar terms.

    Arguments:
        expr: The kernel used for the integration
        boundaries: Integral boundaries specified as arg, start, end
    """
    var = boundaries[0]

    if not isinstance(var, Symbol):
        var = Symbol(var)

    summands, basis = expr.simplify().expand().as_terms()

    out = 0
    for term, (_, powers, _) in summands:
        kernel = 1
        for ee, pp in zip(basis, powers):
            if var in ee.free_symbols:
                kernel *= ee ** pp

        integrated = cached_integrate(kernel, boundaries)

        out += term / kernel * integrated

    return out


SPHERICAL_BASE_SUBS = {
    **get_spherical_substitutions("p", "i"),
    **get_spherical_substitutions("p", "o"),
}
ANGLE_BASE_SUBS = get_angular_substitutions()


def integrate_out_angle(expr: Symbol) -> Symbol:
    """Replaces cartesian expressions with spherical expressions, substitutes to
    angular CMS coordiantes and integrates out `Phi`
    """
    return integrate(expr.subs(SPHERICAL_BASE_SUBS).subs(ANGLE_BASE_SUBS))
