"""Utility function helpful for using sympy."""
from sympy import S, fraction, together, lambdify
from numpy import allclose, random
from numpy.testing import assert_allclose


def _stochastically_equal(
    expr1: S,
    expr2: S,
    rtol: float = 1.0e-16,
    atol: float = 0,
    low: float = -1.0,
    high: float = 1.0,
    scale: float = 1.0,
    size: int = 100,
    seed: int = None,
):
    """Stochastically evaluate both expressions to check if they are stochastically_equal.

    Uses a uniform distribution for all free symbols in both expressions ranging from
    ``scale * low`` to ``scale * high`` of size ``size`` and compares expressions using
    numpys ``allclose``.
    """
    expr = fraction(together(expr1 - expr2))[0]
    args = expr.free_symbols
    fcn = lambdify(args, expr)

    if seed:
        random.seed(seed)

    inp = random.uniform(low, high, size=(len(args), size))
    return fcn(*inp)
    return allclose(res, 0, rtol=rtol, atol=atol)


def stochastically_equal(
    expr1: S,
    expr2: S,
    rtol: float = 1.0e-16,
    atol: float = 0,
    low: float = -1.0,
    high: float = 1.0,
    scale: float = 1.0,
    size: int = 100,
    seed: int = None,
):
    """Stochastically evaluate both expressions to check if they are stochastically_equal.

    Uses a uniform distribution for all free symbols in both expressions ranging from
    ``scale * low`` to ``scale * high`` of size ``size`` and compares expressions using
    numpys ``allclose``.
    """
    res = _stochastically_equal(
        expr1,
        expr2,
        rtol=rtol,
        atol=atol,
        low=low,
        high=high,
        scale=scale,
        size=size,
        seed=seed,
    )
    return allclose(res, 0, rtol=rtol, atol=atol)


def assert_stochastically_equal(
    expr1: S,
    expr2: S,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    low: float = -1.0,
    high: float = 1.0,
    scale: float = 1.0,
    size: int = 100,
    seed: int = None,
):
    """Stochastically evaluate both expressions to check if they are stochastically_equal.

    Uses a uniform distribution for all free symbols in both expressions ranging from
    ``scale * low`` to ``scale * high`` of size ``size`` and compares expressions using
    numpys ``allclose``.
    """
    res = _stochastically_equal(
        expr1,
        expr2,
        rtol=rtol,
        atol=atol,
        low=low,
        high=high,
        scale=scale,
        size=size,
        seed=seed,
    )
    assert_allclose(res, 0, atol=atol, rtol=rtol)
