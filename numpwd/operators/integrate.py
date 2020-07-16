"""Compute operators from spin decomposed elements."""
from typing import List, Dict, Union, Optional, Tuple
from functools import lru_cache
from itertools import product
from logging import getLogger

from numpy import ndarray, abs, array
from pandas import DataFrame
from sympy import S, Function

from numpwd.integrate.analytic import integrate, cached_integrate

from numpwd.integrate.angular import ReducedAngularPolynomial, get_x_mesh, get_phi_mesh
from numpwd.integrate.numeric import ExpressionMap
from numpwd.qchannels.cg import get_cg, get_j_range
from numpwd.qchannels.lsj import get_m_range

from numpwd.operators.base import CHANNEL_COLUMNS

CG = Function("CG")
FACT = CG("l_o", "ml_o", "s_o", "ms_o", "j_o", "mj_o")
FACT *= CG("l_i", "ml_i", "s_i", "ms_i", "j_i", "mj_i")
FACT *= CG("l_i", "ml_i", "la", "mla", "l_o", "ml_o")

LOGGER = getLogger("numpwd")


class Integrator:
    """Wrapper class for running angular integrations for given expression."""

    def __init__(
        self,
        red_ang_poly: ReducedAngularPolynomial,
        args: List[Tuple[str, ndarray]],
        m_lambda_max: Optional[int] = None,
        pwd_fact_lambda: Optional[S] = None,
        use_cache: bool = True,
        numeric_zero: float = 1.0e-14,
        real_only: bool = True,
    ):
        """Initialize angular ingrator with given mesh."""
        self.poly = red_ang_poly
        self._func = self._integrated_cached if use_cache else self._integrate
        self.pwd_fact_lambda = (
            pwd_fact_lambda if pwd_fact_lambda is not None else S("exp(-I*(mla)*Phi)")
        )
        self.m_lambda_max = (
            m_lambda_max if m_lambda_max is not None else self.poly.lmax * 2
        )
        angular_keys = ("x_o", "x_i", "phi")
        self._args = tuple([key for key, _ in args if key not in angular_keys])
        self._values = tuple([val for key, val in args if key not in angular_keys])
        self._args += angular_keys
        self._values += (self.poly.x, self.poly.x, self.poly.phi)
        self.numeric_zero = numeric_zero
        self.real_only = real_only

    def _integrate(self, expr):
        LOGGER.debug("Integrating: %s", expr)
        data = []
        for m_lambda in range(-self.m_lambda_max, self.m_lambda_max + 1):
            big_phi_integrated = integrate(
                expr * self.pwd_fact_lambda.subs({"mla": m_lambda}), ("Phi", 0, "2*pi")
            )
            LOGGER.debug("m_lambda=%d -> %s", m_lambda, big_phi_integrated)
            if big_phi_integrated:
                fcn = ExpressionMap(big_phi_integrated, self._args)
                tensor = fcn(*self._values)
                res = self.poly.integrate(tensor, m_lambda)
            else:
                res = dict()
            for (l_o, l_i, la, mla), matrix in res.items():

                if (abs(matrix) < self.numeric_zero).all():
                    continue
                if self.real_only:
                    if (abs(matrix.imag) > self.numeric_zero).any():
                        raise AssertionError(
                            "Specified to return real data but imag of components for"
                            f" l_o={l_o}, l_i={l_i}, lambda={la}, m_lambda={mla}"
                            " not numericaly zero."
                        )
                    matrix = matrix.real

                data.append(
                    {
                        "l_o": l_o,
                        "l_i": l_i,
                        "lambda": la,
                        "m_lambda": mla,
                        "matrix": matrix,
                    }
                )
        return data

    @lru_cache(maxsize=128)
    def _integrated_cached(self, expr):
        return self._integrate(expr)

    def __call__(self, expr):
        """Integrates out all angular dependence of expression."""
        return self._func(expr)


def integrate_spin_decomposed_operator(
    spin_momentum_expressions: List[Dict[str, Union[int, S]]],
    args: List[Tuple[str, ndarray]],
    nx: int,
    nphi: int,
    lmax: int,
    m_lambda_max: Optional[int] = None,
    cache_integrals: bool = True,
    numeric_zero: float = 1.0e-14,
    real_only: bool = True,
) -> Tuple[DataFrame, ndarray]:
    r"""Runs angular integrals and contracts ls to j for spin decomposed two-nucleon operator.

    Computes
    $$
    O_{(l_o s_o)j_o m_{j_o} (l_i s_i)j_i m_{j_i}}(p_o, p_i, \\vec{q})
    =
    \\sum_{\\lambda m_\\lambda}
    \\sum\\limits_{m_{s_o} m_{s_i}}
    \\sum\\limits_{m_{l_o} m_{l_i}}
    \\left\\langle
        l_o m_{l_o}, s_o m_{s_o} \\big\\vert j_o m_{j_o}
    \\right\\rangle
    \\left\\langle
        l_i m_{l_i}, s_i m_{s_i} \\big\\vert j_i m_{j_i}
    \\right\\rangle
        \\int d x_o d x_i d \\phi_o d \\phi_i
        Y_{l_o m_{l_o}}^*(x_o, \\phi_o)
        Y_{l_i m_{l_i}}(x_i, \\phi_i)
        \\\\ \\times
        \\left\\langle
            \\vec p_o; s_o m_{s_o}
            \\big\\vert
            \\hat O(\\vec p_o, \\vec p_i, \\vec q)
            \\big\\vert
            \\vec p_i; s_i m_{s_i}
        \\right\\rangle
    $$
    For spin decomposed operator (``spin_momentum_expressions``)
    $$
    \\left\\langle
        \\vec p_o; s_o m_{s_o}
        \\big\\vert
        \\hat O(\\vec p_o, \\vec p_i, \\vec q)
        \\big\\vert
        \\vec p_i; s_i m_{s_i}
    \\right\\rangle
    $$
    This also allows treatment of external quantum numbers.

    Arguments:
        spin_momentum_expressions: List of spin matrix elements (dicts) which must have
            the keys ["s_o", "ms_o", "s_i", "ms_i", "expr"] where the spin values range
            from 0..1 for s and -1..1 for ms values. The expr key is a sympy expression
            containing only symbols for "p_o", "p_i", "q", "x_i", "x_o", "phi" and "Phi".

        args: Orderd list of operator arguments (not used in angular integration).
            For example, if the operator should be evaluated at
            ```
            args = [
                ("p_o", [1, 2, 3, 4])
                ("p_i", [6, 7])
            ]
            ```
            the final operator matrix will be of shape `(n_channel, 4, 2)` where where
            ```
            op[n_channel, 0, 1] = op(channel, 1, 7)
            ```

        nx: Number of polar mesh points.

        nphi: Number of azimuthal mesh points.

        lmax: Highest l value to run spin decomposition for.

        m_lambda_max: Maximal absolute value of m_lambda (restricts the number of
            m_lambda values).
            This can be useful if the operator denominator has finite powers of momenta
            as ``int( denominator(Phi) * exp(- I * Phi * m_lambda), (Phi, 0, 2*pi))``
            can be zero for a select range of ``m_lambda`` values.
            Defaults to 2 * lmax.

        cache_integrals: Save intermediate integrals in memory. Speeds up computation
            if sufficient memory is present.

        numeric_zero: If all matrix elements of a given channel are smaller
            (absolute value) than this number, the channel will be dropped.

        real_only: Return real components of matrix only. Raises an error if imaginary
            parts are larger than ``numeric_zero``.

        Returns: Channels and matrix of the operator. Channels are a data frame
            specifying which (ls)j quantum numbers correspond to which matrix entry
            (index of df == first index of matrix).
    """
    # check arguments
    #   Check spin
    spe = spin_momentum_expressions.copy()
    required_cols = set(["s_o", "ms_o", "s_i", "ms_i", "expr"])
    cols = None
    for el in spe:
        cols = list(el.keys()) if not cols else cols
        missig_cols = required_cols - set(cols)
        if missig_cols:
            raise KeyError(
                f"Spin element {el} does not match expected keys."
                f" Required but not present: {missig_cols}."
            )
        if set(cols) - set(el.keys()):
            raise KeyError(f"Spin element {el} does not provide all columns.")

    channel_columns = CHANNEL_COLUMNS + [
        col for col in cols if col not in required_cols
    ]

    # Allocate integration class
    angular_info = {}
    angular_info["x"], angular_info["wx"] = get_x_mesh(nx)
    angular_info["phi"], angular_info["wphi"] = get_phi_mesh(nphi)
    red_ang_poly = ReducedAngularPolynomial(**angular_info, lmax=lmax)
    m_lambda_max = m_lambda_max if m_lambda_max is not None else 2 * lmax
    integrator = Integrator(
        red_ang_poly=red_ang_poly,
        args=args,
        m_lambda_max=m_lambda_max,
        use_cache=cache_integrals,
        real_only=real_only,
        numeric_zero=numeric_zero,
    )

    tmp = dict()
    for spin_channel in spe:
        spin_channel = spin_channel.copy()
        for angular_channel in integrator(spin_channel.pop("expr")):
            ranges = {
                "j_o": get_j_range(spin_channel["s_o"], angular_channel["l_o"]),
                "j_i": get_j_range(spin_channel["s_i"], angular_channel["l_i"]),
                "ml_o": get_m_range(angular_channel["l_o"]),
            }
            # sum over all m_s, m_l and collect by j m_j
            for vals in product(*ranges.values()):
                pars = dict(zip(ranges.keys(), vals))
                pars.update(spin_channel)
                pars.update(angular_channel)
                pars["mla"] = pars.pop("m_lambda")
                pars["la"] = pars.pop("lambda")
                pars["ml_i"] = pars["ml_o"] - pars["mla"]
                pars["mj_i"] = pars["ml_i"] + pars["ms_i"]
                pars["mj_o"] = pars["ml_o"] + pars["ms_o"]
                if abs(pars["ml_i"]) > pars["l_i"]:
                    continue
                if abs(pars["mj_i"]) > pars["j_i"]:
                    continue
                if abs(pars["mj_o"]) > pars["j_o"]:
                    continue

                fact = float(FACT.subs(pars).replace(CG, get_cg))
                if abs(fact) < numeric_zero:
                    continue

                channel = tuple(pars.get(col) for col in channel_columns)
                tmp[channel] = tmp.get(channel, 0) + fact * angular_channel["matrix"]

    if cache_integrals:
        LOGGER.debug("Integrator cache: %s", integrator._integrated_cached.cache_info())
    LOGGER.debug("Analytic integrator cache: %s", cached_integrate.cache_info())

    # Sort and remove channels which cancel
    matrix = []
    channels = []
    for channel in sorted(tmp):
        mat = tmp[channel]
        if (abs(mat) > numeric_zero).any():
            channels.append(channel)
            matrix.append(mat)

    return (
        DataFrame(data=channels, columns=channel_columns),
        array(matrix),
    )
