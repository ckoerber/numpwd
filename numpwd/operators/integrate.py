"""Compute operators from expression."""
from typing import List, Dict, Union, Optional, Tuple
from functools import lru_cache
from itertools import product

from numpy import ndarray, abs, array
from pandas import DataFrame
from sympy import S, Function

from numpwd.operators.base import Operator, CHANNEL_COLUMNS
from numpwd.integrate.analytic import integrate
from numpwd.integrate.angular import ReducedAngularPolynomial, get_x_mesh, get_phi_mesh
from numpwd.integrate.numeric import ExpressionMap
from numpwd.qchannels.cg import get_cg, get_j_range
from numpwd.qchannels.lsj import get_m_range


CG = Function("CG")
FACT = CG("l_o", "ml_o", "s_o", "ms_o", "j_o", "mj_o")
FACT *= CG("l_i", "ml_i", "s_i", "ms_i", "j_i", "mj_i")
FACT *= CG("l_i", "ml_i", "la", "mla", "l_o", "ml_o")


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
        data = []
        for m_lambda in range(-self.m_lambda_max, self.m_lambda_max + 1):
            big_phi_integrated = integrate(
                expr * self.pwd_fact_lambda.subs({"mla": m_lambda}), ("Phi", 0, "2*pi")
            )
            if big_phi_integrated:
                fcn = ExpressionMap(big_phi_integrated, self._args)
                tensor = fcn(*self._values)
                res = self.poly.integrate(tensor, m_lambda)
            else:
                res = dict()
            for (l_o, l_i, la, mla), matrix in res.items():

                if all(abs(matrix) < self.numeric_zero):
                    continue
                if self.real_only:
                    if any(abs(matrix.imag) > self.numeric_zero):
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
    isospin_expressions: List[Dict[str, Union[int, S]]],
    args: List[Tuple[str, ndarray]],
    nx: int,
    nphi: int,
    lmax: int,
    m_lambda_max: Optional[int] = None,
    cache_integral: bool = True,
    numeric_zero: float = 1.0e-14,
    real_only: bool = True,
) -> Operator:
    """Runs all angular integrals against spin decomposed two-nucleon operator.

    Arguments:
        spin_momentum_expressions: List of spin matrix elements (dicts) which must have
            the keys ["s_o", "ms_o", "s_i", "ms_i", "expr"] where the spin values range
            from 0..1 for s and -1..1 for ms values. The expr key is a sympy expression
            containing only symbols for "p_o", "p_i", "q", "x_i", "x_o", "phi" and "Phi".

        isospin_expression: List of isospin matrix elements (dicts) which must have
            the keys ["t_o", "mt_o", "t_i", "mt_i", "expr"].

        lmax: Highest l value to run spin decomposition for.

        mesh_info: Dict containing mesh information about "p", "wp", "x", "wx",
            "phi", "wphi".
    """
    # check arguments
    #   Check spin
    spe = spin_momentum_expressions.copy()
    required_spin_keys = set(
        ["s_o", "ms_o", "s_i", "ms_i", "ms_ex_o", "ms_ex_i", "expr"]
    )
    for el in spe:
        if required_spin_keys != set(el.keys()):
            raise KeyError(
                f"Spin element {el} does not provide all required spin keys"
                f" ({required_spin_keys})."
            )

    #   Check isospin
    ie = isospin_expressions.copy()
    iso_mat = {}
    required_isospin_keys = set(["t_o", "mt_o", "t_i", "mt_i", "expr"])
    for el in ie:
        if required_isospin_keys != set(el.keys()):
            raise KeyError(
                f"Spin element {el} does not provide all required spin keys."
            )
        iso_mat[(["t_o"], ["mt_o"], ["t_i"], ["mt_i"])] = float(el["expr"])

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
        use_cache=cache_integral,
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

                channel = tuple(pars.get(key) for key in CHANNEL_COLUMNS)
                tmp[channel] = tmp.get(channel, 0) + fact * angular_channel["matrix"]

    # Sort and remove channels which cancel
    matrix = []
    channels = []
    for channel in sorted(tmp):
        mat = tmp[channel]
        if any(abs(mat) > numeric_zero):
            channels.append(channel)
            matrix.append(mat)

    operator = Operator()
    operator.channels = DataFrame(data=channels, columns=CHANNEL_COLUMNS)
    operator.matrix = array(matrix)
    operator.isospin = iso_mat
    operator.args = args
    operator.mesh_info = angular_info
    operator.misc = {
        "lmax": lmax,
        "m_lambda_max": m_lambda_max,
        "spin_momentum_expressions": spe,
        "isospin_expressions": ie,
        "cache_integral": cache_integral,
        "numeric_zero": numeric_zero,
        "real_only": real_only,
    }

    return operator
