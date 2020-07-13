"""Compute operators from expression."""
from typing import List, Dict, Union, Optional
from functools import lru_cache

from pandas import Series
from sympy import S, Function

from numpwd.operators.base import Operator
from numpwd.integrate.analytic import integrate

CG = Function("CG")
FACT = CG("l_o", "ml_o", "s_o", "ms_o", "j_o", "mj_o")
FACT *= CG("l_i", "ml_i", "s_i", "ms_i", "j_i", "mj_i")
FACT *= CG("l_i", "ml_i", "lambda", "m_lambda", "l_o", "ml_o")


def _group_agg_rank_project_lambda(
    df, m_lambda_max=2, pwd_fact_lambda=S("exp(-I*(m_lambda)*Phi)")
):
    data = dict()
    # sum over ms nuc
    for row in df.to_dict("records"):
        # Save results for unique ms DM, s nuc m_lambda
        for m_lambda in range(-m_lambda_max, m_lambda_max + 1):
            out = data.get(m_lambda, S(0))
            data[m_lambda] = out + row["val"] * pwd_fact_lambda.subs(
                {**row, "m_lambda": m_lambda}
            )

    # Run angular integrations
    for key, val in data.items():
        data[key] = integrate(val, ("Phi", 0, "2*pi"))

    out = Series(data, name="val")
    out.index.name = m_lambda
    return out


class Integrator:
    def __init__(
        self,
        red_ang_poly,
        m_lambda_max: Optional[int] = None,
        pwd_fact_lambda: Optional[S] = None,
        use_cache: bool = True,
    ):
        self.poly = red_ang_poly
        self._func = self._integrated_cached if use_cache else self._integrate
        self.pwd_fact_lambda = (
            pwd_fact_lambda
            if pwd_fact_lambda is not None
            else S("exp(-I*(m_lambda)*Phi)")
        )
        self.m_lambda_max = (
            m_lambda_max if m_lambda_max is not None else self.poly.lmax * 2
        )

    def _integrate(self, expr):
        data = []
        for m_lambda in range(-self.m_lambda_max, self.m_lambda_max + 1):
            big_phi_integrated = integrate(
                expr * self.pwd_fact_lambda, ("Phi", 0, "2*pi")
            )
            tensor = ExpressionMap(big_phi_integrated, ("x_o", "x_i", "phi"))
            res = self.poly.integrate(m_lambda, tensor)
            for (l_o, l_i, la, mla), val in res.items():

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

    def __call(self, expr):
        return self._func(expr)


def get_pwd_operator(
    spin_momentum_expressions: List[Dict[str, Union[int, S]]],
    isospin_expression: List[Dict[str, Union[int, S]]],
    lmax: int,
    m_lambda_max: Optional[int] = None,
    cache_integral: bool = True,
    **mesh_info,
) -> Operator:
    """Run partial wave decomposition of two-nucleon operator from expression.

    Arguments:
        spin_momentum_expressions: List of spin matrix elements (dicts) which must have
            the keys ["s_o", "ms_o", "s_i", "ms_i", "expr"] where the spin values range
            from 0..1 for s and -1..1 for ms values. The expr key is a sympy expression
            containing ponly symbols for "p_o", "p_i", "q", "x_i", "x_o", "phi" and "Phi".

        isospin_expression: List of isospin matrix elements (dicts) which must have
            the keys ["t_o", "mt_o", "t_i", "mt_i", "expr"].

        lmax: Highest l value to run spin decomposition for.

        mesh_info: Dict containing mesh information about "p", "wp", "x", "wx",
            "phi", "wphi".
    """
    # check arguments
    for el in spin_momentum_expressions:
        if not set(["s_o", "ms_o", "s_i", "ms_i", "expr"]) - set(el.keys()):
            raise KeyError(
                f"Spin element {el} does not provide all required spin keys."
            )
    for el in isospin_expression:
        if not set(["t_o", "mt_o", "t_i", "mt_i", "expr"]) - set(el.keys()):
            raise KeyError(
                f"Spin element {el} does not provide all required spin keys."
            )

    m_lambda_max = m_lambda_max if m_lambda_max is not None else 2 * lmax
    integrator = Integrator(m_lambda_max=m_lambda_max, use_cache=cache_integral)

    data = dict()
    for spin_channel in sorted_channels:
        for angular_channel in integrator(spin_channel.pop("expr")):
            ranges = {
                "j_o": get_j_range(spin_channel["s_o"], spin_channel["l_o"]),
                "j_i": get_j_range(spin_channel["s_i"], spin_channel["l_i"]),
                "ml_o": get_m_range(spin_channel["l_o"]),
            }
            # sum over all m_s, m_l and collect by j m_j
            for vals in product(*ranges.values()):
                pars = dict(zip(ranges.keys(), vals))
                pars.update(spin_channel)
                pars.update(angular_channel)
                pars["m_lambda"] = pars.pop("m_lambda")
                pars["lambda"] = pars.pop("lambda")
                pars["ml_i"] = pars["ml_o"] - pars["m_la"]
                pars["mj_i"] = pars["ml_i"] + pars["ms_i"]
                pars["mj_o"] = pars["ml_o"] + pars["ms_o"]
                if abs(pars["ml_i"]) > pars["l_i"]:
                    continue
                if abs(pars["mj_i"]) > pars["j_i"]:
                    continue
                if abs(pars["mj_o"]) > pars["j_o"]:
                    continue

                key = (pars["j_o"], pars["j_i"], pars["mj_o"], pars["mj_i"])
                tmp = data.get(key, S(0))
                data[key] = (
                    tmp
                    + float(FACT.subs(pars).replace(CG, get_cg))
                    * angular_channel["matrix"]
                )

    channels = data.keys()
    matrix = np.array(data.values())
