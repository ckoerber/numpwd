"""
"""
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from logging import getLogger

from numpy import ndarray
from numpy import array as np_array
from pandas import DataFrame
from sympy import S, sympify

from numpwd import __version__
from numpwd.qchannels.spin import get_spin_matrix_element_ex, get_spin_matrix_element
from numpwd.operators.base import Operator
from numpwd.operators.integrate import integrate_spin_decomposed_operator


LOGGER = getLogger("numpwd")


try:
    import cupy as cp
except ImportError:
    LOGGER.debug("Cupy not available")
    cp = None


def subs_all(expr, substitutions):
    for subs in substitutions:
        expr = expr.subs(subs)
    return expr.rewrite("exp").expand()


def decompose_operator(
    spin_momentum_expression: S,
    isospin_expression: S,
    args: List[Tuple[str, ndarray]],
    substitutions: List[Dict[str, str]] = None,
    spin_decomposition_kwargs: Dict[str, Any] = None,
    integration_kwargs: Dict[str, Any] = None,
    misc: Optional[Dict[str, Any]] = None,
    integrate_sd_kwargs: Optional[Dict[str, Any]] = None,
) -> Operator:
    """Runs entire spin partial wave decomposition for operator.

    Arguments:
        isospin_expression: List of isospin matrix elements (dicts) which must have
            the keys ["t_o", "mt_o", "t_i", "mt_i", "expr"].
    """
    substitutions = substitutions or []
    spin_decomposition_kwargs = spin_decomposition_kwargs or {}
    integration_kwargs = integration_kwargs or {}
    integrate_sd_kwargs = integrate_sd_kwargs or {}

    operator = Operator()
    operator.misc = dict(misc) if misc else {}

    spin_momentum_expression = sympify(spin_momentum_expression)
    operator.misc["spin_momentum_expression"] = spin_momentum_expression
    operator.misc["isospin_expression"] = isospin_expression
    operator.misc["subsititutions"] = substitutions

    operator.misc["pauli_symbol_spin"] = spin_decomposition_kwargs.get(
        "pauli_symbol_spin", "sigma"
    )
    operator.misc["pauli_label_ex"] = spin_decomposition_kwargs.get(
        "pauli_label_ex", "_ex"
    )
    operator.misc["pauli_symbol_isospin"] = spin_decomposition_kwargs.get(
        "pauli_symbol_isospin", "tau"
    )

    operator.misc["numeric_zero"] = integration_kwargs.get("numeric_zero", 1.0e-8)
    operator.misc["lmax"] = integration_kwargs.get("lmax", 6)
    operator.misc["m_lambda_max"] = integration_kwargs.get("m_lambda_max")

    operator.mesh_info["azimuthal_type"] = "linear"
    operator.mesh_info["nphi"] = integration_kwargs.get(
        "nphi", 2 * operator.misc["lmax"] + 2
    )
    operator.mesh_info["polar_type"] = "leggaus"
    operator.mesh_info["nx"] = integration_kwargs.get("nx", operator.misc["lmax"] + 1)

    gpu = integration_kwargs.get("gpu", False)
    array = cp.array if gpu and cp is not None else np_array
    args = [(key, array(val)) for key, val in args]

    operator.args = args

    operator.misc["numpwd_version"] = __version__
    operator.misc["computation_start"] = datetime.now()

    LOGGER.info("Running isospin decomposition of operator %s", isospin_expression)
    isospin = get_spin_matrix_element(
        isospin_expression, pauli_symbol=operator.misc["pauli_symbol_isospin"],
    )
    df = DataFrame(isospin)
    df = df.rename(columns={col: col.replace("s", "t") for col in df.columns})
    operator.isospin = df.set_index([col for col in df.columns if col != "expr"]).expr

    LOGGER.info(
        "Running angular decomposition of operator %s", spin_momentum_expression
    )
    ext_spin_matrix = str(operator.misc["pauli_symbol_spin"]) + str(
        operator.misc["pauli_label_ex"]
    )
    if ext_spin_matrix in str(spin_momentum_expression):
        LOGGER.debug("Using decomposition with external pauli matrix")
        spins = get_spin_matrix_element_ex(
            spin_momentum_expression,
            pauli_symbol=operator.misc["pauli_symbol_spin"],
            ex_label=operator.misc["pauli_label_ex"],
        )
    else:
        LOGGER.debug("No external pauli matrix detected. Using regular decomposition.")
        spins = get_spin_matrix_element(
            spin_momentum_expression, pauli_symbol=operator.misc["pauli_symbol_spin"]
        )

    LOGGER.debug("Found %d non-zero channels (see below)", len(spins))
    LOGGER.debug(DataFrame(spins))

    LOGGER.debug("Applying substitutions")
    for spin_mat in spins:
        spin_mat["expr"] = subs_all(spin_mat["expr"], substitutions)

    LOGGER.debug("Running integrations")
    operator.channels, operator.matrix = integrate_spin_decomposed_operator(
        spins,
        args=args,
        nx=operator.mesh_info["nx"],
        nphi=operator.mesh_info["nphi"],
        lmax=operator.misc["lmax"],
        numeric_zero=operator.misc["numeric_zero"],
        m_lambda_max=operator.misc["m_lambda_max"],
        gpu=gpu,
        **integrate_sd_kwargs,
    )

    operator.misc["computation_end"] = datetime.now()

    return operator
