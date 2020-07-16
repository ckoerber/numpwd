"""Test runs the entire opertor integration pipeline for spin-dependent operator."""
from numpy import array
from pandas.testing import assert_frame_equal
from sympy import S

from numpwd.integrate.analytic import SPHERICAL_BASE_SUBS, ANGLE_BASE_SUBS
from numpwd.operators.base import CHANNEL_COLUMNS
from numpwd.operators.expression import decompose_operator

from tests.integrate.op32_test import fixture_legacy_decomposition  # noqa


def test_decompose_operator(legacy_decomposition):
    """Runs spin, isospin and angular decomposition for operator 32."""
    spin_expression = S("(sigma_ex1 * k1 + sigma_ex2 * k2 + sigma_ex3 * k3)")
    spin_expression *= S("sigma10 * (sigma21 * k1 + sigma22 * k2 + sigma23 * k3)")
    isospin_expression = S("tau10 * tau20")
    args = [("p_o", array([200])), ("p_i", array([100])), ("q_3", array([300]))]
    momentum_subs = {f"k{n}": f"q_{n}/2 + p_i{n} - p_o{n}" for n in [1, 2, 3]}
    qz_subs = {"q_1": 0, "q_2": 0}
    substitutions = [momentum_subs, SPHERICAL_BASE_SUBS, ANGLE_BASE_SUBS, qz_subs]
    spin_decomposition_kwargs = {}
    integration_kwargs = {
        "nx": 3,
        "nphi": 7,
        "lmax": 2,
        "numeric_zero": 1.0e-8,
        "m_lambda_max": None,
    }

    op = decompose_operator(
        spin_expression,
        isospin_expression,
        args=args,
        substitutions=substitutions,
        spin_decomposition_kwargs=spin_decomposition_kwargs,
        integration_kwargs=integration_kwargs,
    )

    channels = op.channels.copy()
    channels["matrix"] = op.matrix.flatten()
    index_cols = CHANNEL_COLUMNS + ["ms_ex_o", "ms_ex_i"]
    res = channels.set_index(index_cols).sort_index()

    assert_frame_equal(res, legacy_decomposition)
