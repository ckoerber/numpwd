"""Tests for exporting and importing opertor to and from h5."""
from logging import getLogger, DEBUG, basicConfig

from numpy import arange, random
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from sympy import sympify

from pytest import fixture

from numpwd.integrate.analytic import SPHERICAL_BASE_SUBS, ANGLE_BASE_SUBS
from numpwd.operators.base import Operator, CHANNEL_COLUMNS
from numpwd.operators.h5 import read, write

basicConfig(level=DEBUG)
LOGGER = getLogger("numpwd")


@fixture(name="random_op")
def fixture_random_op() -> Operator:
    """Returns random operator"""
    random.seed(42)

    spin_expression = sympify("(sigma_ex1 * k1 + sigma_ex2 * k2 + sigma_ex3 * k3)")
    substitutions = [SPHERICAL_BASE_SUBS, ANGLE_BASE_SUBS]
    integration_kwargs = {
        "nx": 3,
        "nphi": 7,
        "lmax": 2,
        "numeric_zero": 1.0e-8,
        "m_lambda_max": None,
    }

    op = Operator()
    op.args = [("p_o", arange(10)), ("p_i", arange(5)), ("q_3", arange(2))]
    op.matrix = random.randint(0, 10, size=(20, 10, 5, 2))
    op.channels = DataFrame(
        data=random.randint(0, 10, size=(20, len(CHANNEL_COLUMNS))),
        columns=CHANNEL_COLUMNS,
    )
    op.misc = {
        "spin_expression": spin_expression,
        "substitutions": substitutions,
        "integration_kwargs": integration_kwargs,
    }
    op.isospin = {(0, 0, 0, 0): 1.0}
    op.mesh_info = {"n_p_o": 10, "n_p_i": 5, "q_3": 2}
    op.check()

    return op


def test_decompose_operator(random_op):
    """Exports and imports operator from and to h5. Asserts they are equal."""
    pass
