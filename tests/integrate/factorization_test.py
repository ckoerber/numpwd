"""Tests factorization input expression vs not factorizing it."""
from pandas.testing import assert_frame_equal
from sympy import sympify

import pytest

from numpwd.operators.base import CHANNEL_COLUMNS
from numpwd.operators.integrate import integrate_spin_decomposed_operator

from tests.integrate.op32_test import fixture_spin_decomposition  # noqa


def test_integrate_spin_decomposed_operator_factorization(spin_decomposition):
    """Run angular integrations of spin decomposed op factorized vs not factorized."""
    args = [
        ("p_o", [200]),
        ("p_i", [100]),
        ("q_3", [300]),
    ]

    # modify expresion by new factor
    factor = 1 / sympify("1 + p_o**2 + p_i**2")
    sd = []
    for dd in spin_decomposition:
        pars = dd.copy()
        pars["expr"] *= factor
        sd.append(pars)

    channels, matrix = integrate_spin_decomposed_operator(
        sd, args=args, nx=3, nphi=7, lmax=2, numeric_zero=1.0e-8,
    )
    channels["matrix"] = matrix.flatten()

    channels_factorized, matrix_factorized = integrate_spin_decomposed_operator(
        spin_decomposition,
        args=args,
        nx=3,
        nphi=7,
        lmax=2,
        numeric_zero=1.0e-8,
        spin_momentum_factor=factor,
    )
    channels_factorized["matrix"] = matrix_factorized.flatten()

    index_cols = CHANNEL_COLUMNS + ["ms_ex_o", "ms_ex_i"]

    res = channels.set_index(index_cols).sort_index()
    res_factorized = channels_factorized.set_index(index_cols).sort_index()

    assert_frame_equal(res, res_factorized)


def test_integrate_spin_decomposed_operator_factorization_assertion_error(
    spin_decomposition,
):
    """Run angular integrations of spin decomposed op factorized with wrong input."""
    args = [
        ("p_o", [200]),
        ("p_i", [100]),
        ("q_3", [300]),
    ]

    # modify expresion by new factor
    factor = 1 / sympify("Phi")

    with pytest.raises(AssertionError):
        integrate_spin_decomposed_operator(
            spin_decomposition,
            args=args,
            nx=3,
            nphi=7,
            lmax=2,
            numeric_zero=1.0e-8,
            spin_momentum_factor=factor,
        )
