"""Tests for the spin module
"""
from numpwd.qchannels.spin import get_spin_matrix_element


def test_s_dot_s_spin_pwd():
    """Checks if sigma_1 . sigma_2 PWD returns expected result
    """
    expected = [
        {"s_o": 0, "ms_o": 0, "s_i": 0, "ms_i": 0, "val": -3},
        {"s_o": 1, "ms_o": -1, "s_i": 1, "ms_i": -1, "val": 1},
        {"s_o": 1, "ms_o": 0, "s_i": 1, "ms_i": 0, "val": 1},
        {"s_o": 1, "ms_o": 1, "s_i": 1, "ms_i": 1, "val": 1},
    ]
    expr = "s11 * s21 + s12 * s22 + s13 * s23"
    res = get_spin_matrix_element(expr, pauli_symbol="s")

    assert res == expected
