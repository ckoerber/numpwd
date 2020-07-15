"""Checks if projection op32 onto spin channels reproduces Sachin's partial results."""
from os import path, pardir
from sympy import S, simplify
from pandas import read_csv, DataFrame

from numpwd.qchannels.spin import get_spin_matrix_element_ex

LEGACY_FILE = path.join(
    path.dirname(__file__), pardir, "data", "op32_spin_decomposition.csv"
)


def test_op32_spin_decomposition():
    """Checks if sigma_1 . sigma_2 PWD returns expected result."""
    legacy_df = read_csv(LEGACY_FILE)
    for col in ["ms_dm_o", "ms_dm_i", "val"]:
        legacy_df[col] = legacy_df[col].apply(S)

    index_cols = [col for col in legacy_df.columns if col != "val"]
    legacy_df = legacy_df.set_index(index_cols).sort_index()

    expr = S("(sigma_dm1 * k1 + sigma_dm2 * k2 + sigma_dm3 * k3)")
    expr *= S("sigma10 * (sigma21 * k1 + sigma22 * k2 + sigma23 * k3)")

    data = get_spin_matrix_element_ex(expr, pauli_symbol="sigma", ex_label="_dm")
    df = DataFrame(data).set_index(index_cols).sort_index()

    diff = (legacy_df.val - df.val).apply(simplify)
    assert all(diff == 0)
