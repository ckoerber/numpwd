"""Tests assoctiated with spin-dependent dark matter exchange current (operator 32)."""
from os import path, pardir

import pytest

from numpy import array
from pandas import read_csv
from pandas.testing import assert_frame_equal
from sympy import S

from numpwd.integrate.analytic import SPHERICAL_BASE_SUBS, ANGLE_BASE_SUBS
from numpwd.operators.base import CHANNEL_COLUMNS
from numpwd.operators.integrate import integrate_spin_decomposed_operator


DATA_DIR = path.join(path.dirname(__file__), pardir, "data")
SPIN_DECOMPOSITION_FILE = path.join(DATA_DIR, "op32_spin_decomposition.csv")
FULL_DECOMPOSITION_FILE = path.join(DATA_DIR, "op32_full_decomposition.csv")


@pytest.fixture(name="spin_decomposition")
def fixture_spin_decomposition():
    """Reads spin decomposition from file."""
    momentum_subs = {f"k{n}": f"q_{n}/2 + p_i{n} - p_o{n}" for n in [1, 2, 3]}
    qz_subs = {"q_1": 0, "q_2": 0}

    def subs_all(expr):
        return (
            expr.subs(momentum_subs)
            .subs(SPHERICAL_BASE_SUBS)
            .subs(ANGLE_BASE_SUBS)
            .subs(qz_subs)
            .rewrite("exp")
            .expand()
        )

    df = read_csv(SPIN_DECOMPOSITION_FILE)
    for col in ["ms_dm_o", "ms_dm_i", "val"]:
        df[col] = df[col].apply(S)

    df = df.rename(columns={**{col: col.replace("dm", "ex") for col in df.columns}})
    df["expr"] = df.pop("val").apply(subs_all)

    return df.to_dict("records")


@pytest.fixture(name="legacy_decomposition")
def fixture_legacy_decomposition():
    """Reads full legacy decomposition from file."""
    df = read_csv(FULL_DECOMPOSITION_FILE)
    for col in ["ms_ex_o", "ms_ex_i"]:
        df[col] = df[col].apply(S)

    return df[CHANNEL_COLUMNS + ["val"]].set_index(CHANNEL_COLUMNS).sort_index()


def test_integrate_spin_decomposed_operator(spin_decomposition, legacy_decomposition):
    """Runs angular integrations of spin decomposed op against legacy data.
    """
    args = [("p_o", array([200])), ("p_i", array([100])), ("q_3", array([300]))]
    channels, matrix = integrate_spin_decomposed_operator(
        spin_decomposition, args=args, nx=3, nphi=7, lmax=2, numeric_zero=1.0e-8,
    )

    channels["val"] = matrix.flatten()
    res = channels.set_index(CHANNEL_COLUMNS).sort_index()
    assert_frame_equal(res, legacy_decomposition)
