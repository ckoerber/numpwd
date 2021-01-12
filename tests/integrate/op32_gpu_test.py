"""Tests GPU vs CPU integration."""
from pandas.testing import assert_frame_equal

from pytest import mark

from numpwd.operators.base import CHANNEL_COLUMNS
from numpwd.operators.integrate import integrate_spin_decomposed_operator

from tests.integrate.op32_test import (  # noqa
    fixture_spin_decomposition,
    fixture_legacy_decomposition,
)

try:
    import cupy as cp
except ImportError:
    cp = None


@mark.skipif(cp is None, reason="Was not able to import cupy.")
def test_integrate_spin_decomposed_operator_gpu(
    spin_decomposition, legacy_decomposition
):
    """Run angular integrations of spin decomposed op on gpu against legacy data."""
    args = [
        ("p_o", cp.array([200])),
        ("p_i", cp.array([100])),
        ("q_3", cp.array([300])),
    ]
    channels, matrix = integrate_spin_decomposed_operator(
        spin_decomposition,
        args=args,
        nx=3,
        nphi=7,
        lmax=2,
        numeric_zero=1.0e-8,
        gpu=True,
    )

    assert isinstance(matrix, cp.ndarray)

    channels["matrix"] = matrix.get().flatten()
    index_cols = CHANNEL_COLUMNS + ["ms_ex_o", "ms_ex_i"]
    res = channels.set_index(index_cols).sort_index()
    assert_frame_equal(res, legacy_decomposition)
