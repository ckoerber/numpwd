"""Submodule provides routines for folding angular and spin PWDs."""
import numpy as np
import pandas as pd

from numpwd.convolution.channels import get_channel_overlap_indices
from numpwd.densities import Density
from numpwd.operators import Operator

try:
    import cupy as cp
except ImportError:
    cp = None


def convolute(dens: Density, op: Operator, tol: float = 1.0e-7) -> np.ndarray:
    """Convolutes density and operator."""
    if len(dens.matrix.shape) != 3:
        raise AssertionError(
            f"Density matrix must be three dimensional"
            f" but was of dim: {dens.matrix.shape}"
        )
    if len(op.matrix.shape) == 3:
        matrix = op.matrix
    if len(op.matrix.shape) == 4:
        if op.args[2][0] not in ("k", "q", "qval", "q3"):
            raise AssertionError(
                'Third op argument must be one of `("k", "q", "qval", "q3")`'
                f" but was {op.args[2][0]}"
            )
        qval = dens.current_info["qval"]
        q_diff = np.abs(op.args[2][1] - qval)
        q_idx = np.argmin(q_diff)
        if q_diff[q_idx] > tol:
            raise ValueError(
                "Was not able to find matching current momenta for"
                f" density {dens} and operator {op} at tolerance {tol}."
                f" best match {q_diff[q_idx]} for {qval} and {op.args[2][1]}"
            )
        matrix = op.matrix[:, :, :, q_idx]

    assert isinstance(op.isospin, pd.DataFrame)
    # ensure backwards comp
    isospin = op.isospin.rename(columns={"expr": "iso"})
    assert "iso" in isospin.columns

    backend = cp if cp is not None and isinstance(dens.p, cp.ndarray) else np

    backend.testing.assert_allclose(dens.p, op.args[0][1])
    backend.testing.assert_allclose(dens.p, op.args[1][1])

    # Find allowed transitions
    idx1, idx2 = get_channel_overlap_indices(dens.channels, op.channels)

    # Compute the isospin matrix element
    iso_fact = backend.array(
        pd.merge(dens.channels.loc[idx1], isospin, how="left")["iso"].fillna(0).values
    )
    weight = (dens.p ** 2 * dens.wp).reshape(-1, 1)
    weight = weight.T * weight
    res = (
        (dens.matrix[idx1] * matrix[idx2] * weight).sum(axis=(1, 2)) * iso_fact
    ).sum()
    return res.dtype.type(res)  # explicitly cast dtype
