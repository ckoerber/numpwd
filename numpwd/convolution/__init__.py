"""Submodule provides routines for folding angular and spin PWDs."""
import numpy as np
import pandas as pd

from numpwd.convolution.channels import get_channel_overlap_indices
from numpwd.densities import Density
from numpwd.operators import Operator


def convolute(dens: Density, op: Operator) -> np.ndarray:
    """Convolutes density and operator."""
    assert len(dens.matrix.shape) == 3
    assert len(op.matrix.shape) == 3
    assert isinstance(op.isospin, pd.DataFrame)
    assert "iso" in op.isospin.columns
    np.testing.assert_allclose(dens.p, op.args[0][1])
    np.testing.assert_allclose(dens.p, op.args[1][1])

    # Find allowed transitions
    idx1, idx2 = get_channel_overlap_indices(dens.channels, op.channels)

    # Compute the isospin matrix element
    iso_fact = (
        pd.merge(dens.channels.loc[idx1], op.isospin, how="left")["iso"]
        .fillna(0)
        .values
    )
    weight = (dens.p ** 2 * dens.wp).reshape(-1, 1)
    weight = weight.T * weight
    return (
        (dens.matrix[idx1] * op.matrix[idx2] * weight).sum(axis=(1, 2)) * iso_fact
    ).sum()
