"""Angular operators used for numerically integrating elements."""
from typing import Dict, Tuple, Optional

from itertools import product
from warnings import warn
from logging import getLogger

from numpwd.utils.backend import get_available_memory

from tqdm import tqdm

import numpy as np
from numpy.polynomial.legendre import leggauss

from scipy.special import sph_harm

from pandas import DataFrame

from numpwd.qchannels.cg import get_cg as cg

LOGGER = getLogger("numpwd")

try:
    import cupy as cp
except ImportError:
    LOGGER.debug("Cupy not available")
    cp = None


def get_x_mesh(nx: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""Returns Legendre Gauss mesh for \\(x \\in [0, 1]\\) angle.

    Returns:
        x: The angle variable
        wx: The integration weight
    """
    return leggauss(nx)


def get_phi_mesh(nphi: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""Returns linear mesh for \\(\\phi \\in [0, 2 \\pi]\\) angle.

    Returns:
        phi: The angle variable
        wphi: The integration weight
    """
    phi = np.arange(0, nphi) * 2 * np.pi / nphi
    wphi = np.ones(nphi) * 2 * np.pi / nphi
    return phi, wphi


class ReducedAngularPolynomial:
    r"""Stores the reduced angular polynomial.

    This polynomial is defined as

    $$
    \\begin{aligned}
    O_{(l_o l_i) \\lambda m_{\\lambda}}(x_o, x_i, \\phi)
    =
    \\frac{2 \\lambda + 1}{2 l_o + 1}
    \\sum_{m_o m_i} &
    \\left\\langle l_i m_i , \\lambda m_{\\lambda}
    \\middle\\vert l_o m_o
    \\right\\rangle
    \\\\\\\\ & \\times
    Y_{l_o m_o}^*(x_o, \\phi_o) Y_{l_i m_i}(x_i, \\phi_i)
    \\exp\\{ i \\phi \\frac{m_i + m_o}{2} \\}
    \\end{aligned}
    $$

    where

    $$
    \\Phi = \\frac{\\phi_i + \\phi_o}{2}
    \\, , \\qquad
    \\phi = \\phi_i - \\phi_o
    $$

    This makes use that the integration over angle $\\Phi$ can be factored out:
    $$
    \\exp\\{ -i \\Phi (m_o - m_i)\\} = \\exp\\{ -i \\Phi m_{la} \\}
    $$
    """

    def __init__(
        self,
        x: np.ndarray,
        phi: np.ndarray,
        lmax: int = 4,
        wx: Optional[np.ndarray] = None,
        wphi: Optional[np.ndarray] = None,
        gpu: bool = False,
    ):
        """Allocates the angular matrix element."""
        self._iter = 0

        if gpu and cp is None:
            raise ValueError("Could not load cupy but specified gpu backend.")

        self.x = x
        self.phi = phi
        self.wx = wx
        self.wphi = wphi

        self.lmax = lmax

        self.columns = None
        self.nchannels = None
        self._allocate_channels()

        self.matrix = None
        self._allocate_matrix()

        if gpu:
            self.matrix = cp.array(self.matrix)
            self.x = cp.array(self.x)
            self.phi = cp.array(self.phi)
            self.wx = cp.array(self.wx)
            self.wphi = cp.array(self.wphi)

    def _allocate_channels(self):
        """Allocates the quantum channels."""
        self.columns = ("lo", "li", "la", "mla")

        channels = []
        for lo, li in product(range(self.lmax + 1), range(self.lmax + 1)):
            for la in range(abs(li - lo), li + lo + 1):
                for mla in range(-la, la + 1):
                    channels.append((lo, li, la, mla))

        self.nchannels = len(channels)
        self.channels = np.array(channels, dtype=int)
        self.channel_df = DataFrame(data=self.channels, columns=self.columns)

    def _allocate_matrix(self):
        """Allocates the angular matrix element."""
        if len(self.x.shape) != 1:
            raise AssertionError("x-array should be one dimensional.")

        if len(self.phi.shape) != 1:
            raise AssertionError("phi-array should be one dimensional.")

        nx = len(self.x)
        nphi = len(self.phi)

        self.matrix = np.zeros(
            shape=(self.nchannels, nx, nx, nphi), dtype=np.complex128
        )

        # Put half here because np.exp(-1j * np.pi) / np.exp(-2j * np.pi) ** (1/2) == -1
        e_i_phi_half = np.exp(1j * self.phi / 2).reshape((1, 1, nphi))

        phi = 0
        theta = np.arccos(self.x)  # pylint: disable=E1111

        ylmi = {}
        ylmo = {}
        for ll in range(self.lmax + 1):
            for ml in range(-ll, ll + 1):
                # WARNING: scipy uses the opposite convention as we use in physics
                # e.g.: l=n, ml=m, phi=theta [0, 2pi], theta=phi [0, pi]
                # ---> left is physics, right is scipy <---
                # The scipy call structure is thus sph_harm(m, n, theta, phi)
                # which means we want sph_harm(ml, l, phi, theta)
                yml = sph_harm(ml, ll, phi, theta)
                ylmo[ll, ml] = yml.reshape((nx, 1, 1))
                ylmi[ll, ml] = yml.reshape((1, nx, 1))

        for idx, (lo, li, la, mla) in enumerate(self.channels):
            for mlo, mli in product(range(-lo, lo + 1), range(-li, li + 1)):
                if mli + mla != mlo:
                    continue

                # Note that Y_lm have been computed for phi = 0
                # Since exp(I mla Phi) is in spin element
                # This term only contains (ml_i + ml_o) / 2
                self.matrix[idx] += (
                    ylmo[lo, mlo]
                    * ylmi[li, mli]
                    * e_i_phi_half ** (mli + mlo)
                    * cg(li, mli, la, mla, lo, mlo, numeric=True)
                    * (2 * la + 1)
                    / (2 * lo + 1)
                )

    def __iter__(self):
        """Iterates angular polynomial."""
        return self

    def __next__(self) -> Tuple[Dict[str, int], np.ndarray]:
        """Returns channel and matrix."""
        if self._iter < self.nchannels:
            ii = self._iter
            self._iter += 1
        else:
            self._iter = 0
            raise StopIteration
        return self.channel_df.loc[ii].to_dict(), self.matrix[ii]

    def integrate(
        self, matrix: np.ndarray, mla: int, adaptive_chunks: bool = True,
    ):
        r"""Runs angular integrations against provided matrix.

        $$
        \\int d x_o d x_i d phi A(lo, li, la, mla) M(mla)
        $$

        Arguments:
            matrix: The array to integrate against. Last three dimensions must be
                x_o, x_i, phi too match integration.
            mla: Value for m_lambda to integrate against (filters polynomial).
            max_chunk_size: If arrays become to large, this specifies how many ang poly
                channels will be integrated over. Reduce this number to decrease memory
                bandwith but decrease performance. Defaults to all channels at once.

        Notes:
            For this method to work, you must set wx and wphi attributes.
        """
        if self.wx is None:
            raise ValueError(
                "Integration weight `wx` not specified (pass it to class init)."
            )
        if self.wphi is None:
            raise ValueError(
                "Integration weight `wphi` not specified (pass it to class init)."
            )
        mat_shape = matrix.shape
        if mat_shape[-3:] != self.matrix.shape[-3:]:
            raise ValueError("Shape of input matrix does not match own shape.")

        # find channels to integrate over
        mask = self.channels[:, -1] == mla
        if mask.sum() == 0:
            warn(
                "Found no reduced angular polynomial channels for"
                f" mla = {mla} and lmax = {self.lmax}."
            )

        # prepare integratrion kernel and bring in proper shape
        kernel = (
            self.matrix[mask]
            * self.wx.reshape(len(self.wx), 1, 1)
            * self.wx.reshape(1, len(self.wx), 1)
            * self.wphi.reshape(1, 1, len(self.wphi))
        ).reshape(
            mask.sum(),
            *[1] * (len(mat_shape) - 3),
            len(self.wx),
            len(self.wx),
            len(self.wphi),
        )

        if (
            cp is not None
            and isinstance(matrix, cp.ndarray)
            and not isinstance(kernel, cp.ndarray)
        ):
            LOGGER.debug("Porting angular poly kernel to GPU")
            kernel = cp.array(kernel)

        chunks = 1
        if adaptive_chunks:
            LOGGER.debug("Adaptively setting size of array.")
            matrix_size = np.product(mat_shape) * len(mask)
            total_bytes = matrix_size * 2 * 16  # two matrices of this size, complex

            mem = get_available_memory(
                gpu=cp is not None and isinstance(matrix, cp.ndarray)
            )

            LOGGER.debug("Available memory: %1.2f GB.", mem / 1024 ** 3)
            LOGGER.debug("Expected array sizes: %1.2f GB.", total_bytes / 1024 ** 3)

            if total_bytes > 0.9 * mem:
                chunks = int(total_bytes / (0.9 * mem))
                LOGGER.debug("Dividing up integration into %d chunks.", chunks)

        out = {}
        for channels_chunk, kernel_chunk in tqdm(
            zip(
                np.array_split(self.channels[mask], chunks),
                np.array_split(kernel, chunks),
            )
        ):
            res_chunk = (kernel_chunk * matrix.reshape(1, *mat_shape)).sum(
                axis=(-3, -2, -1)
            )
            for channel, res in zip(channels_chunk, res_chunk):
                out[tuple(channel)] = res

        return out
