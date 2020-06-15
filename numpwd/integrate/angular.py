# pylint: disable=C0103
"""Angular operators used for numerically integrating elements
"""
from typing import Dict, Tuple

from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

from scipy.special import sph_harm  # pylint: disable=E0611

from pandas import DataFrame

from numpwd.qchannels.cg import get_cg as cg


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


class ReducedAngularPolynomial:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    r"""Stores the reduced angular polynomial

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

    def __init__(self, x: np.ndarray, phi: np.ndarray, lmax: int = 4):
        """Allocates the angular matrix element
        """
        self._iter = 0

        self.x = x
        self.phi = phi

        self.lmax = lmax

        self.columns = None
        self.nchannels = None
        self._allocate_channels()

        self.matrix = None
        self._allocate_matrix()

    def _allocate_channels(self):
        """Allocates the quantum channels
        """
        self.columns = ("lo", "li", "la", "mla")

        channels = []
        for lo, li in product(range(self.lmax + 1), range(self.lmax + 1)):
            for la in range(abs(li - lo), li + lo + 1):
                for mla in range(-la, la + 1):
                    channels.append((lo, li, la, mla))

        self.nchannels = len(channels)
        self.channels = np.array(channels, dtype=int)
        self.channel_df = DataFrame(data=self.channels, columns=self.columns)

    def _allocate_matrix(self):  # pylint: disable=R0914
        """Allocates the angular matrix element
        """
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
        for l in range(self.lmax + 1):
            for ml in range(-l, l + 1):
                # WARNING: scipy uses the opposite convention as we use in physics
                ##: e.g.: l=n, ml=m, phi=theta [0, 2pi], theta=phi [0, pi]
                ## ---> left is physics, right is scipy <---
                ## The scipy call structure is thus sph_harm(m, n, theta, phi)
                ## which means we want sph_harm(ml, l, phi, theta)
                yml = sph_harm(ml, l, phi, theta)
                ylmo[l, ml] = yml.reshape((nx, 1, 1))
                ylmi[l, ml] = yml.reshape((1, nx, 1))

        for idx, (lo, li, la, mla) in enumerate(self.channels):
            for mlo, mli in product(range(-lo, lo + 1), range(-li, li + 1)):
                if mli + mla != mlo:
                    continue

                # Note that Y_lm have been computed for phi = 0
                ## Since exp(I mla Phi) is in spin element
                ## This term only contains (ml_i + ml_o) / 2
                self.matrix[idx] += (
                    ylmo[lo, mlo]
                    * ylmi[li, mli]
                    * e_i_phi_half ** (mli + mlo)
                    * cg(li, mli, la, mla, lo, mlo, numeric=True)
                    * (2 * la + 1)
                    / (2 * lo + 1)
                )

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Dict[str, int], np.ndarray]:
        """Return channel and matrix
        """
        if self._iter < self.nchannels:
            ii = self._iter
            self._iter += 1
        else:
            self._iter = 0
            raise StopIteration
        return self.channel_df.loc[ii].to_dict(), self.matrix[ii]
