# pylint: disable=C0103
"""Tests for the angular polynomial
"""
from unittest import TestCase

import numpy as np
import pandas as pd

from numpwd.integrate.angular import ReducedAngularPolynomial, get_x_mesh, get_phi_mesh


class ReducedAngularPolynomialTestCase(TestCase):  # pylint: disable=R0902
    """Tests for the reducde angular polynomial
    """

    places = 7

    def setUp(self):
        """Sets up the angular mesh
        """
        self.nx = 10
        self.x, self.wx = get_x_mesh(self.nx)

        self.nphi = 20
        self.phi, self.wphi = get_phi_mesh(self.nphi)

        self.lmax = 2
        self.poly = ReducedAngularPolynomial(self.x, self.phi, lmax=self.lmax)

    def test_01_x_mesh(self):
        r"""Tests if x integrations work as expected.

        Integrates
        $$
        \\int_{0}^{\\pi} d \\theta \\sin \\theta \\cos^2 \\theta
        =
        \\int_{-1}^{1} d x x^2
        =
        \\frac{2}{3}
        $$
        """
        integral = np.sum(self.x ** 2 * self.wx)
        self.assertAlmostEqual(2 / 3, integral, places=self.places)

    def test_02_phi_mesh(self):
        r"""Tests if phi integrations work as expected.

        Integrates
        $$
        \\int_{0}^{2 \\pi} d \\phi \\cos^2 \\phi
        =
        \\pi
        $$
        """
        integral = np.sum(np.cos(self.phi) ** 2 * self.wphi)
        self.assertAlmostEqual(np.pi, integral, places=self.places)

    def test_03_shapes(self):
        """Test if angular polynomial has correct shapes
        """
        with self.subTest("Test number of symmetric channels"):
            # for each li == lo we have 0...2li entries for la
            ## This is Sum(l=0..lmax, 2l+1) = (lmax + 1) + lmax * (lmax+1)
            n_symmetric_expected = self.lmax * (self.lmax + 1) + (1 + self.lmax)
            n_symmetric_actual = (
                self.poly.channel_df.query("lo == li")[["lo", "li", "la"]]
                .drop_duplicates()
                .shape[0]
            )
            self.assertEqual(n_symmetric_expected, n_symmetric_actual)

        with self.subTest("Entries present"):
            np.testing.assert_equal(
                np.arange(self.lmax + 1), self.poly.channel_df["li"].unique()
            )
            np.testing.assert_equal(
                np.arange(self.lmax + 1), self.poly.channel_df["lo"].unique()
            )
            np.testing.assert_equal(
                np.arange(2 * self.lmax + 1), self.poly.channel_df["la"].unique()
            )
            np.testing.assert_equal(
                np.arange(-2 * self.lmax, 2 * self.lmax + 1),
                np.sort(self.poly.channel_df["mla"].unique()),
            )

        with self.subTest("Test matrix dimension"):
            self.assertEqual(
                self.poly.matrix.shape,
                (self.poly.nchannels, self.nx, self.nx, self.nphi),
            )

    def test_04_poly_iteration(self):
        """Test if iterating over channels gives the expected result
        """
        for idx, (chan, mat) in enumerate(self.poly):
            self.assertEqual(chan, self.poly.channel_df.loc[idx].to_dict())
            np.testing.assert_equal(mat, self.poly.matrix[idx])

        with self.subTest("Second iteration"):
            for idx, (chan, mat) in enumerate(self.poly):
                self.assertEqual(chan, self.poly.channel_df.loc[idx].to_dict())
                np.testing.assert_equal(mat, self.poly.matrix[idx])

    def test_05_orthogonality(self):
        r"""Tests orhtogonality of reduced angular polynomial

        $$
        \\begin{aligned}
        & 2 \\pi
        \\frac{2 \\lambda_1 + 1}{2 l_{o_1} + 1}
        \\int_{0}^{2 \\pi} d \\phi
        \\int_{0}^{\\pi} d x_o
        \\int_{0}^{\\pi} d x_i
        O_{(l_{o_1} l_{i_1}) \\lambda_1 m_{\\lambda}}^*(x_o, x_i, \\phi)
        O_{(l_{o_2} l_{i_2}) \\lambda_2 m_{\\lambda}}  (x_o, x_i, \\phi)
        \\\\\\\\ & =
        \\delta_{l_{o_1}l_{o_2}}
        \\delta_{l_{i_1}l_{i_2}}
        \\delta_{\\lambda_1\\lambda_2}
        \\end{aligned}
        $$

        This relation uses that when both polynomials have the same \\(m_{\\lambda}\\),
        The reduced angular polynomial is effectively the same as the full polynomial.
        Since the \\(\\Phi\\) integration is still missing (but constant), one also has
        to multiply by \\(2 \\pi\\).
        """
        channels = self.poly.channel_df.reset_index().rename(columns={"index": "id"})

        # create the outer product of 1 and 2 channels by joining on m_lambda
        ids = channels[["id", "mla"]]
        prod = pd.merge(ids, ids, how="outer", on="mla", suffixes=("1", "2")).drop(
            columns="mla"
        )
        prod = pd.merge(prod, channels, left_on="id1", right_on="id").drop(
            columns=["id", "mla"]
        )
        prod = pd.merge(
            prod, channels, left_on="id2", right_on="id", suffixes=("1", "2")
        ).drop(columns="id")

        # prepare integration
        tmp = np.zeros([self.nx, self.nx, self.nphi])
        wxo = self.wx.reshape([self.nx, 1, 1]) + tmp
        wxi = self.wx.reshape([1, self.nx, 1]) + tmp
        wphi = self.wphi.reshape([1, 1, self.nphi]) + tmp

        # integrate
        res = np.sum(
            self.poly.matrix[prod["id1"]].conj()
            * self.poly.matrix[prod["id2"]]
            * wxo
            * wxi
            * wphi,
            axis=(1, 2, 3),
        )

        # checks
        with self.subTest("Imag vanishes"):
            self.assertTrue(np.abs(res.imag).sum() < 1.0e-12)

        prod["res"] = (
            res.real * (2 * prod["la1"] + 1) / (2 * prod["lo1"] + 1) * np.pi * 2
        )

        with self.subTest("Off diagonal vanish"):
            non_zero_off_diag = prod.query("id1 != id2 and res > 1.e-12").shape[0]
            self.assertEqual(non_zero_off_diag, 0)

        with self.subTest("Diagonal is identity"):
            diag_entries = prod.query("id1 == id2")["res"]
            expected = np.ones(self.poly.nchannels, dtype=float)
            np.testing.assert_almost_equal(expected, diag_entries, decimal=self.places)
