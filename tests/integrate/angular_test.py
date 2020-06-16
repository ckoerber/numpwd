# pylint: disable=C0103
"""Tests for the angular polynomial."""
from itertools import product

from unittest import TestCase

import numpy as np
import pandas as pd

from scipy.special import sph_harm

from numpwd.qchannels.cg import get_cg
from numpwd.integrate.angular import ReducedAngularPolynomial, get_x_mesh, get_phi_mesh


class ReducedAngularPolynomialTestCase(TestCase):  # pylint: disable=R0902
    """Tests for the reducde angular polynomial."""

    places = 7

    def setUp(self):
        """This method creates angular meshes."""
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
        """Tests if angular polynomial has correct shapes."""
        with self.subTest("Test number of symmetric channels"):
            # for each li == lo we have 0...2li entries for la
            # This is Sum(l=0..lmax, 2l+1) = (lmax + 1) + lmax * (lmax+1)
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
        """Tests if iterating over channels gives the expected result."""
        for idx, (chan, mat) in enumerate(self.poly):
            self.assertEqual(chan, self.poly.channel_df.loc[idx].to_dict())
            np.testing.assert_equal(mat, self.poly.matrix[idx])

        with self.subTest("Second iteration"):
            for idx, (chan, mat) in enumerate(self.poly):
                self.assertEqual(chan, self.poly.channel_df.loc[idx].to_dict())
                np.testing.assert_equal(mat, self.poly.matrix[idx])

    def test_05_orthogonality(self):
        r"""Tests orhtogonality of reduced angular polynomial.

        $$
        \\begin{aligned}
        &
        2 \\pi
        \\int_{0}^{2 \\pi} d \\phi
        \\int_{0}^{\\pi} d x_o
        \\int_{0}^{\\pi} d x_i
        O_{(l_{o_1} l_{i_1}) \\lambda_1 m_{\\lambda}}^*(x_o, x_i, \\phi)
        O_{(l_{o_2} l_{i_2}) \\lambda_2 m_{\\lambda}}  (x_o, x_i, \\phi)
        \\\\\\\\ & =
        \\frac{2 \\lambda_1 + 1}{2 l_{o_1} + 1}
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

        # Normalize result to be equal to one
        prod["res"] = (
            res.real / (2 * prod["la1"] + 1) * (2 * prod["lo1"] + 1) * np.pi * 2
        )

        with self.subTest("Off diagonal vanish"):
            non_zero_off_diag = prod.query("id1 != id2 and res > 1.e-12").shape[0]
            self.assertEqual(non_zero_off_diag, 0)

        with self.subTest("Diagonal is identity"):
            diag_entries = prod.query("id1 == id2")["res"]
            expected = np.ones(self.poly.nchannels, dtype=float)
            np.testing.assert_almost_equal(expected, diag_entries, decimal=self.places)

    def _generate_op(self):
        r"""Routine generates spherical harmonic operator.

        Operator corresponds to
        $$
        \\int d \\Phi
        e^{i m_\\lambda \\Phi}
        Y_{l_{b1} m_{b1}}(x_1, \\phi_1) Y^*_{l_{b2} m_{b2}}(x_2, \\phi_2)
        $$
        """
        channels = []
        for lo, li in product(range(self.lmax + 1), range(self.lmax + 1)):
            for mlo, mli in product(range(-lo, lo + 1), range(-li, li + 1)):
                channels.append([lo, mlo, li, mli])

        nchannels = len(channels)
        pphi = 0
        theta = np.arccos(self.x)  # pylint: disable=E1111
        e_i_phi_half = np.exp(1j * self.phi / 2).reshape([1, 1, self.nphi])

        matrix = np.zeros(
            shape=[nchannels, self.nx, self.nx, self.nphi], dtype=np.complex128
        )

        ylmi = {}
        ylmo = {}
        for ll in range(self.lmax + 1):
            for ml in range(-ll, ll + 1):
                # WARNING: scipy uses the opposite convention as we use in physics
                #: e.g.: l=n, ml=m, phi=theta [0, 2pi], theta=phi [0, pi]
                # ---> left is physics, right is scipy <---
                # The scipy call structure is thus sph_harm(m, n, theta, phi)
                # which means we want sph_harm(ml, l, phi, theta)
                yml = sph_harm(ml, ll, pphi, theta)
                ylmo[ll, ml] = yml.reshape([self.nx, 1, 1])
                ylmi[ll, ml] = yml.reshape([1, self.nx, 1])

        for idx, (lo, mlo, li, mli) in enumerate(channels):
            # Note that Y_lm have been computed for phi = 0
            # Since exp(I mla Phi) is in spin element
            # This term only contains (ml_i + ml_o) / 2
            matrix[idx] += (
                ylmo[lo, mlo]
                * ylmi[li, mli]
                * e_i_phi_half ** (-(mli + mlo))
                * 2
                * np.pi
            )

        return channels, matrix

    @staticmethod
    def _analytic_result(op_qn, ang_qn):
        r"""Routine returns analytic result.

        $$
        \frac{2 \lambda + 1}{2l_{b1} + 1}
        \left\langle l_{b1} m_{b1}, \lambda m_\lambda \vert l_{b1} m_{b1} \right\rangle
        \delta_{l_{a1} l_{b1}}
        \delta_{l_{a2} l_{b2}}
        $$
        """
        lo_o, mlo_o, li_o, mli_o = op_qn
        lo_a, li_a, la_a, mla_a = ang_qn

        fact = (2 * la_a + 1) / (2 * lo_o + 1)
        return (
            fact * get_cg(li_o, mli_o, la_a, mla_a, lo_o, mlo_o, numeric=True)
            if lo_o == lo_a and li_o == li_a
            else 0
        )

    def test_06_ylm_operator(self):
        r"""Tests if angular integration with ylm operator returns expected result.

        Details:

            Using the orhtogonality of spherical harmonics
            $$
            \\int d x_1 d \\phi_1
            Y_{l_a m_a}^*(x_1, \phi_1) Y_{l_b m_b}(x_1, \phi_1)
            =
            \\delta_{l_a l_b} \\delta_{m_a m_b}
            $$
            this method tests that
            $$
            \\frac{2 \\lambda + 1}{2l_{a1} + 1}
            \\sum_{m_{a1} m_{a2}}
            \\left\\langle
                l_{a1} m_{a1}, \\lambda m_\\lambda \\vert l_{a2} m_{a2}
            \\right\\rangle
            \\int d x_1 d \\phi_1 \\int d x_2 d \\phi_2
            Y_{l_{a1} m_{a1}}^*(x_1, \\phi_1) Y_{l_{a2} m_{a2}}(x_2, \\phi_2)
            Y_{l_{b1} m_{b1}}(x_1, \\phi_1) Y^*_{l_{b2} m_{b2}}(x_2, \\phi_2)
            =
            \\frac{2 \\lambda + 1}{2l_{a1} + 1}
            \\sum_{m_{a1} m_{a2}}
            \\left\\langle
                l_{a1} m_{a1}, \\lambda m_\\lambda \\vert l_{a2} m_{a2}
            \\right\\rangle
            \\delta_{l_{a1} l_{b1}} \\delta_{m_{a1} m_{b1}}
            \\delta_{l_{a2} l_{b2}} \\delta_{m_{a2} m_{b2}}
            $$
            by identifying the angular polynomial and corresponding op.
        """
        op_channels, op_matrix = self._generate_op()

        for (id_op, op_qn), (id_ang, ang_qn) in product(
            enumerate(op_channels), enumerate(self.poly.channels)
        ):
            with self.subTest("Channel product", op_qn=op_qn, ang_qn=ang_qn):
                mlo_o, mli_o = op_qn[1], op_qn[3]
                mla_a = ang_qn[3]

                res_numeric = (
                    np.sum(
                        op_matrix[id_op]
                        * self.poly.matrix[id_ang]
                        * self.wx.reshape((self.nx, 1, 1))
                        * self.wx.reshape((1, self.nx, 1))
                        * self.wphi.reshape((1, 1, self.nphi))
                    )
                    if mla_a == mlo_o - mli_o
                    else 0
                )

                res_analytic = self._analytic_result(op_qn, ang_qn)

                self.assertAlmostEqual(res_numeric, res_analytic, 14)

    def test_07_expression_1(self) -> None:  # noqa: D202
        r"""Tests if angular pwd of $(p_i - p_o) \cdot q$ returns expected result.

        Computes
        $$
        \\sum_{m_i m_o}
        \\left\\langle l_i m_i , \\lambda m_\\lambda | l_o m_o \\right\\rangle
        \int d \Omega_1 d \Omega_2
            Y_{l_o m_o}^*(\Omega_o) Y_{l_i m_i}(\Omega_i)
            \left[\vec p_o - \vec p_i \right]\cdot \vec q
        $$
        for general q using the angular PWD formalism

        Expected result (after exanding scalar product and substituing Ylms).
        """

        from numpy.polynomial.legendre import leggauss
        from sympy import expand_trig, S
        from numpwd.integrate.analytic import integrate
        from numpwd.integrate.numeric import ExpressionMap
        from numpwd.qchannels.cg import get_cg

        y00_fact = S("1/2 * sqrt(1/pi)")
        y11_fact = S("1/2 * sqrt(3/pi/2)")
        expected = [
            {
                "lo": 0,
                "li": 1,
                "la": 1,
                "mla": 1,
                "val": S("p_i")
                / y00_fact
                / y11_fact
                / 2
                * (-S("q1") - S("q2 / I"))
                * get_cg(1, 1, 1, -1, 0, 0, numeric=True)
                * 3,
            },
            {
                "lo": 0,
                "li": 1,
                "la": 1,
                "mla": -1,
                "val": S("p_i")
                / y00_fact
                / y11_fact
                / 2
                * (S("q1") - S("q2 / I"))
                * get_cg(1, 1, 1, -1, 0, 0, numeric=True)
                * 3,
            },
            {
                "lo": 1,
                "li": 0,
                "la": 1,
                "mla": 1,
                "val": S("p_o")
                / y00_fact
                / y11_fact
                / 2
                * (-S("q1") - S("q2 / I"))
                * get_cg(0, 0, 1, 1, 1, 1, numeric=True)
                * 3
                / 3,
            },
            {
                "lo": 1,
                "li": 0,
                "la": 1,
                "mla": -1,
                "val": S("p_o")
                / y00_fact
                / y11_fact
                / 2
                * (+S("q1") - S("q2 / I"))
                * get_cg(0, 0, 1, -1, 1, -1, numeric=True)
                * 3
                / 3,
            },
            {
                "lo": 0,
                "li": 1,
                "la": 1,
                "mla": 0,
                "val": -S("p_i * q3 / (1/4/pi * sqrt(3)) * 3")
                * get_cg(1, 0, 1, 0, 0, 0, numeric=True),
            },
            {
                "lo": 1,
                "li": 0,
                "la": 1,
                "mla": 0,
                "val": S("p_o * q3 / (1/4/pi * sqrt(3)) * 3 / 3")
                * get_cg(0, 0, 1, 0, 1, 0, numeric=True),
            },
        ]

        p_dot_q = (
            "p{i} * (q1 * cos(phi{i}) * sqrt(1 - x{i}**2)"
            " + q2 * sin(phi{i}) * sqrt(1 - x{i}**2)"
            " + q3 * x{i})"
        )
        expr = S(p_dot_q.format(i="_o")) - S(p_dot_q.format(i="_i"))

        # run angular integrations (only results for mla == 1 survive)
        big_phi_int_expr = {}
        for mla in range(-1, 2):
            big_phi_int_expr[mla] = integrate(
                expand_trig(expr.subs({"phi_i": "Phi + phi/2", "phi_o": "Phi - phi/2"}))
                * S(f"exp(-I * {mla} * Phi)")
            )
        # allocate grid
        p_o = np.array([2, 3])
        p_i = np.array([4, 3, 5])
        q1, q2, q3 = 1, 2, 3
        q = np.sqrt([q1 ** 2 + q2 ** 2 + q3 ** 2])

        nx = 30
        nphi = 20
        lmax = 3

        phi, wphi = get_phi_mesh(nphi)
        x, wx = leggauss(nx)

        # allocate reduced angular polynomial
        poly = ReducedAngularPolynomial(x, phi, lmax=lmax)
        poly_kernel = (
            poly.matrix
            * wx.reshape((1, nx, 1, 1))
            * wx.reshape((1, 1, nx, 1))
            * wphi.reshape((1, 1, 1, nphi))
        )

        # integrate
        df = poly.channel_df.copy()
        data = []
        for mla in df.mla.unique():
            idx = df["mla"] == mla

            expr = big_phi_int_expr.get(mla, S(0)).subs({"q1": q1, "q2": q2, "q3": q3})
            if expr:
                op_fcn = ExpressionMap(expr, ("p_o", "p_i", "q", "x_o", "x_i", "phi"))
                op_matrix = op_fcn(p_o, p_i, q, x, x, phi)
            else:
                op_matrix = np.zeros((len(p_o), len(p_i), len(q), nx, nx, nphi))

            res = np.sum(
                op_matrix.reshape((1, len(p_o), len(p_i), len(q), nx, nx, nphi))
                * poly_kernel[idx].reshape((idx.sum(), 1, 1, 1, nx, nx, nphi)),
                axis=(-1, -2, -3),
            )
            for chan, val in zip(df.loc[idx].to_dict("records"), res):
                if (np.abs(val) > 1.0e-10).any():
                    data.append({**chan, "val": val})

        print(data)

        # Run tests
        for res in data:
            with self.subTest("Check quantum numbers", channel=res):
                self.assertTrue(
                    (res["li"] == 1 and res["lo"] == 0)
                    or (res["li"] == 0 and res["lo"] == 1)
                )
                self.assertEqual(res["la"], 1)
                self.assertTrue(res["mla"] in [-1, 0, 1])

        # Convert expected results to array
        expected = {
            (el["lo"], el["li"], el["la"], el["mla"]): ExpressionMap(
                el["val"].subs({"q1": q1, "q2": q2, "q3": q3}), ("p_o", "p_i", "q")
            )(p_o, p_i, q)
            for el in expected
        }

        for res in data:
            key = (res["lo"], res["li"], res["la"], res["mla"])
            if key in expected:
                val = res.pop("val")
                np.testing.assert_almost_equal(
                    expected[key], val, err_msg=f"data: {res}"
                )
