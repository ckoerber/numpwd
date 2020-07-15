"""Tests assoctiated with scalar dark matter two-pion exchange current."""
import pytest

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from sympy import S, sqrt, expand_trig

from numpwd.qchannels.lsj import project_op
from numpwd.qchannels.spin import get_spin_matrix_element, dict_to_data
from numpwd.integrate.analytic import SPHERICAL_BASE_SUBS, ANGLE_BASE_SUBS, integrate


@pytest.fixture(name="two_pion_numerator_expr")
def fixture_two_pion_numerator_expr():
    """Returns numerator of two-pion exchange current.

    op = sigma1 . k1 sigma2 . k2 with k1,2 = q/2 +/- (p_i - p_o)
    """
    sig1_k1 = S("sigma11 * l11 + sigma12  * l12 + sigma13 * l13")
    sig2_k2 = S("sigma21 * l21 + sigma22  * l22 + sigma23 * l23")

    kernel = sig1_k1 * sig2_k2
    kernel = kernel.subs(
        {"l11": "+p_i1 - p_o1", "l12": "+p_i2 - p_o2", "l13": "+p_i3 - p_o3 + q/2"}
    )
    kernel = kernel.subs(
        {"l21": "-p_i1 + p_o1", "l22": "-p_i2 + p_o2", "l23": "-p_i3 + p_o3 + q/2"}
    )
    return kernel


@pytest.fixture(name="two_pion_numerator_spin_element")
def fixture_two_pion_numerator_spin_element(two_pion_numerator_expr):
    """Computes < s_o ms_o | op | s_i ms_i > for op defined above."""
    return get_spin_matrix_element(two_pion_numerator_expr)


@pytest.fixture(name="two_pion_numerator_first_integral")
def fixture_two_pion_numerator_first_integral(two_pion_numerator_spin_element):
    """Returns spin pwd plus integtal.

    Projects < s_o ms_o | op | s_i ms_i > onto sigma, m_sigma, multiplies by
    exp(I m_sigma (Phi - phi/2)) and integrates out Phi.
    """
    df = DataFrame(
        dict_to_data(
            project_op(two_pion_numerator_spin_element, "s_o", "s_i", value_key="expr"),
            columns=["s_o", "s_i", "sigma", "m_sigma"],
            value_key="expr",
        )
    )
    df["res"] = df.apply(
        lambda el: integrate(
            expand_trig(
                el["expr"].subs(SPHERICAL_BASE_SUBS).subs(ANGLE_BASE_SUBS)
                * S(f"exp(I * {el['m_sigma']} * (Phi - phi /2))")
            )
            .rewrite(S("exp"))
            .simplify()
            .expand()
        ),
        axis=1,
    )

    return (
        df[df["res"] != 0]
        .dropna()
        .set_index(["s_o", "s_i", "sigma", "m_sigma"])
        .sort_index()[["res"]]
    )


@pytest.fixture(name="legacy_results")
def fixture_legacy_results():
    """Returns legacy results published in ... for the complete numerator pipeline."""
    alpha = S("p_i**2 + p_o**2 - q**2/4")
    beta = S("exp(I * phi) * p_i * sqrt(1 - x_i**2) - p_o * sqrt(1 - x_o**2)")
    beta_conj = S("exp(-I * phi) * p_i * sqrt(1 - x_i**2) - p_o * sqrt(1 - x_o**2)")
    delta = S("p_i * x_i - p_o * x_o")
    gamma = S("p_i * p_o * (x_i * x_o  + cos(phi)* sqrt(1-x_i**2) * sqrt(1-x_o**2))")
    q = S("q")
    pi = S("pi")

    return (
        DataFrame(
            [
                {
                    "s_o": 0,
                    "s_i": 0,
                    "sigma": 0,
                    "m_sigma": 0,
                    "res": 2 * pi * (alpha - 2 * gamma),
                },
                {
                    "s_o": 0,
                    "s_i": 1,
                    "sigma": 1,
                    "m_sigma": -1,
                    "res": -sqrt(6) * pi * beta * q,
                },
                {
                    "s_o": 0,
                    "s_i": 1,
                    "sigma": 1,
                    "m_sigma": 1,
                    "res": -sqrt(6) * pi * beta_conj * q,
                },
                {
                    "s_o": 1,
                    "s_i": 0,
                    "sigma": 1,
                    "m_sigma": -1,
                    "res": -sqrt(2) * pi * beta * q,
                },
                {
                    "s_o": 1,
                    "s_i": 0,
                    "sigma": 1,
                    "m_sigma": 1,
                    "res": -sqrt(2) * pi * beta_conj * q,
                },
                {
                    "s_o": 1,
                    "s_i": 1,
                    "sigma": 0,
                    "m_sigma": 0,
                    "res": -S("2/3*pi") * (alpha - 2 * gamma),
                },
                {
                    "s_o": 1,
                    "s_i": 1,
                    "sigma": 2,
                    "m_sigma": -2,
                    "res": -S("2*sqrt(5/3)*pi") * beta ** 2,
                },
                {
                    "s_o": 1,
                    "s_i": 1,
                    "sigma": 2,
                    "m_sigma": -1,
                    "res": -S("4*sqrt(5/3)*pi") * beta * delta,
                },
                {
                    "s_o": 1,
                    "s_i": 1,
                    "sigma": 2,
                    "m_sigma": +0,
                    "res": -S("1/3*sqrt(10)*pi")
                    * (
                        alpha
                        + 3 * delta ** 2
                        - 3 * (beta * beta_conj).simplify()
                        - 2 * gamma
                        - S("3 / 4") * q ** 2
                    ),
                },
                {
                    "s_o": 1,
                    "s_i": 1,
                    "sigma": 2,
                    "m_sigma": +1,
                    "res": S("4*sqrt(5/3)*pi") * beta_conj * delta,
                },
                {
                    "s_o": 1,
                    "s_i": 1,
                    "sigma": 2,
                    "m_sigma": +2,
                    "res": -S("2*sqrt(5/3)*pi") * beta_conj ** 2,
                },
            ]
        )
        .set_index(["s_o", "s_i", "sigma", "m_sigma"])
        .sort_index()
        .applymap(lambda el: el.rewrite(S("exp")).simplify().expand())
    )


def test_legacy_numerator_pwd(two_pion_numerator_first_integral, legacy_results):
    """Asserts legacy operator matches new computation."""
    assert_frame_equal(two_pion_numerator_first_integral, legacy_results)
