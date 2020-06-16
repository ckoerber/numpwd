"""Tests for the numeric integration routines."""
from pytest import raises

from numpy import arange, zeros
from numpy.testing import assert_array_equal

from numpwd.integrate.numeric import ExpressionMap


def test_basic_usage():
    """Checks if ExpressionMap returns expected results."""
    expr = "a * b / c"

    a = arange(0, 3)
    b = arange(1, 2) * 10
    c = arange(1, 4)

    result = a.reshape(len(a), 1, 1) * b.reshape(1, len(b), 1) / c.reshape(1, 1, len(c))
    actual = ExpressionMap(expr, ("a", "b", "c"))(a, b, c)
    assert_array_equal(result, actual)


def test_independet_args():
    """Checks if ExpressionMap has right shape even if independent of expressions."""
    expr = "a * b"

    a = arange(0, 3)
    b = arange(1, 2) * 10
    c = arange(1, 4)

    result = zeros((len(a), len(b), len(c)))
    result += a.reshape(len(a), 1, 1) * b.reshape(1, len(b), 1)
    actual = ExpressionMap(expr, ("a", "b", "c"))(a, b, c)

    assert_array_equal(result, actual)


def test_missing_args():
    """Checks if error is raised when not all arguments are passed."""
    expr = "a * b / c"

    with raises(KeyError):
        ExpressionMap(expr, ("a", "b"))
