"""Tools for numeric integrations
"""
from typing import List

from numpy import meshgrid, ndarray, ones
from sympy import lambdify, Symbol


class ExprToTensorWrapper:  # pylint: disable=R0903
    """Wrapper class for converting sympy expressions to numpy tensors

    Example:
        expr = "x * y"
        args = ("x", "y")
        f = ExprToTensorWrapper(expr, args)
        x = np.arange(5)
        y = np.arange(10)
        array = f(x, y)
        array.shape # (5, 10)
        all(array == x.reshape(5, 1) * y.reshape(1, 10)) # True
    """

    def __init__(self, expr: Symbol, args: List[Symbol]):
        """Init the wrapper with expression to convert and arguments to match.
        The arguments form the tensor dimensions of the out array.
        """
        self.expr = expr
        self.args = args
        self._func = lambdify(args, expr, modules="numpy")

    @staticmethod
    def _args_to_flat_tensor(*args: List[ndarray]) -> List[ndarray]:
        """Converts individual linear arrays to flattend tensor arrays

        Example:
            x, y = [1, 2], ["a", "b", "c"]
            res = ExprToTensorWrapper._args_to_flat_tensor(x, y)
            res[0] = [1, 1, 1, 2, 2, 2]
            res[1] = ["a", "b", "c", "a", "b", "c"]
        """
        return [arr.flatten() for arr in meshgrid(*args, sparse=False, indexing="ij")]

    def __call__(self, *args: List[ndarray]) -> ndarray:
        """Convert sympy expression to numpy tensor for given input variables

        Arguments:
            args: List of 1D arrays
        """
        shape = tuple(len(arg) for arg in args)
        flat_args = self._args_to_flat_tensor(*args)
        flat_res = self._func(*flat_args)
        return (
            flat_res.reshape(shape)
            if isinstance(flat_res, ndarray)
            else ones(shape) * flat_res
        )
