"""Tools for numeric integrations."""
from typing import List

from numpy import ndarray
from sympy import lambdify, Symbol, S, separatevars


class ExpressionMap:  # pylint: disable=R0903
    """Wrapper class for converting sympy expressions to numpy tensors.

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

    def __init__(self, expr: S, args: List[Symbol]):
        """Init the wrapper with expression to convert and arguments to match.

        The arguments form the tensor dimensions of the out array.
        """
        expr = separatevars(S(expr) if isinstance(expr, str) else expr)

        self.args = tuple(Symbol(key) for key in args)
        args_set = set(self.args)
        if not expr.free_symbols.issubset(args_set):
            raise KeyError(
                f"Arguments do not capture all symbols in expression."
                f" Missing symbols: {expr.free_symbols - args_set}"
            )

        self._missing_args = args_set - expr.free_symbols

        self.expr = expr
        self._func = lambdify(args, expr, modules="numpy")

    def __call__(self, *args: List[ndarray]) -> ndarray:
        """Convert sympy expression to numpy tensor for given input variables.

        Arguments:
            args: List of 1D arrays
        """
        shape = tuple(len(arg) for arg in args)
        reshaped_args = []
        for n_arg, arg in enumerate(args):
            arg_shape = [1] * len(shape)
            arg_shape[n_arg] = len(arg)
            reshaped_args.append(arg.reshape(arg_shape))
        out = self._func(*reshaped_args)

        if self._missing_args:
            for n_arg, arg in enumerate(self.args):
                if arg in self._missing_args:
                    out = out.repeat(axis=n_arg, repeats=len(args[n_arg]))
        return out

    def __str__(self) -> str:
        """Method returns operator and arg string."""
        return "f(" + ", ".join(self.args) + f") = {self.expr}"
