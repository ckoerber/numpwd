"""Tools for numeric integrations
"""
from typing import List, Optional, Generator

from itertools import product
from numpy import meshgrid, ndarray, prod, array_split
from sympy import lambdify, Symbol


class ExpressionTensor:  # pylint: disable=R0903
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
        args_set = set(Symbol(key) for key in args)
        if not expr.free_symbols.issubset(args_set):
            raise KeyError(
                f"Arguments do not capture all symbols in expression."
                f" Missing symbols: {expr.free_symbols - args_set}"
            )

        self.expr = expr
        self.args = args
        self._func = lambdify(args, expr, modules="numpy")

    def _allocate_flat(self, *args, chunks: Optional[int] = None):
        """Converts individual linear arrays to flattend tensor arrays.

        Because internally, all in arrays are converted to the final tensor shape, it
        is possible to allocate chunkwise allocate the final tensor.
        This reduces memory bandwith in favor of performance.
        """
        if chunks is None:
            return self._func(*self._args_to_flat_tensor(*args))
        else:
            # pre allocate out array
            shape = prod(tuple(len(arg) for arg in args))
            dtype = self._func(*(arg[0] for arg in args)).dtype
            # create array agnostic of backend
            out = type(args[0])(shape, dtype=dtype)

            # chunkwise allocate out array by in array chunks
            start = 0
            for chunk in self._chunkwise_args_to_flat_tensor(*args, chunks=(chunks,)):
                end = start + chunk[0].size
                out[start:end] = self._func(*chunk)
                start = end
            return out

    @classmethod
    def _chunkwise_args_to_flat_tensor(
        cls, *args: List[ndarray], chunks: Optional[List[int]] = None
    ) -> Generator[List[ndarray], None, None]:
        """
        """
        chunks = chunks or []
        chunked_args = []
        for n_arg, arg in enumerate(args):
            chunk_size = chunks[n_arg] if n_arg < len(chunks) else 1

            if n_arg > 1 and chunk_size > 1:
                raise NotImplementedError(
                    "Chunking arguments different than first currently not supported."
                )

            chunked_args.append(array_split(arg, chunk_size))

        for args_chunk in product(*chunked_args):
            yield cls._args_to_flat_tensor(*args_chunk)

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

    def __call__(self, *args: List[ndarray], chunks=None) -> ndarray:
        """Convert sympy expression to numpy tensor for given input variables

        Arguments:
            args: List of 1D arrays
        """
        shape = tuple(len(arg) for arg in args)
        flat_res = self._allocate_flat(*args, chunks=chunks)

        if hasattr(flat_res, "shape"):
            return flat_res.reshape(shape)
        # create constant array agnostic of backend
        elif isinstance(flat_res, (int, float, complex)):
            empty = type(args[0])(shape)
            empty[:] = flat_res
            return empty
        else:
            raise TypeError(
                "Could not create tensor for provided argument types."
                f" Return type of sympy to numpy conversion: {type(flat_res)}"
            )

    def __str__(self) -> str:
        """Method returns operator and arg string."""
        return "f(" + ", ".join(self.args) + f") = {self.expr}"
