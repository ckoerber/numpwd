"""Read operators from H5 file."""
from itertools import product
from re import search

from h5py import File
from pandas import DataFrame

from sympy import AtomicExpr, Expr
from datetime import datetime

from numpwd.utils.h5io import get_dsets, write_data, H5ValuePrep
from numpwd.densities.base import Density


def prep_sympy(expr):
    return {"data": str(expr)}, {"dtype": "sympy"}


def prep_datetime(dt):
    format = "%Y-%m-%d %H:%M:%S"
    return {"data": dt.strftime(format)}, {"dtype": "datetime", "format": format}


H5_VALUE_PREP = H5ValuePrep({(AtomicExpr, Expr): prep_sympy, datetime: prep_datetime})


def read_h5(filename: str) -> Density:
    """Reads operator from h5file.

    Arguments:
        filename: Path and file name to file.
    """
    if not isinstance(filename, str):
        raise TypeError("Densities must be initialized by file name.")

    with File(filename, "r") as h5f:
        dsets = get_dsets(h5f, load_dsets=True)
