"""Read operators from H5 file."""
from itertools import product
from re import search

from h5py import File
from pandas import DataFrame

from numpwd.utils.h5io import get_dsets
from numpwd.densities.base import Density


def read_h5(filename: str) -> Density:
    """Reads operator from h5file.

    Arguments:
        filename: Path and file name to file.
    """
    if not isinstance(filename, str):
        raise TypeError("Densities must be initialized by file name.")

    with File(filename, "r") as h5f:
        dsets = get_dsets(h5f, load_dsets=True)
