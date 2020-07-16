"""Read operators from H5 file."""
from h5py import File as H5File
from pandas import DataFrame, Series

from sympy import AtomicExpr, Expr
from datetime import datetime

from numpwd.utils.h5io import get_dsets, write_data, H5ValuePrep, read_data
from numpwd.operators.base import Operator


def prep_sympy(expr):
    return {"data": str(expr)}, {"dtype": "sympy"}


def prep_datetime(dt):
    format = "%Y-%m-%d %H:%M:%S"
    return {"data": dt.strftime(format)}, {"dtype": "datetime", "format": format}


def prep_dataframe(df):
    tmp = df.reset_index()
    values = tmp.values
    values = values if values.dtype != object else tmp.to_csv(index=False)
    return {"data": values}, {"dtype": "dataframe", "columns": tmp.columns.to_list()}


H5_VALUE_PREP = H5ValuePrep(
    {
        (AtomicExpr, Expr): prep_sympy,
        datetime: prep_datetime,
        (DataFrame, Series): prep_dataframe,
    }
)


def write(operator: Operator, filename: str):
    """Writes operator from h5file.

    Arguments:
        operator: operator to export.
        filename: Path and file name to file.
    """
    with H5File("test.h5", "w") as h5f:
        for key in ["matrix", "channels", "args", "isospin", "mesh_info", "misc"]:
            write_data(getattr(operator, key), h5f, key, h5_value_prep=H5_VALUE_PREP)


def read(filename: str) -> Operator:
    """Reads operator from h5file.

    Arguments:
        filename: Path and file name to file.
    """
    operator = Operator()
    with H5File("test.h5", "w") as h5f:
        for key in ["matrix", "channels", "args", "isospin", "mesh_info", "misc"]:
            setattr(operator, key, read_data(h5f[key], h5_value_prep=H5_VALUE_PREP))

    return operator
