"""Read operators from H5 file."""
from h5py import File as H5File
from pandas import DataFrame, Series, read_csv
from io import StringIO

from sympy import AtomicExpr, Expr, sympify
from datetime import datetime

from numpwd.utils.h5io import write_data, H5ValuePrep, read_data
from numpwd.operators.base import Operator

try:
    import cupy as cp
except ImportError:
    cp = None


def prep_sympy(expr: Expr):
    """Converts sympy expression to h5 writable format."""
    return {"data": str(expr)}, {"dtype": "sympy"}


def read_sympy(arg, **kwargs) -> Expr:
    """Reads sympy expression from string to expression."""
    return sympify(arg)


def prep_datetime(dt):
    """Converts datetime object to string format and safes format as h5 meta."""
    format = "%Y-%m-%d %H:%M:%S"
    return {"data": dt.strftime(format)}, {"dtype": "datetime", "format": format}


def read_datetime(arg, **kwargs):
    """Reads datetime from h5 string and parses to format."""
    return datetime.strptime(arg, kwargs["format"])


def prep_dataframe(df):
    """Converts data frame to array and stores columns if possible, else to csv."""
    meta = {"dtype": "dataframe"}
    if df.index.nlevels == 1:
        meta["index_col"] = df.index.name or "index"
    else:
        meta["index_cols"] = [
            key or f"level_{n}" for n, key in enumerate(df.index.names)
        ]

    tmp = df.reset_index()
    meta["columns"] = tmp.columns.to_list()

    values = tmp.values
    values = values if values.dtype != object else tmp.to_csv(index=False)

    column_dtypes = {f"{key}_dt": str(val) for key, val in tmp.dtypes.items()}
    meta.update(column_dtypes)

    return {"data": values}, meta


def read_dataframe(arg, **kwargs):
    """Converts csv or array data to dataframe."""
    if isinstance(arg, bytes):
        arg = arg.decode("utf-8")

    df = (
        read_csv(StringIO(arg))
        if isinstance(arg, str)
        else DataFrame(data=arg, columns=kwargs["columns"])
    )

    dtypes = {
        key.replace("_dt", ""): val
        for key, val in kwargs.items()
        if key.endswith("_dt")
    }
    if dtypes:
        df = df.astype(dtypes)

    if "index_col" in kwargs and kwargs["index_col"] in df.columns:
        df = df.set_index(kwargs["index_col"])

    if "index_cols" in kwargs and set(kwargs["index_cols"]).issubset(df.columns):
        df = df.set_index(list(kwargs["index_cols"]))

    return df


def prep_cupy(array):
    """Moves array to cpu (numpy) before writing."""
    return {"data": cp.asnumpy(array)}, {"dtype": "cupy"}


def read_cupy(arg, **kwargs):
    """Moves array to GPU if cupy is available. Else CPU."""
    return arg
    # return cp.array(arg) if cp is not None else arg


PREP_MAP = {
    (AtomicExpr, Expr): prep_sympy,
    datetime: prep_datetime,
    (DataFrame, Series): prep_dataframe,
}
if cp is not None:
    PREP_MAP[cp.ndarray] = prep_cupy

READ_MAP = {
    "sympy": read_sympy,
    "datetime": read_datetime,
    "cupy": read_cupy,
    "dataframe": read_dataframe,
}


H5_VALUE_PREP = H5ValuePrep(write_registry=PREP_MAP, read_registry=READ_MAP)


def write(operator: Operator, filename: str):
    """Writes operator from h5file.

    Arguments:
        operator: operator to export.
        filename: Path and file name to file.
    """
    with H5File(filename, "w") as h5f:
        for key in ["matrix", "channels", "args", "isospin", "mesh_info", "misc"]:
            write_data(getattr(operator, key), h5f, key, h5_value_prep=H5_VALUE_PREP)


def read(filename: str) -> Operator:
    """Reads operator from h5file.

    Arguments:
        filename: Path and file name to file.
    """
    operator = Operator()
    with H5File(filename, "r") as h5f:
        for key in ["matrix", "channels", "args", "isospin", "mesh_info", "misc"]:
            setattr(operator, key, read_data(h5f[key], h5_value_prep=H5_VALUE_PREP))

    return operator
