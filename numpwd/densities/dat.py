"""Wrapper for single nucleon densities objects (dat files)."""
import re

import numpy as np
import pandas as pd


def parse_fortran_funny(string):
    """Convert fortran format arrys in file to python objects."""
    for pat, subs in {
        f"{key}*{val}": ", ".join([val] * int(key))
        for key, val in set(
            re.findall(r"([0-9]+)\*([\-0-9]+)", re.sub(r"\s+", " ", string))
        )
    }.items():
        string = string.replace(pat, subs)

    arr = np.array(list(map(int, string.split(","))))
    nd = len(arr) // 8
    return pd.DataFrame(
        data=arr.reshape([nd, 8]),
        columns=[
            "ms3_x2",
            "mt3_x2",
            "mjtot_x2",
            "ms3p_x2",
            "mt3p_x2",
            "mjtotp_x2",
            "k",
            "bk",
        ],
    )


def read_1N_density(address):
    """Read in one-body density files."""
    pattern = r"MAXRHO1BINDEX\s+\=\s+(?P<max_rho_index>[0-9]+)"
    pattern += r".*"
    pattern += r"RHO1BINDX\s+\=(?P<rho_index>[0-9\*\,\-\s]+)"
    pattern += r".*"
    pattern += r"\/\s+(?P<om_theta>[0-9\.\-\+E ]+\n)"
    pattern += r"\s+(?P<rho>[0-9\.\-\+E\s]+\n)"

    dtypes = {
        "max_rho_index": int,
        "om_theta": lambda el: np.array([float(ee) for ee in el.split(" ") if ee]),
        "rho": lambda el: np.array([float(ee) for ee in el.split(" ") if ee]),
        "rho_index": parse_fortran_funny,
    }

    with open(address, "r") as inp:
        t = inp.read()
    dd = re.search(pattern, t, re.MULTILINE | re.DOTALL).groupdict()
    for key, val in dtypes.items():
        dd[key] = val(dd[key])

    channels = dd["rho_index"].copy()
    channels["rho"] = dd["rho"]
    dd["channels"] = channels
    return dd
