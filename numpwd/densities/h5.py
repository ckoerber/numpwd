"""H5 Specialization of densities which populate relavant fields from H5Files.

This interface matches densities Andreas provides.
"""
from itertools import product
from re import search

from h5py import File
from pandas import DataFrame

from numpwd.utils.h5io import get_dsets
from numpwd.densities.base import Density


def read_h5(filename: str) -> Density:
    """Reads densities from h5file.

    Arguments:
        filename: Path and file name to file.
    """
    if not isinstance(filename, str):
        raise TypeError("Densities must be initialized by file name.")

    with File(filename, "r") as h5f:
        dsets = get_dsets(h5f, load_dsets=True)

    name = None
    for key in dsets:
        match = search(r"(RHO_om=[\+\-0-9\.E]+_th=[\+\-0-9\.E]+)", key)
        if match:
            name = match.group(0)
            break

    if not name:
        raise KeyError("Could not locate rho group.")

    density = Density()

    density.matrix = dsets.pop(f"{name}/RHO")
    density.p = dsets.pop("p12p")
    density.wp = dsets.pop("p12w")

    meshtype = dsets.pop("meshtype")
    if meshtype.dtype == "S1":
        meshtype = meshtype.tostring().decode("UTF-8")
    elif isinstance(meshtype[0], str):
        meshtype = "".join(meshtype)
    density.mesh_info = {"meshtype": meshtype}
    if density.mesh_info["meshtype"] == "TRNS":
        for key in ["n1", "n2", "ntot", "p1", "p2", "p3"]:
            density.mesh_info[key] = dsets.pop(f"p12{key}")

    density.current_info = {
        "qval": dsets.pop(f"{name}/qval"),
        "omval": dsets.pop(f"{name}/omval"),
        "thetaqval": dsets.pop(f"{name}/thetaqval"),
        "thetaval": dsets.pop(f"{name}/thetaval"),
    }

    qn2Nchan = dsets.pop("qn2Nchan")
    rhoindx = dsets.pop("rhoindx") - 1

    density.jx2 = int(qn2Nchan.T[-1].max())

    data = []
    columns = ["l", "s", "j", "t", "mt", "mj", "mjtotx2"]
    cols_o = list(map(lambda el: f"{el}_o", columns))
    cols_i = list(map(lambda el: f"{el}_i", columns))
    for (id_o, c_o), (id_i, c_i) in product(
        *[enumerate(qn2Nchan), enumerate(qn2Nchan)]
    ):
        channel = dict(zip(cols_o, c_o))
        channel.update(dict(zip(cols_i, c_i)))
        channel["id"] = rhoindx[id_o, id_i]
        data.append(channel)
    density.channels = DataFrame(data).query("id != -1").set_index("id").sort_index()

    density.misc = dsets
    density.misc["filename"] = filename

    return density
