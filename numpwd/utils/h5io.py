"""H5py tools to simplify parsing h5files."""
from typing import Optional, Union, Dict
from os import path

from numpy import ndarray
from h5py import File, Group, Dataset


def get_dsets(
    container: Union[File, Group],
    parent_name: Optional[str] = None,
    load_dsets: bool = False,
) -> Dict[str, Union[Dataset, ndarray]]:
    """Access an HDF5 container and extracts datasets.

    The method is recursivly called if the container contains further containers.

    Arguments:
        container: Union[h5py.File, h5py.Group]
            The HDF5 group or file to iteratetively search.

        parent_name: Optional[str] = None
            The name of the parent container.

        load_dsets: bool = False
            If False, data sets are not opened (lazy load).
            If True, returns Dict with numpy arrays as values.

    Returns:
        A dictionary containing the full path HDF path (e.g., `groupA/subgroupB`)
        to the data set as keys and the unloaded values of the set as values.
    """
    dsets = {}
    for key in container:
        obj = container[key]

        address = path.join(parent_name, key) if parent_name else key

        if isinstance(obj, Dataset):
            dsets[address] = obj[()] if load_dsets else obj
        elif isinstance(obj, Group):
            dsets.update(get_dsets(obj, parent_name=address, load_dsets=load_dsets))

    return dsets
