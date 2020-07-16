"""H5py tools to simplify parsing h5files."""
from typing import Optional, Union, Dict, Any, Tuple, Callable
from os import path

from numpy import ndarray, array
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


class H5ValuePrep:
    """Wrapper class which converts arbitrary dtypes into h5 writable format.
    """

    def __init__(
        self,
        registry: Dict[
            object, Callable[[object], Tuple[Dict[str, Any], Dict[str, Any]]]
        ] = None,
    ):
        self.registry = registry or {}

    def __call__(self, obj) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Try parsing data to expected shape."""
        if isinstance(obj, ndarray) or isinstance(obj, str):
            return {"data": obj}, {}
        else:
            try:
                if array(obj).dtype != object:
                    return {"data": obj}, {}
            except Exception:
                pass

        obj_type = str(type(obj))
        for cls, prep_fcn in self.registry.items():
            if isinstance(obj, cls):
                return prep_fcn(obj)

        raise TypeError(f"Don't know how to prepare data of type {obj_type}")


def write_data(
    data: Dict[str, Any],
    container: Union[File, Group],
    parent_name: Optional[str] = None,
    h5_value_prep: Optional[H5ValuePrep] = None,
    **kwargs,
) -> Dict[str, Union[Dataset, ndarray]]:
    """Access an HDF5 container and (recursivly) write dictionary keys into datasets.

    Arguments:
        container: Union[h5py.File, h5py.Group]
            The HDF5 group or file to iteratetively search.

        parent_name: Optional[str] = None
            The name of the parent container.
    """
    h5_value_prep = h5_value_prep or H5ValuePrep()

    for key, val in data.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Key {parent_name}/{key} has illeagel format."
                " Can only write string keys."
            )

        address = path.join(parent_name, key) if parent_name else key

        if isinstance(val, dict):
            write_data(val, container, parent_name=address)
        elif isinstance(val, list):
            write_data(
                {str(n): val for n, val in enumerate(val)},
                container,
                parent_name=address,
            )
        elif val is None:
            continue
        else:
            options, attrs = h5_value_prep(val)
            dset = container.create_dataset(address, **kwargs, **options)
            for key, val in attrs.items():
                dset.attrs[key] = val
