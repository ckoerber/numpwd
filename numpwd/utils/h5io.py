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
        write_registry: Dict[
            object, Callable[[object], Tuple[Dict[str, Any], Dict[str, Any]]]
        ] = None,
        read_registry: Dict[
            object, Callable[[object], Tuple[Dict[str, Any], Dict[str, Any]]]
        ] = None,
    ):
        self.write_registry = write_registry or {}
        self.read_registry = read_registry or {}

    def prepare(self, obj) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Try parsing data to expected shape."""
        if isinstance(obj, ndarray) or isinstance(obj, str):
            return {"data": obj}, {}
        else:
            for cls, prep_fcn in self.write_registry.items():
                if isinstance(obj, cls):
                    return prep_fcn(obj)
            try:
                if array(obj).dtype != object:
                    return {"data": obj}, {}
            except Exception:
                pass

            raise TypeError(f"Don't know how to prepare data of type {type(obj)}")

    def read(self, obj, **kwargs) -> Any:
        """Try parsing data to expected shape."""
        dtype = kwargs.pop("dtype", None)
        if dtype is not None:
            if dtype not in self.read_registry:
                raise TypeError(f"Don't know how to read data of type {dtype}")
            obj = self.read_registry[dtype](obj, **kwargs)
        return obj


def write_data(
    data: Dict[str, Any],
    parent: Union[File, Group],
    name: str,
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
    h5_value_prep = h5_value_prep if h5_value_prep is not None else H5ValuePrep()

    dsets = []
    if isinstance(data, dict):
        container = parent.create_group(name)
        container.attrs["dtype"] = "dict"
        for key, val in data.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Key {key} has illeagel format. Can only write string keys."
                )
            dsets += write_data(
                val, container, key, h5_value_prep=h5_value_prep, **kwargs
            )

    elif isinstance(data, (tuple, list)):
        container = parent.create_group(name)
        container.attrs["dtype"] = str(type(data))
        for n, el in enumerate(data):
            dsets += write_data(
                el, container, str(n), h5_value_prep=h5_value_prep, **kwargs
            )

    elif data is None:
        pass

    else:
        options, attrs = h5_value_prep.prepare(data)
        dset = parent.create_dataset(name, **kwargs, **options)
        for key, val in attrs.items():
            dset.attrs[key] = val

        dsets += [dset]

    return dsets


def read_data(
    container: Union[File, Group], h5_value_prep: Optional[H5ValuePrep] = None,
):

    h5_value_prep = h5_value_prep if h5_value_prep is not None else H5ValuePrep()

    dsets = get_dsets(container, load_dsets=True)
