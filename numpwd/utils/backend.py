"""Backend related utility functions."""
from psutil import virtual_memory

try:
    import cupy as cp
except ImportError:
    cp = None


def get_available_memory(gpu: bool = False) -> int:
    """Returns available memory on specified backend."""
    if gpu:
        if cp is None:
            raise ValueError("Trying to infer GPU memory without cupy installed.")
        mem = cp.get_default_memory_pool().free_bytes()
    else:
        mem = virtual_memory().available
    return mem
