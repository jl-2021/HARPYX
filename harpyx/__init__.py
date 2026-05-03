"""
harpyx: xarray integration for HARPY HAR and SL4 files

Provides functions to read (and write, for HAR) HARPY files as xarray
DataArrays and Datasets with full support for labelled dimensions and
coordinates. SL4 (GEMPACK solution) files are read-only.

Basic Usage
-----------
>>> import harpyx
>>>
>>> # Read entire HAR file as Dataset
>>> ds = harpyx.read_har_to_dataset('model.har')
>>>
>>> # Read single header as DataArray
>>> production = harpyx.read_har_to_dataarray('model.har', 'PROD')
>>>
>>> # Write DataArray to HAR file
>>> harpyx.write_dataarray_to_har(production, 'output.har')
>>>
>>> # Read an SL4 solution file
>>> sol = harpyx.read_sl4_to_dataset('solution.sl4')

See Also
--------
harpyx.io : HAR I/O functions
harpyx.sl4 : SL4 read functions
harpyx.validation : Metadata validation
"""

__version__ = "0.1.0"
__author__ = "James Lennox"

from .io import (
    list_har_headers,
    read_har_sets,
    read_har_to_dataarray,
    read_har_to_dataset,
    write_dataarray_to_har,
    write_dataset_to_har,
    write_sets_to_har,
)
from .sl4 import (
    list_sl4_variables,
    read_sl4_sets,
    read_sl4_to_dataarray,
    read_sl4_to_dataset,
)
from .validation import validate_har_metadata

__all__ = [
    "__version__",
    "read_har_sets",
    "read_har_to_dataarray",
    "read_har_to_dataset",
    "write_sets_to_har",
    "write_dataarray_to_har",
    "write_dataset_to_har",
    "validate_har_metadata",
    "list_har_headers",
    "read_sl4_sets",
    "read_sl4_to_dataarray",
    "read_sl4_to_dataset",
    "list_sl4_variables",
]
