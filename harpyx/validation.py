"""
Metadata validation functions for harpyx.

Validates that DataArrays have required metadata for HAR file writing.
"""

from typing import List

import xarray as xr

__all__ = ["validate_har_metadata"]


def validate_har_metadata(
    dataarray: xr.DataArray,
    *,
    require_header_id: bool = True,
    require_long_name: bool = False,
    require_coordinates: bool = True,
) -> List[str]:
    """
    Validate that a DataArray has required metadata for HAR writing.

    Parameters
    ----------
    dataarray : xr.DataArray
        Array to validate.
    require_header_id : bool, default True
        Require 'header_id' in attrs.
    require_long_name : bool, default False
        Require 'long_name' in attrs.
    require_coordinates : bool, default True
        Require all dimensions to have coordinates.

    Returns
    -------
    list of str
        List of validation errors. Empty if valid.

    Examples
    --------
    >>> import harpyx
    >>> import xarray as xr
    >>>
    >>> da = xr.DataArray(
    ...     [[1, 2], [3, 4]],
    ...     dims=['COM', 'REG'],
    ...     coords={'COM': ['A', 'B'], 'REG': ['X', 'Y']},
    ...     attrs={'header_id': 'PROD'},
    ... )
    >>> errors = harpyx.validate_har_metadata(da)
    >>> len(errors)
    0
    >>>
    >>> da2 = xr.DataArray([[1, 2]], dims=['dim_0', 'dim_1'])
    >>> errors = harpyx.validate_har_metadata(da2)
    >>> print(errors[0])
    Missing required attribute: 'header_id'
    """
    errors: List[str] = []

    if require_header_id and "header_id" not in dataarray.attrs:
        errors.append("Missing required attribute: 'header_id'")

    if require_long_name and "long_name" not in dataarray.attrs:
        errors.append("Missing required attribute: 'long_name'")

    if require_coordinates:
        for dim in dataarray.dims:
            if dim not in dataarray.coords:
                errors.append(f"Missing coordinates for dimension '{dim}'")

    return errors
