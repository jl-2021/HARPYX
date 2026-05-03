"""
Read GEMPACK SL4 (solution) files as xarray objects.

SL4 files are decoded by HARPY's :class:`harpy.sl4.SL4` class, which returns
each variable and set as a :class:`HeaderArrayObj` — the same type used for
HAR headers. This means most of the xarray-construction logic (duplicate
dimension disambiguation, scalar handling, coord building) is shared with
:mod:`harpyx.io`.

Read-only: harpyx does not write SL4 files (HARPY does not support this).

Quirks
------
- When an SL4 file contains subtotals, HARPY appends a synthetic ``#RESULTS``
  dimension to every variable, with element labels ``["Cumulative",
  <subtotal descriptions>...]``. harpyx renames this dim to ``RESULTS`` for
  ergonomics; the original HAR name is preserved in
  ``attrs['dimension_sets']``.
- All variable arrays come back as ``float32`` regardless of any underlying
  type information — this is a HARPY behaviour.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

from harpy.sl4 import SL4

from .io import _check_file, _make_xr_dims, _strip

__all__ = [
    "list_sl4_variables",
    "read_sl4_sets",
    "read_sl4_to_dataarray",
    "read_sl4_to_dataset",
]

_RESULTS_HAR_NAME = "#RESULTS"
_RESULTS_XR_NAME = "RESULTS"
_SKIP_VARS_SENTINEL = "___harpyx_internal_skip_all_vars___"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _xr_dim_for_set_name(name: Optional[str]) -> Optional[str]:
    """Normalise a HARPY-side set name into an xarray dim name.

    HARPY's :class:`SL4` lowercases all user set names internally
    (sl4.py:125) but variables' :pyattr:`HeaderArrayObj.setNames` preserve
    the file's original case. To keep a single coordinate per set across
    a Dataset, harpyx lowercases user set names. The synthetic
    ``#RESULTS`` set is renamed to uppercase ``RESULTS`` so it stands out.
    """
    if name is None:
        return None
    stripped = str(name).strip()
    if stripped.lower() == "#results":
        return _RESULTS_XR_NAME
    return stripped.lower()


def _decode_var_type(raw) -> str:
    """varType from VCTP may be bytes or numpy str — normalise to a stripped str."""
    if isinstance(raw, bytes):
        return raw.decode("ascii", errors="ignore").strip()
    return str(raw).strip()


def _sets_from_sl4(sl4: SL4) -> Dict[str, xr.DataArray]:
    """Build set DataArrays from an already-decoded SL4 instance."""
    result: Dict[str, xr.DataArray] = {}
    for raw_name in sl4.setNames:
        hao = sl4.getSet(raw_name)
        elements = [str(e).strip() for e in hao.array.tolist()]
        dim_name = _xr_dim_for_set_name(raw_name)

        attrs: Dict = {"har_set_name": str(raw_name).strip()}
        long_name = _strip(hao.long_name)
        if long_name:
            attrs["long_name"] = long_name

        result[dim_name] = xr.DataArray(
            np.array(elements),
            dims=[dim_name],
            coords={dim_name: elements},
            name=dim_name,
            attrs=attrs,
        )
    return result


def _build_dataarray_from_hao(
    hao,
    *,
    variable_name: str,
    var_type: str,
    source_file: Optional[str],
) -> xr.DataArray:
    """Convert a variable HeaderArrayObj from SL4 into an xr.DataArray."""
    long_name = _strip(hao.long_name)
    coeff_name = _strip(hao.coeff_name)

    attrs: Dict = {
        "header_id": (coeff_name or variable_name).strip(),
        "source_format": "SL4",
        "var_type": var_type,
    }
    if long_name:
        attrs["long_name"] = long_name
    if coeff_name:
        attrs["coefficient_name"] = coeff_name
    if source_file is not None:
        attrs["source_file"] = source_file

    if hao.array.ndim == 0:
        return xr.DataArray(hao.array, attrs=attrs)

    raw_names: List[Optional[str]] = list(hao.setNames)
    raw_elements: List[Optional[List[str]]] = list(hao.setElements)

    mapped_names = [_xr_dim_for_set_name(n) for n in raw_names]
    xr_dims, _ = _make_xr_dims(mapped_names)

    coords: Dict = {}
    for xr_dim, elements in zip(xr_dims, raw_elements):
        if elements is not None:
            coords[xr_dim] = [str(e).strip() for e in elements]

    har_names = [str(n).strip() if n is not None else None for n in raw_names]
    if list(xr_dims) != har_names:
        attrs["dimension_sets"] = har_names

    return xr.DataArray(hao.array, dims=xr_dims, coords=coords, attrs=attrs)


def _normalise_var_lookup(sl4: SL4, variable_name: str) -> str:
    """Return the file's stored form of ``variable_name`` (case-insensitive lookup)."""
    target = variable_name.strip().lower()
    for n in sl4.variableNames:
        if n.strip().lower() == target:
            return n
    raise KeyError(
        f"Variable '{variable_name}' not found in SL4 file. "
        f"Available: {[n.strip() for n in sl4.variableNames]}"
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def read_sl4_sets(filepath: str) -> Dict[str, xr.DataArray]:
    """
    Read all set definitions from an SL4 file.

    Includes the synthetic ``#RESULTS`` set (renamed to ``RESULTS``) when
    the file contains subtotals.

    Parameters
    ----------
    filepath : str
        Path to SL4 file.

    Returns
    -------
    dict
        Mapping from xarray dim name to a 1-D coordinate DataArray.

    Examples
    --------
    >>> import harpyx
    >>> sets = harpyx.read_sl4_sets('solution.sl4')
    >>> list(sets)
    ['sect', 'fac', 'num_sect', 'RESULTS']
    """
    _check_file(filepath)
    sl4 = SL4(filepath, extractList=[_SKIP_VARS_SENTINEL])
    return _sets_from_sl4(sl4)


def read_sl4_to_dataarray(
    filepath: str,
    variable_name: str,
    *,
    sl4_obj: Optional[SL4] = None,
    include_file_metadata: bool = True,
) -> xr.DataArray:
    """
    Read a single SL4 variable into an xarray DataArray.

    Parameters
    ----------
    filepath : str
        Path to SL4 file.
    variable_name : str
        Name of the variable (case-insensitive).
    sl4_obj : harpy.sl4.SL4, optional
        Already-decoded SL4 instance — internal use, lets
        :func:`read_sl4_to_dataset` avoid reopening the file per variable.
    include_file_metadata : bool, default True
        If True, include source file path in attrs.

    Returns
    -------
    xr.DataArray
        Variable values with labelled dimensions and coordinates. Always
        float32. Always carries ``attrs['var_type']`` (e.g. 'c', 'p', 'l',
        'o') and ``attrs['source_format'] == 'SL4'``.
    """
    _check_file(filepath)
    sl4 = sl4_obj if sl4_obj is not None else SL4(filepath, extractList=[variable_name])

    canonical_name = _normalise_var_lookup(sl4, variable_name)
    hao = sl4.getVariable(canonical_name)
    var_type = _decode_var_type(sl4.varType(canonical_name))

    source = str(Path(filepath).resolve()) if include_file_metadata else None
    return _build_dataarray_from_hao(
        hao,
        variable_name=canonical_name.strip(),
        var_type=var_type,
        source_file=source,
    )


def read_sl4_to_dataset(
    filepath: str,
    variable_names: Optional[List[str]] = None,
    *,
    include_file_metadata: bool = True,
) -> xr.Dataset:
    """
    Read an SL4 file into an xarray Dataset.

    Parameters
    ----------
    filepath : str
        Path to SL4 file.
    variable_names : list of str, optional
        Specific variables to extract. If None, all variables are read.
        Passed through as HARPY's ``extractList`` so unwanted variables
        aren't decoded.
    include_file_metadata : bool, default True
        Include source file path in Dataset.attrs.

    Returns
    -------
    xr.Dataset
        Dataset with one data variable per SL4 variable, sharing
        coordinates from the file's set definitions.
    """
    _check_file(filepath)
    sl4 = SL4(filepath, extractList=variable_names)

    sets = _sets_from_sl4(sl4)

    data_vars: Dict[str, xr.DataArray] = {}
    for var_name in sl4.variableNames:
        canonical = var_name.strip()
        try:
            da = read_sl4_to_dataarray(
                filepath,
                canonical,
                sl4_obj=sl4,
                include_file_metadata=False,
            )
        except Exception as exc:
            warnings.warn(
                f"Could not read variable '{canonical}': {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue
        data_vars[canonical] = da

    ds_attrs: Dict = {"source_format": "SL4"}
    if include_file_metadata:
        ds_attrs["source_file"] = str(Path(filepath).resolve())

    return xr.Dataset(data_vars, coords=sets, attrs=ds_attrs)


def list_sl4_variables(
    filepath: str,
    *,
    include_metadata: bool = True,
) -> Union[List[str], List[Dict]]:
    """
    List variables in an SL4 file.

    Parameters
    ----------
    filepath : str
        Path to SL4 file.
    include_metadata : bool, default True
        If True, return a dict per variable with name, type, long_name,
        shape and set names. If False, return bare variable names.

    Returns
    -------
    list
        List of variable names (str) or metadata dicts.
    """
    _check_file(filepath)
    sl4 = SL4(filepath)

    if not include_metadata:
        return [n.strip() for n in sl4.variableNames]

    result: List[Dict] = []
    for var_name in sl4.variableNames:
        hao = sl4.getVariable(var_name)
        var_type = _decode_var_type(sl4.varType(var_name))
        set_names = (
            [str(n).strip() for n in hao.setNames] if hao.array.ndim > 0 else []
        )
        result.append(
            {
                "variable_name": var_name.strip(),
                "var_type": var_type,
                "long_name": _strip(hao.long_name),
                "shape": tuple(hao.array.shape),
                "set_names": set_names,
            }
        )
    return result
