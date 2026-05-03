"""
Core I/O functions for reading and writing HAR files as xarray objects.

This module provides the main interface for converting between HARPY's HAR format
and xarray DataArrays/Datasets.
"""

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import xarray as xr

from harpy import HarFileObj
from harpy.har_file_io import HarFileIO
from harpy.header_array import HeaderArrayObj

from .validation import validate_har_metadata

__all__ = [
    "read_har_sets",
    "read_har_to_dataarray",
    "read_har_to_dataset",
    "write_sets_to_har",
    "write_dataarray_to_har",
    "write_dataset_to_har",
    "list_har_headers",
]

_UNLABELED_TYPES = frozenset({"RL", "2R"})
_LABELED_DATA_TYPES = frozenset({"RE", "2I"})
_MAX_LONG_NAME = 70


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_file(filepath: str) -> None:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"HAR file not found: {filepath}")


def _strip(s: Optional[str]) -> str:
    return (s or "").strip()


def _pad_long_name(s: str) -> str:
    return s[:_MAX_LONG_NAME].ljust(_MAX_LONG_NAME)


def _make_xr_dims(raw_names: List[Optional[str]]) -> tuple:
    """Convert raw HARPY set names to xarray dim names, disambiguating duplicates.

    Returns (xr_dims, original_names).
    xr_dims:        list[str] — names safe for xarray (no duplicates)
    original_names: list[str|None] — original HARPY set names
    """
    name_counts = Counter(n for n in raw_names if n is not None)
    dim_counters: Dict[str, int] = {}
    xr_dims: List[str] = []
    for i, name in enumerate(raw_names):
        if name is None:
            xr_dims.append(f"dim_{i}")
        elif name_counts[name] > 1:
            dim_counters[name] = dim_counters.get(name, 0) + 1
            xr_dims.append(f"{name}__{dim_counters[name]}")
        else:
            xr_dims.append(name)
    return xr_dims, list(raw_names)


def _har_names_from_xr_dims(da: xr.DataArray) -> List[str]:
    """Recover original HARPY set names from a DataArray.

    Uses attrs['dimension_sets'] if present; otherwise strips __N suffixes.
    """
    if "dimension_sets" in da.attrs:
        return list(da.attrs["dimension_sets"])
    return [re.sub(r"__\d+$", "", d) for d in da.dims]


def _build_hao(
    da: xr.DataArray,
    header_id: str,
    long_name: str,
    coefficient_name: str,
) -> HeaderArrayObj:
    """Build a HeaderArrayObj from an xarray DataArray."""
    ln = _pad_long_name(long_name or header_id)
    cn = (coefficient_name or header_id)[:12].ljust(12)

    if da.ndim == 0:
        arr = np.asarray(da.values, dtype=np.float32)
        return HeaderArrayObj.HeaderArrayFromData(
            array=arr, coeff_name=cn, long_name=ln, sets=None, setElDict=None
        )

    har_set_names = _har_names_from_xr_dims(da)

    # Build element dict — duplicate set names share the same entry
    set_el_dict: Dict[str, List[str]] = {}
    for xr_dim, har_name in zip(da.dims, har_set_names):
        if xr_dim in da.coords:
            set_el_dict[har_name] = [str(e).strip() for e in da.coords[xr_dim].values]

    if np.issubdtype(da.dtype, np.integer):
        arr = da.values.astype(np.int32)
    else:
        arr = da.values.astype(np.float32)

    return HeaderArrayObj.HeaderArrayFromData(
        array=arr,
        coeff_name=cn,
        long_name=ln,
        sets=har_set_names,
        setElDict=set_el_dict,
    )


def _supplement_dims_from_sets(
    raw_names: List[Optional[str]],
    raw_elements: List[Optional[List[str]]],
    shape: tuple,
    sets: Dict[str, xr.DataArray],
) -> tuple:
    """Try to fill None dim names using pre-loaded set DataArrays matched by size.

    Only fills where the match is unambiguous (exactly one set of that size).
    """
    sizes_to_sets: Dict[int, List[str]] = {}
    for set_name, set_da in sets.items():
        sizes_to_sets.setdefault(len(set_da), []).append(set_name)

    names = list(raw_names)
    elements = list(raw_elements)
    for i, (name, elems) in enumerate(zip(names, elements)):
        if name is None:
            candidates = sizes_to_sets.get(shape[i], [])
            if len(candidates) == 1:
                set_name = candidates[0]
                names[i] = set_name
                elements[i] = sets[set_name].values.tolist()
    return names, elements


def _default_set_header_id(set_name: str) -> str:
    """Generate a default 4-char header ID for a set: 'S' + first 3 chars."""
    return ("S" + set_name[:3]).upper()


def _resolve_header_id(da: xr.DataArray, header_id: Optional[str]) -> str:
    hid = header_id or da.attrs.get("header_id")
    if not hid:
        raise ValueError(
            "No header_id provided and 'header_id' not found in DataArray.attrs. "
            "Pass header_id= or set da.attrs['header_id']."
        )
    return str(hid).upper()[:4]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def read_har_sets(
    filepath: str,
    set_ids: Optional[List[str]] = None,
) -> Dict[str, xr.DataArray]:
    """
    Read set headers (1C type) from a HAR file.

    Identifies sets by long_name starting with "Set ".

    Parameters
    ----------
    filepath : str
        Path to HAR file.
    set_ids : list of str, optional
        Specific header IDs to read. If None, read all 1C headers
        that are identified as sets.

    Returns
    -------
    dict
        Dictionary mapping set names to 1-D coordinate DataArrays.

    Examples
    --------
    >>> import harpyx
    >>> sets = harpyx.read_har_sets('model.har')
    >>> sets.keys()
    dict_keys(['COM', 'REG', 'IND'])
    >>> sets['REG']
    <xarray.DataArray 'REG' (REG: 4)>
    array(['USA', 'EU', 'China', 'Japan'], dtype='<U12')
    """
    _check_file(filepath)
    hfi = HarFileIO.readHarFileInfo(filepath)
    upper_ids = {s.strip().upper() for s in set_ids} if set_ids is not None else None

    result: Dict[str, xr.DataArray] = {}
    for ha_name in hfi.getHeaderArrayNames():
        if upper_ids is not None and ha_name not in upper_ids:
            continue
        ha_info = hfi.getHeaderArrayInfo(ha_name)
        if ha_info.data_type != "1C":
            continue
        long_name = _strip(ha_info.long_name)
        if not long_name.startswith("Set "):
            continue

        set_name = long_name.split()[1]
        hao = HarFileIO.readHeader(hfi, ha_name)
        elements = [e.strip() for e in hao.array.tolist()]

        result[set_name] = xr.DataArray(
            np.array(elements),
            dims=[set_name],
            coords={set_name: elements},
            name=set_name,
            attrs={"header_id": ha_name, "long_name": long_name},
        )
    return result


def read_har_to_dataarray(
    filepath: str,
    header_id: str,
    *,
    sets: Optional[Dict] = None,
    reject_unlabeled: bool = True,
    include_file_metadata: bool = True,
) -> xr.DataArray:
    """
    Read a single HAR header into an xarray DataArray.

    Supports: RE (labeled real), 2I (labeled int), scalars.
    Rejects: RL, 2R (unlabeled arrays) when reject_unlabeled=True.

    Parameters
    ----------
    filepath : str
        Path to HAR file.
    header_id : str
        4-character header identifier (case-insensitive).
    sets : dict, optional
        Pre-loaded sets from read_har_sets(). Used to supplement missing
        coordinate info for 2I headers, which lose set labels on write/read.
    reject_unlabeled : bool, default True
        If True, raise TypeError for RL/2R types.
    include_file_metadata : bool, default True
        If True, include source file metadata in attrs.

    Returns
    -------
    xr.DataArray
        Array with labeled dimensions and coordinates.

    Examples
    --------
    >>> import harpyx
    >>> da = harpyx.read_har_to_dataarray('model.har', 'PROD')
    >>> da.dims
    ('COM', 'IND')
    >>> da.coords['COM']
    <xarray.DataArray 'COM' (COM: 3)>
    array(['Agriculture', 'Manufacturing', 'Services'])
    """
    _check_file(filepath)
    hfi = HarFileIO.readHarFileInfo(filepath)
    ha_name = header_id.strip().upper()

    if ha_name not in hfi:
        raise KeyError(f"Header '{header_id}' not found in {filepath}")

    ha_info = hfi.getHeaderArrayInfo(ha_name)
    data_type = ha_info.data_type

    if data_type == "1C":
        raise TypeError(
            f"Header '{ha_name}' is a set header (1C type). Use read_har_sets() instead."
        )

    if reject_unlabeled and data_type in _UNLABELED_TYPES:
        raise TypeError(
            f"Header '{ha_name}' has type '{data_type}' (unlabeled array). "
            "Only labeled types (RE, 2I) are supported. "
            "Pass reject_unlabeled=False to suppress this error."
        )

    hao = HarFileIO.readHeader(hfi, ha_name)

    # Build base attrs
    attrs: Dict = {"header_id": ha_name, "har_type": data_type}
    long_name = _strip(ha_info.long_name)
    if long_name:
        attrs["long_name"] = long_name
    coeff_name = _strip(getattr(ha_info, "coeff_name", None))
    if coeff_name:
        attrs["coefficient_name"] = coeff_name
    if include_file_metadata:
        attrs["source_file"] = str(Path(filepath).resolve())

    # Scalar: shape is () after reading
    if hao.array.ndim == 0:
        return xr.DataArray(hao.array, attrs=attrs)

    raw_names: List[Optional[str]] = hao.setNames
    raw_elements: List[Optional[List[str]]] = hao.setElements

    # Supplement unlabeled dims (2I case) from pre-loaded sets
    if sets is not None and any(n is None for n in raw_names):
        raw_names, raw_elements = _supplement_dims_from_sets(
            raw_names, raw_elements, hao.array.shape, sets
        )

    xr_dims, original_names = _make_xr_dims(raw_names)

    coords: Dict = {}
    for xr_dim, elements in zip(xr_dims, raw_elements):
        if elements is not None:
            coords[xr_dim] = [e.strip() for e in elements]

    # Record original HAR set names when dimension disambiguation occurred
    if xr_dims != original_names:
        attrs["dimension_sets"] = original_names

    return xr.DataArray(hao.array, dims=xr_dims, coords=coords, attrs=attrs)


def read_har_to_dataset(
    filepath: str,
    header_ids: Optional[List[str]] = None,
    *,
    reject_unlabeled: bool = True,
    include_file_metadata: bool = True,
) -> xr.Dataset:
    """
    Read a HAR file into an xarray Dataset.

    Parameters
    ----------
    filepath : str
        Path to HAR file.
    header_ids : list of str, optional
        Specific data header IDs to read. If None, reads all supported headers.
    reject_unlabeled : bool, default True
        Skip RL/2R types with a warning instead of raising.
    include_file_metadata : bool, default True
        Include file provenance in Dataset.attrs.

    Returns
    -------
    xr.Dataset
        Dataset with coords and data variables.

    Examples
    --------
    >>> import harpyx
    >>> ds = harpyx.read_har_to_dataset('model.har')
    >>> ds
    <xarray.Dataset>
    Dimensions:  (COM: 3, IND: 5, REG: 4)
    Coordinates:
      * COM      (COM) <U12 'Agriculture' 'Manufacturing' 'Services'
      * IND      (IND) <U12 ...
      * REG      (REG) <U12 'USA' 'EU' 'China' 'Japan'
    Data variables:
        PROD     (COM, IND) float32 ...
        CONS     (COM, REG) float32 ...
    """
    _check_file(filepath)
    hfi = HarFileIO.readHarFileInfo(filepath)

    # Load all set headers first for coordinate supplementation
    sets = read_har_sets(filepath)

    # Determine which data headers to read
    if header_ids is not None:
        target_ids = [h.strip().upper() for h in header_ids]
    else:
        target_ids = []
        for ha_name in hfi.getHeaderArrayNames():
            ha_info = hfi.getHeaderArrayInfo(ha_name)
            dt = ha_info.data_type
            if dt == "1C":
                continue  # sets are handled separately
            if dt in _UNLABELED_TYPES:
                if reject_unlabeled:
                    warnings.warn(
                        f"Skipping header '{ha_name}' (type '{dt}', unlabeled). "
                        "Pass reject_unlabeled=False to include it.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
            target_ids.append(ha_name)

    data_vars: Dict[str, xr.DataArray] = {}
    for ha_name in target_ids:
        if ha_name not in hfi:
            raise KeyError(f"Header '{ha_name}' not found in {filepath}")
        ha_info = hfi.getHeaderArrayInfo(ha_name)
        dt = ha_info.data_type
        if dt in _UNLABELED_TYPES and reject_unlabeled:
            warnings.warn(
                f"Skipping header '{ha_name}' (type '{dt}', unlabeled).",
                UserWarning,
                stacklevel=2,
            )
            continue
        try:
            da = read_har_to_dataarray(
                filepath,
                ha_name,
                sets=sets,
                reject_unlabeled=reject_unlabeled,
                include_file_metadata=False,
            )
            data_vars[ha_name] = da
        except Exception as exc:
            warnings.warn(
                f"Could not read header '{ha_name}': {exc}",
                UserWarning,
                stacklevel=2,
            )

    # Merge shared coordinates across variables
    shared_coords: Dict[str, xr.DataArray] = {}
    for set_name, set_da in sets.items():
        shared_coords[set_name] = set_da

    ds_attrs: Dict = {}
    if include_file_metadata:
        ds_attrs["source_file"] = str(Path(filepath).resolve())

    return xr.Dataset(data_vars, coords=shared_coords, attrs=ds_attrs)


def write_sets_to_har(
    filepath: str,
    sets: Dict[str, Union[List, np.ndarray]],
    *,
    mode: Literal["w", "a"] = "w",
    header_id_map: Optional[Dict[str, str]] = None,
    long_name_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write coordinate sets to a HAR file as 1C headers.

    Parameters
    ----------
    filepath : str
        Path to output HAR file.
    sets : dict
        Mapping from set name to element array/list.
    mode : {'w', 'a'}, default 'w'
        'w' = create/overwrite, 'a' = append to existing file.
    header_id_map : dict, optional
        Mapping from set name to header ID. Default: 'S' + first 3 chars of name.
    long_name_map : dict, optional
        Mapping from set name to description text passed to SetHeaderFromData.
        (SetHeaderFromData auto-prepends "Set {name} " to whatever is passed.)

    Examples
    --------
    >>> import harpyx
    >>> sets = {
    ...     'COM': ['Agriculture', 'Manufacturing', 'Services'],
    ...     'REG': ['USA', 'EU', 'China', 'Japan'],
    ... }
    >>> harpyx.write_sets_to_har('output.har', sets)
    """
    hnames: List[str] = []
    haos: List[HeaderArrayObj] = []

    for set_name, elements in sets.items():
        if isinstance(elements, xr.DataArray):
            elems = [str(e).strip() for e in elements.values.tolist()]
        elif isinstance(elements, np.ndarray):
            elems = [str(e).strip() for e in elements.tolist()]
        else:
            elems = [str(e).strip() for e in elements]

        header_id = (header_id_map or {}).get(set_name, _default_set_header_id(set_name))
        desc = (long_name_map or {}).get(set_name, "")

        hao = HeaderArrayObj.SetHeaderFromData(
            setName=set_name,
            setElements=elems,
            long_name=desc,
        )
        hnames.append(header_id.upper()[:4])
        haos.append(hao)

    _write_to_file(filepath, hnames, haos, mode)


def write_dataarray_to_har(
    dataarray: xr.DataArray,
    filepath: str,
    header_id: Optional[str] = None,
    *,
    mode: Literal["w", "a"] = "w",
    long_name: Optional[str] = None,
    coefficient_name: Optional[str] = None,
    write_sets: bool = True,
    set_header_ids: Optional[Dict[str, str]] = None,
    validate_metadata: bool = True,
) -> None:
    """
    Write an xarray DataArray to a HAR file.

    Parameters
    ----------
    dataarray : xr.DataArray
        Array to write. Must have coordinates for all dimensions.
    filepath : str
        Path to output HAR file.
    header_id : str, optional
        4-character header ID. Falls back to dataarray.attrs['header_id'].
    mode : {'w', 'a'}, default 'w'
        File write mode.
    long_name : str, optional
        Long name for the header. Falls back to dataarray.attrs['long_name'].
    coefficient_name : str, optional
        Coefficient name. Falls back to dataarray.attrs['coefficient_name'].
    write_sets : bool, default True
        If True, write coordinate sets as 1C headers before the data header.
    set_header_ids : dict, optional
        Mapping from dimension name to header ID for set headers.
    validate_metadata : bool, default True
        Validate before writing.

    Examples
    --------
    >>> import harpyx
    >>> import xarray as xr
    >>> import numpy as np
    >>>
    >>> da = xr.DataArray(
    ...     np.random.rand(3, 4),
    ...     dims=['COM', 'REG'],
    ...     coords={
    ...         'COM': ['Agriculture', 'Manufacturing', 'Services'],
    ...         'REG': ['USA', 'EU', 'China', 'Japan'],
    ...     },
    ...     attrs={'header_id': 'PROD'},
    ... )
    >>> harpyx.write_dataarray_to_har(da, 'output.har')
    """
    if validate_metadata:
        errors = validate_har_metadata(dataarray)
        if errors:
            raise ValueError(
                f"DataArray failed metadata validation:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    hid = _resolve_header_id(dataarray, header_id)
    ln = long_name or dataarray.attrs.get("long_name", hid) or hid
    cn = coefficient_name or dataarray.attrs.get("coefficient_name", hid) or hid

    hnames: List[str] = []
    haos: List[HeaderArrayObj] = []

    if write_sets and dataarray.ndim > 0:
        har_set_names = _har_names_from_xr_dims(dataarray)
        seen: set = set()
        for xr_dim, har_name in zip(dataarray.dims, har_set_names):
            if har_name in seen:
                continue
            seen.add(har_name)
            if xr_dim not in dataarray.coords:
                continue
            elements = [str(e).strip() for e in dataarray.coords[xr_dim].values]
            sid = (set_header_ids or {}).get(har_name, _default_set_header_id(har_name))
            set_hao = HeaderArrayObj.SetHeaderFromData(
                setName=har_name, setElements=elements, long_name=""
            )
            hnames.append(sid.upper()[:4])
            haos.append(set_hao)

    hao = _build_hao(dataarray, hid, ln, cn)
    hnames.append(hid)
    haos.append(hao)

    _write_to_file(filepath, hnames, haos, mode)


def write_dataset_to_har(
    dataset: xr.Dataset,
    filepath: str,
    *,
    mode: Literal["w", "a"] = "w",
    header_id_map: Optional[Dict[str, str]] = None,
    validate_metadata: bool = True,
) -> None:
    """
    Write an xarray Dataset to a HAR file.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to write.
    filepath : str
        Path to output HAR file.
    mode : {'w', 'a'}, default 'w'
        File write mode.
    header_id_map : dict, optional
        Mapping from variable name to header ID. Falls back to variable attrs.
    validate_metadata : bool, default True
        Validate all variables before writing.

    Examples
    --------
    >>> import harpyx
    >>> ds = harpyx.read_har_to_dataset('input.har')
    >>> # ... modify ds ...
    >>> harpyx.write_dataset_to_har(ds, 'output.har')
    """
    hnames: List[str] = []
    haos: List[HeaderArrayObj] = []
    seen_sets: set = set()

    for var_name, da in dataset.data_vars.items():
        if validate_metadata:
            errors = validate_har_metadata(da, require_header_id=False)
            coord_errors = [e for e in errors if "coordinate" in e.lower()]
            if coord_errors:
                raise ValueError(
                    f"Variable '{var_name}' failed metadata validation:\n"
                    + "\n".join(f"  - {e}" for e in coord_errors)
                )

        hid = (header_id_map or {}).get(var_name) or da.attrs.get("header_id") or var_name
        hid = str(hid).upper()[:4]
        ln = da.attrs.get("long_name", hid) or hid
        cn = da.attrs.get("coefficient_name", hid) or hid

        # Write set headers not yet written
        if da.ndim > 0:
            har_set_names = _har_names_from_xr_dims(da)
            for xr_dim, har_name in zip(da.dims, har_set_names):
                if har_name in seen_sets:
                    continue
                if xr_dim not in da.coords:
                    continue
                seen_sets.add(har_name)
                elements = [str(e).strip() for e in da.coords[xr_dim].values]
                sid = _default_set_header_id(har_name)
                set_hao = HeaderArrayObj.SetHeaderFromData(
                    setName=har_name, setElements=elements, long_name=""
                )
                hnames.append(sid)
                haos.append(set_hao)

        hao = _build_hao(da, hid, ln, cn)
        hnames.append(hid)
        haos.append(hao)

    _write_to_file(filepath, hnames, haos, mode)


def list_har_headers(
    filepath: str,
    *,
    include_metadata: bool = True,
    header_type: Optional[Literal["set", "data", "all"]] = "all",
) -> Union[List[str], List[Dict]]:
    """
    List headers in a HAR file.

    Parameters
    ----------
    filepath : str
        Path to HAR file.
    include_metadata : bool, default True
        If True, return detailed metadata dicts; otherwise return header IDs.
    header_type : {'set', 'data', 'all'}, default 'all'
        Filter by header type. 'set' returns only 1C headers, 'data' excludes them.

    Returns
    -------
    list
        List of header IDs (str) or metadata dicts.

    Examples
    --------
    >>> import harpyx
    >>> headers = harpyx.list_har_headers('model.har')
    >>> for h in headers:
    ...     print(f"{h['header_id']}: {h['long_name']}")
    """
    _check_file(filepath)
    hfi = HarFileIO.readHarFileInfo(filepath)
    result: List = []

    for ha_name in hfi.getHeaderArrayNames():
        ha_info = hfi.getHeaderArrayInfo(ha_name)
        data_type = ha_info.data_type
        is_set = data_type == "1C"

        if header_type == "set" and not is_set:
            continue
        if header_type == "data" and is_set:
            continue

        if include_metadata:
            shape: tuple = tuple(ha_info.file_dims) if ha_info.file_dims else ()
            result.append(
                {
                    "header_id": ha_name,
                    "data_type": data_type,
                    "long_name": _strip(ha_info.long_name),
                    "shape": shape,
                }
            )
        else:
            result.append(ha_name)

    return result


# ---------------------------------------------------------------------------
# Private write helper
# ---------------------------------------------------------------------------


def _write_to_file(
    filepath: str,
    hnames: List[str],
    haos: List[HeaderArrayObj],
    mode: str,
) -> None:
    """Write headers to filepath, respecting mode='w' (overwrite) or 'a' (append)."""
    if mode == "a" and Path(filepath).exists():
        # Load existing headers into memory, add new ones, write all out
        hf = HarFileObj(filepath)
        # Pre-load all existing headers so writeToDisk can write them
        for existing_name in list(hf.getHeaderArrayNames()):
            _ = hf[existing_name]
        for hname, hao in zip(hnames, haos):
            hf[hname] = hao
        hf.writeToDisk(filepath)
    else:
        HarFileIO.writeHeaders(filepath, hnames, haos)
