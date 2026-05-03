# harpyx

xarray integration for HARPY HAR and SL4 files. **Alpha (v0.1.0)**

Provides `xr.DataArray` and `xr.Dataset` interfaces for HARPY's binary
formats used in GEMPACK/ORANI-style CGE models:

- **HAR** (Header Array) files — read and write.
- **SL4** (GEMPACK solution) files — read only (HARPY does not write SL4).

## Installation

```bash
pip install git+https://github.com/jl-2021/HARPYX.git
```

This automatically installs [harpy3](https://github.com/GEMPACKsoftware/HARPY) from GitHub as well.

If you want to use a local HARPY clone (e.g. for development), install it first:

```bash
pip install -e /path/to/HARPY
pip install git+https://github.com/jl-2021/HARPYX.git
```

## Quick start

### HAR files

```python
import harpyx

# Read entire HAR file as a labeled Dataset
ds = harpyx.read_har_to_dataset("model.har")

# Read a single header
production = harpyx.read_har_to_dataarray("model.har", "PROD")
print(production.dims)   # ('COM', 'IND')

# Inspect file contents without loading data
for h in harpyx.list_har_headers("model.har"):
    print(h["header_id"], h["data_type"], h["shape"])

# Write back
harpyx.write_dataarray_to_har(production, "output.har")
harpyx.write_dataset_to_har(ds, "output.har")
```

### SL4 solution files

```python
import harpyx

# Read entire SL4 file
sol = harpyx.read_sl4_to_dataset("solution.sl4")

# Read a single variable
prices = harpyx.read_sl4_to_dataarray("solution.sl4", "p_PC")
print(prices.dims)              # ('sect', 'RESULTS')
print(prices.attrs["var_type"]) # 'p' (percent change)

# Read just the sets (cheap — skips variable decoding)
sets = harpyx.read_sl4_sets("solution.sl4")

# List variables with metadata
for v in harpyx.list_sl4_variables("solution.sl4"):
    print(v["variable_name"], v["var_type"], v["shape"])
```

## Duplicate dimensions

Both HAR and SL4 allow repeated set names (e.g. bilateral trade: COM × REG ×
REG). harpyx renames them with `__1`/`__2` suffixes so xarray can hold them:

```python
trade = harpyx.read_har_to_dataarray("model.har", "TRAD")
trade.dims          # ('COM', 'REG__1', 'REG__2')
trade.attrs["dimension_sets"]  # ['COM', 'REG', 'REG']  — original HAR names
```

The original names are restored automatically on HAR write.

## SL4 specifics

- **`#RESULTS` dimension**: when an SL4 file contains subtotals, HARPY appends
  a `#RESULTS` axis (elements: `Cumulative` + each subtotal) to every
  variable. harpyx renames this to `RESULTS`. The original `#RESULTS` is
  preserved in `attrs['dimension_sets']`.
- **Lowercase dim names**: HARPY's `SL4` decoder normalises user set names
  to lowercase internally. harpyx follows that convention so coordinates
  merge cleanly across variables in a Dataset.
- **`var_type`**: each DataArray carries `attrs['var_type']` from the SL4
  VCTP header (`'c'`, `'p'`, `'l'`, `'o'` — change, percent change, level,
  ordinary change).
- **Always float32**: HARPY decodes all SL4 variables as `float32`.

## Known limitations

- **SL4 is read-only**. HARPY does not support writing SL4 files.
- **2I integer headers lose coordinate labels** on HAR write/read (HARPY
  limitation). Coordinates can be re-attached by passing
  `sets=harpyx.read_har_sets(filepath)` to `read_har_to_dataarray`. A fix is
  planned in a future HARPY release.
- **RL headers** (unlabeled legacy real arrays) are read-only; HARPY cannot
  write them. harpyx rejects them by default (`reject_unlabeled=True`).
- **2R headers** (unlabeled 2D real arrays) are similarly rejected by
  default. Pass `reject_unlabeled=False` to read them as dimensionless
  DataArrays.

## Supported types

### HAR

| Type | Description | Read | Write |
|------|-------------|------|-------|
| RE   | Labeled real (up to 7D) | ✓ | ✓ |
| 2I   | 2D integer (labels lost on round-trip) | ✓ | ✓ |
| 1C   | Character set header | ✓ (as sets) | ✓ |
| RL   | Unlabeled real (legacy) | optional | ✗ |
| 2R   | Unlabeled 2D real | optional | ✗ |

### SL4

| Element | Read | Write |
|---------|------|-------|
| Variable arrays (always float32) | ✓ | ✗ |
| Set definitions | ✓ | ✗ |
| Synthetic `#RESULTS` axis (subtotals) | ✓ (renamed `RESULTS`) | ✗ |
| `var_type` (VCTP) | ✓ (in attrs) | ✗ |
