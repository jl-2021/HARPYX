# HARPYX

xarray integration layer for HARPY HAR and SL4 files.

## Purpose

Wraps HARPY's binary formats with xarray DataArray/Dataset, giving users
labeled, self-describing arrays instead of raw numpy arrays. Primary use
case is CGE modelling workflows (GEMPACK/ORANI-style models).

- **HAR** files — read and write.
- **SL4** (GEMPACK solution) files — read only (HARPY does not write SL4).

## Key design decisions

- **Duplicate dimension names** (e.g. bilateral trade: COM × REG × REG) are
  disambiguated with `__1`, `__2` suffixes in xarray dims. The original set
  names are preserved in `attrs['dimension_sets']`.
- **HAR set identification**: 1C headers whose `long_name` starts with
  `"Set "` are treated as coordinate sets; the set name is extracted as the
  second word.
- **HAR type support**: RE (float) and 2I (integer) labeled arrays only. RL
  and 2R (unlabeled) are rejected by default.
- **SL4 `#RESULTS` axis**: when subtotals are present, HARPY appends a
  `#RESULTS` dimension to every variable. harpyx renames it to `RESULTS`
  (the original is preserved in `attrs['dimension_sets']`).
- **SL4 dim case**: HARPY's `SL4` decoder lowercases user set names but
  variables' `setNames` keep the file's case — harpyx normalises everything
  to lowercase so coords merge across variables in a Dataset.

## Package structure

```
harpyx/
    __init__.py     public API (HAR + SL4 functions)
    io.py           read_har_*, write_*, list_har_headers
    sl4.py          read_sl4_*, list_sl4_variables (read only)
    validation.py   validate_har_metadata
tests/
    conftest.py     pytest fixtures (incl. harpy_sl4_file → SJSUB.sl4)
    test_io.py      HAR tests
    test_sl4.py     SL4 tests (34 tests, all passing)
    create_test_data.py  generates tests/testdata/test_harpyx.har
    testdata/
        expected.json        known values for validation
        test_harpyx.har      generated — not committed
        harpy/               junction to HARPY test data — not committed
                             (provides SJSUB.sl4 for SL4 tests)
```

## Dependencies

- **xarray** and **numpy** — standard install via pip.
- **harpy3** — install from local source (not on PyPI):
  ```
  .venv\Scripts\pip install -e C:\Users\e5106648\Code\HARPY
  ```

## Development setup

```bash
# 1. Activate venv
.venv\Scripts\activate

# 2. Install HARPY (package name harpy3, imports as harpy)
pip install -e C:\Users\e5106648\Code\HARPY

# 3. Install harpyx in editable mode
pip install -e .

# 4. Link HARPY test data (Windows — run as Administrator or Developer Mode on)
cd tests\testdata
mklink /J harpy C:\Users\e5106648\Code\HARPY\harpy\tests\testdata
cd ..\..

# 5. Generate harpyx HAR test data (SL4 tests use HARPY's bundled SJSUB.sl4)
python tests/create_test_data.py

# 6. Run all tests
pytest tests/ -v
```

## Implementation status

All HAR and SL4 read/write public functions are implemented and tested.

- HAR: `read_har_sets`, `read_har_to_dataarray`, `read_har_to_dataset`,
  `write_dataarray_to_har`, `write_dataset_to_har`, `write_sets_to_har`,
  `list_har_headers`, `validate_har_metadata`.
- SL4 (read only): `read_sl4_sets`, `read_sl4_to_dataarray`,
  `read_sl4_to_dataset`, `list_sl4_variables`.

## HARPY API reference

Key classes and methods used in the implementation:

```python
from harpy import HarFileObj
from harpy.har_file_io import HarFileIO, HarFileInfoObj
from harpy.header_array import HeaderArrayObj
from harpy._header_sets import _HeaderDims, _HeaderSet
from harpy.sl4 import SL4

# --- HAR ---
hf = HarFileObj('model.har')           # open/create
hf.getHeaderArrayNames()               # list all header IDs
hao = hf['PROD']                       # get HeaderArrayObj
hao.array                              # numpy array
hao.setNames                           # list of set name strings
hao.setElements                        # list of element lists (per dim)
hao.long_name                          # description string
hao.coeff_name                         # coefficient name string
hao.data_type                          # 'RE', '2I', 'RL', '2R', '1C'

HeaderArrayObj.SetHeaderFromData(...)  # create 1C set header
HeaderArrayObj.HeaderArrayFromData(...)  # create data header
hf.writeToDisk('output.har')           # write file

# --- SL4 (read only) ---
sl4 = SL4('solution.sl4', extractList=None)  # decodes at construction
sl4.variableNames                       # list of variable names
sl4.setNames                            # list of set names (lowercased)
sl4.getVariable(name) -> HeaderArrayObj # single variable
sl4.getSet(name)      -> HeaderArrayObj # single set
sl4.varType(name)                       # 'c' / 'p' / 'l' / 'o' (VCTP code)
# Quirk: extractList=[] is treated as None (loads all). Pass a sentinel
# name like ['__skip__'] to suppress variable decoding when only sets are
# needed.
```
