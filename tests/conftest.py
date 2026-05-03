"""
pytest configuration and fixtures for harpyx tests.

Uses:
1. HARPY's existing test files (via junction/symlink in testdata/harpy/)
2. Custom harpyx test files (created by create_test_data.py)
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

TEST_DIR = Path(__file__).parent
TESTDATA_DIR = TEST_DIR / "testdata"
HARPY_TESTDATA_DIR = TESTDATA_DIR / "harpy"


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to harpyx test data directory."""
    return TESTDATA_DIR


@pytest.fixture(scope="session")
def harpy_test_file():
    """
    Path to HARPY's test.har file.

    Contains:
    - ARR7: 7D array with duplicate dimensions
    - NH01: 2D labeled array
    - INTA: 2D integer array
    """
    filepath = HARPY_TESTDATA_DIR / "test.har"
    if not filepath.exists():
        pytest.skip(
            f"HARPY test data not found: {filepath}\n"
            r"Create junction: mklink /J tests\testdata\harpy C:\Users\e5106648\Code\HARPY\harpy\tests\testdata"
        )
    return str(filepath)


@pytest.fixture(scope="session")
def harpy_sl4_file():
    """
    Path to HARPY's SJSUB.sl4 test file (Stylized Johansen with subtotals).

    Contains:
    - User sets: sect (2), fac (2), num_sect (1)
    - Synthetic #RESULTS set (3 elements: Cumulative + 2 subtotals)
    - 11 variables, all type 'p' (% change), including:
        - p_Y: 1-D (RESULTS only) — was a scalar before #RESULTS was appended
        - p_PC: 2-D (sect, RESULTS)
        - p_XC, p_DVCOMIN: 3-D with duplicate SECT dim (sect__1, sect__2, RESULTS)
    """
    filepath = HARPY_TESTDATA_DIR / "SJSUB.sl4"
    if not filepath.exists():
        pytest.skip(
            f"HARPY SL4 test data not found: {filepath}\n"
            r"Create junction: mklink /J tests\testdata\harpy C:\Users\e5106648\Code\HARPY\harpy\tests\testdata"
        )
    return str(filepath)


@pytest.fixture(scope="session")
def harpy_sets_file():
    """
    Path to HARPY's setsnew7.har with many real-world sets.

    Contains 60+ headers including many set definitions.
    """
    filepath = HARPY_TESTDATA_DIR / "setsnew7.har"
    if not filepath.exists():
        pytest.skip(
            f"HARPY test data not found: {filepath}\n"
            r"Create junction: mklink /J tests\testdata\harpy C:\Users\e5106648\Code\HARPY\harpy\tests\testdata"
        )
    return str(filepath)


@pytest.fixture(scope="session")
def harpyx_test_file(test_data_dir):
    """
    Path to custom harpyx test HAR file.

    Created by create_test_data.py and contains:
    - Sets: COM (3 elements), REG (4 elements), IND (5 elements)
    - PROD: 2D labeled float (COM × IND)
    - CONS: 2D labeled float (COM × REG)
    - TRAD: 3D with duplicate dim (COM × REG × REG)
    - GDP:  scalar
    - CNTS: 2D labeled integer (COM × REG)
    - LABR: 2D labeled float (IND × REG)
    """
    filepath = test_data_dir / "test_harpyx.har"
    if not filepath.exists():
        pytest.skip(
            f"harpyx test file not found: {filepath}\nRun: python tests/create_test_data.py"
        )
    return str(filepath)


@pytest.fixture(scope="session")
def expected_values(test_data_dir):
    """Load expected values for test validation."""
    expected_file = test_data_dir / "expected.json"
    if not expected_file.exists():
        return {}
    with open(expected_file) as f:
        return json.load(f)


@pytest.fixture
def simple_dataarray():
    """2D labeled float DataArray (COM × REG, shape 3×4)."""
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    return xr.DataArray(
        data,
        dims=["COM", "REG"],
        coords={
            "COM": ["Agriculture", "Manufacture", "Services"],
            "REG": ["USA", "EU", "China", "Japan"],
        },
        name="Test production data",
        attrs={
            "header_id": "PROD",
            "long_name": "Production by commodity and region",
            "coefficient_name": "XPROD",
        },
    )


@pytest.fixture
def duplicate_dim_dataarray():
    """
    3D DataArray with duplicate dimension names (bilateral trade flows).

    Uses __1/__2 suffix convention: dims = (COM, REG__1, REG__2).
    The original HAR set names ['COM', 'REG', 'REG'] are stored in
    attrs['dimension_sets'].
    """
    data = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    return xr.DataArray(
        data,
        dims=["COM", "REG__1", "REG__2"],
        coords={
            "COM": ["Agriculture", "Manufacture", "Services"],
            "REG__1": ["USA", "EU", "China", "Japan"],
            "REG__2": ["USA", "EU", "China", "Japan"],
        },
        name="Bilateral trade flows",
        attrs={
            "header_id": "TRAD",
            "long_name": "Bilateral exports from origin to destination region",
            "coefficient_name": "XTRAD",
            "dimension_sets": ["COM", "REG", "REG"],
        },
    )


@pytest.fixture
def scalar_dataarray():
    """0-dimensional scalar DataArray."""
    return xr.DataArray(
        1234.5,
        attrs={
            "header_id": "GDP",
            "long_name": "Total GDP across all regions",
            "coefficient_name": "TOTGDP",
        },
    )


@pytest.fixture
def integer_dataarray():
    """2D integer DataArray for 2I type testing (COM × REG, shape 3×4)."""
    data = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.int32)
    return xr.DataArray(
        data,
        dims=["COM", "REG"],
        coords={
            "COM": ["Agriculture", "Manufacture", "Services"],
            "REG": ["USA", "EU", "China", "Japan"],
        },
        name="Count data",
        attrs={
            "header_id": "CNTS",
            "long_name": "Count data by commodity and region",
            "coefficient_name": "XCOUNTS",
        },
    )


@pytest.fixture
def unlabeled_dataarray():
    """2D DataArray without coordinates — should fail validation."""
    data = np.random.rand(3, 4).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["dim_0", "dim_1"],
        attrs={"header_id": "UNLB", "long_name": "Unlabeled test array"},
    )


@pytest.fixture
def sample_sets():
    """Dict mapping set names to element lists."""
    return {
        "COM": ["Agriculture", "Manufacture", "Services"],
        "REG": ["USA", "EU", "China", "Japan"],
        "IND": ["Industry1", "Industry2", "Industry3", "Industry4", "Industry5"],
    }


@pytest.fixture
def sample_dataset(simple_dataarray, duplicate_dim_dataarray, scalar_dataarray):
    """xarray Dataset with 3 data variables."""
    trade_data = duplicate_dim_dataarray.values
    reg_values = simple_dataarray.coords["REG"].values
    return xr.Dataset(
        {
            "PROD": simple_dataarray,
            "GDP": scalar_dataarray,
            "TRAD": (
                ["COM", "REG__1", "REG__2"],
                trade_data,
                {
                    "header_id": "TRAD",
                    "long_name": "Bilateral exports from origin to destination",
                    "dimension_sets": ["COM", "REG", "REG"],
                },
            ),
        },
        coords={
            "COM": simple_dataarray.coords["COM"],
            "REG": simple_dataarray.coords["REG"],
            # REG__1/REG__2 must use their own dim name so xarray attaches them
            # to the TRAD variable's dimensions correctly
            "REG__1": xr.DataArray(reg_values, dims=["REG__1"]),
            "REG__2": xr.DataArray(reg_values, dims=["REG__2"]),
        },
    )


def assert_dataarray_equal(da1: xr.DataArray, da2: xr.DataArray, check_attrs: bool = True) -> None:
    """Assert two DataArrays are equal in shape, dims, coords, values, and attrs."""
    assert da1.shape == da2.shape, f"Shapes differ: {da1.shape} vs {da2.shape}"
    assert da1.dims == da2.dims, f"Dimensions differ: {da1.dims} vs {da2.dims}"

    for dim in da1.dims:
        assert dim in da2.coords, f"Dimension {dim} missing in da2"
        np.testing.assert_array_equal(
            da1.coords[dim].values,
            da2.coords[dim].values,
            err_msg=f"Coordinates for {dim} differ",
        )

    np.testing.assert_allclose(da1.values, da2.values, rtol=1e-5)

    if check_attrs:
        for key in ["header_id", "long_name", "coefficient_name"]:
            if key in da1.attrs:
                assert key in da2.attrs, f"Attribute {key} missing in da2"
                assert da1.attrs[key] == da2.attrs[key], (
                    f"Attribute {key} differs: {da1.attrs[key]} vs {da2.attrs[key]}"
                )
