"""
Test suite for harpyx SL4 read functions.

Tests cover read of GEMPACK SL4 solution files as xarray DataArrays and
Datasets. Uses HARPY's bundled SJSUB.sl4 (Stylized Johansen with
subtotals) as the test fixture.

Test organisation:
- TestSL4Infrastructure : Verify test setup
- TestSL4SetReading     : Read sets from SL4
- TestSL4Variables      : Read individual variables as DataArrays
- TestSL4DimensionNaming: Duplicate dim handling and #RESULTS rename
- TestSL4DataTypes      : Float32 dtype, no integer support
- TestSL4VarType        : VCTP type code surfaced in attrs
- TestSL4Dataset        : Full Dataset operations
- TestSL4ListVariables  : list_sl4_variables function
- TestSL4ErrorHandling  : Error messages and exceptions
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import harpyx


# ====================================================================================
# Infrastructure
# ====================================================================================


class TestSL4Infrastructure:
    """Test that SL4 fixture and harpyx SL4 imports are working."""

    def test_sl4_test_file_exists(self, harpy_sl4_file):
        assert Path(harpy_sl4_file).exists()

    def test_harpyx_sl4_importable(self):
        assert hasattr(harpyx, "read_sl4_sets")
        assert hasattr(harpyx, "read_sl4_to_dataarray")
        assert hasattr(harpyx, "read_sl4_to_dataset")
        assert hasattr(harpyx, "list_sl4_variables")

    def test_sl4_class_loadable(self, harpy_sl4_file):
        from harpy.sl4 import SL4

        sl4 = SL4(harpy_sl4_file)
        assert len(sl4.variableNames) > 0


# ====================================================================================
# Set Reading
# ====================================================================================


@pytest.mark.unit
class TestSL4SetReading:
    """Test reading sets from SL4 files."""

    def test_read_user_sets(self, harpy_sl4_file):
        sets = harpyx.read_sl4_sets(harpy_sl4_file)
        assert "sect" in sets
        assert "fac" in sets
        assert "num_sect" in sets

    def test_results_set_present(self, harpy_sl4_file):
        """SJSUB has subtotals, so #RESULTS (renamed to RESULTS) is present."""
        sets = harpyx.read_sl4_sets(harpy_sl4_file)
        assert "RESULTS" in sets
        assert sets["RESULTS"].attrs["har_set_name"] == "#results"

    def test_set_elements_preserved(self, harpy_sl4_file):
        sets = harpyx.read_sl4_sets(harpy_sl4_file)

        sect = sets["sect"]
        assert len(sect) == 2
        assert "s1" in sect.values
        assert "s2" in sect.values

        fac = sets["fac"]
        assert len(fac) == 2
        assert "labor" in fac.values
        assert "capital" in fac.values

    def test_set_returns_dataarray(self, harpy_sl4_file):
        sets = harpyx.read_sl4_sets(harpy_sl4_file)
        for name, da in sets.items():
            assert isinstance(da, xr.DataArray)
            assert da.ndim == 1
            assert da.dims[0] == name

    def test_results_set_has_cumulative_first(self, harpy_sl4_file):
        sets = harpyx.read_sl4_sets(harpy_sl4_file)
        assert sets["RESULTS"].values[0] == "Cumulative"
        assert len(sets["RESULTS"]) == 3  # Cumulative + 2 subtotals


# ====================================================================================
# Variables
# ====================================================================================


@pytest.mark.unit
class TestSL4Variables:
    """Test reading individual SL4 variables as DataArrays."""

    def test_read_2d_variable(self, harpy_sl4_file):
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        assert da.ndim == 2
        assert da.shape == (2, 3)

    def test_variable_dims_lowercased(self, harpy_sl4_file):
        """User set names come back lowercase to match sl4.setNames."""
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        assert da.dims == ("sect", "RESULTS")

    def test_results_only_variable(self, harpy_sl4_file):
        """p_Y is a scalar variable; #RESULTS makes it 1-D."""
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_Y")
        assert da.dims == ("RESULTS",)
        assert da.shape == (3,)

    def test_variable_lookup_case_insensitive(self, harpy_sl4_file):
        da_lower = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_pc")
        da_upper = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "P_PC")
        np.testing.assert_array_equal(da_lower.values, da_upper.values)

    def test_variable_coords_populated(self, harpy_sl4_file):
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        assert list(da.coords["sect"].values) == ["s1", "s2"]
        assert da.coords["RESULTS"].values[0] == "Cumulative"

    def test_variable_attrs_carry_metadata(self, harpy_sl4_file):
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        assert da.attrs["source_format"] == "SL4"
        assert da.attrs["header_id"] == "p_PC"
        assert "long_name" in da.attrs
        assert "source_file" in da.attrs


# ====================================================================================
# Dimension Naming
# ====================================================================================


@pytest.mark.unit
class TestSL4DimensionNaming:
    """Test #RESULTS rename and duplicate dimension disambiguation."""

    def test_results_renamed_from_har_form(self, harpy_sl4_file):
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        assert "RESULTS" in da.dims
        assert "#RESULTS" not in da.dims

    def test_dimension_sets_attr_records_originals(self, harpy_sl4_file):
        """attrs['dimension_sets'] preserves HARPY's setNames verbatim."""
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        # SECT (uppercase from hao.setNames) and #RESULTS (literal HARPY form)
        assert da.attrs["dimension_sets"] == ["SECT", "#RESULTS"]

    def test_duplicate_dims_suffixed(self, harpy_sl4_file):
        """p_XC has SECT × SECT × #RESULTS → sect__1, sect__2, RESULTS."""
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_XC")
        assert da.dims == ("sect__1", "sect__2", "RESULTS")
        assert da.attrs["dimension_sets"] == ["SECT", "SECT", "#RESULTS"]

    def test_duplicate_coords_match(self, harpy_sl4_file):
        """sect__1 and sect__2 share the same coordinate values."""
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_XC")
        np.testing.assert_array_equal(
            da.coords["sect__1"].values, da.coords["sect__2"].values
        )


# ====================================================================================
# Data Types
# ====================================================================================


@pytest.mark.unit
class TestSL4DataTypes:
    """Test SL4 data type behaviour (always float32)."""

    def test_variable_is_float32(self, harpy_sl4_file):
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        assert da.dtype == np.float32

    def test_all_variables_float32(self, harpy_sl4_file):
        ds = harpyx.read_sl4_to_dataset(harpy_sl4_file)
        for name, da in ds.data_vars.items():
            assert da.dtype == np.float32, f"{name} is not float32"


# ====================================================================================
# var_type
# ====================================================================================


@pytest.mark.unit
class TestSL4VarType:
    """Test var_type attribute populated from VCTP."""

    def test_var_type_attr_set(self, harpy_sl4_file):
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        assert "var_type" in da.attrs

    def test_var_type_value_is_known_code(self, harpy_sl4_file):
        """SJSUB variables are all percent-change (type 'p')."""
        da = harpyx.read_sl4_to_dataarray(harpy_sl4_file, "p_PC")
        assert da.attrs["var_type"].lower() in {"c", "p", "l", "o"}


# ====================================================================================
# Dataset
# ====================================================================================


@pytest.mark.unit
class TestSL4Dataset:
    """Test reading a full SL4 file into a Dataset."""

    def test_read_full_file(self, harpy_sl4_file):
        ds = harpyx.read_sl4_to_dataset(harpy_sl4_file)
        assert len(ds.data_vars) == 11

    def test_shared_coords_merged(self, harpy_sl4_file):
        """sect coord is shared across variables that use it."""
        ds = harpyx.read_sl4_to_dataset(harpy_sl4_file)
        assert "sect" in ds.coords
        # Only one 'sect' coord — no SECT/sect duplication.
        assert "SECT" not in ds.coords

    def test_results_dim_in_dataset(self, harpy_sl4_file):
        ds = harpyx.read_sl4_to_dataset(harpy_sl4_file)
        assert "RESULTS" in ds.coords
        assert ds.sizes["RESULTS"] == 3

    def test_dataset_filter_by_variable_names(self, harpy_sl4_file):
        ds = harpyx.read_sl4_to_dataset(
            harpy_sl4_file, variable_names=["p_PC", "p_Y"]
        )
        assert set(ds.data_vars) == {"p_PC", "p_Y"}

    def test_dataset_carries_source_metadata(self, harpy_sl4_file):
        ds = harpyx.read_sl4_to_dataset(harpy_sl4_file)
        assert ds.attrs.get("source_format") == "SL4"
        assert "source_file" in ds.attrs


# ====================================================================================
# list_sl4_variables
# ====================================================================================


@pytest.mark.unit
class TestSL4ListVariables:
    """Test list_sl4_variables function."""

    def test_returns_metadata_dicts_by_default(self, harpy_sl4_file):
        metas = harpyx.list_sl4_variables(harpy_sl4_file)
        assert len(metas) == 11
        assert all(isinstance(m, dict) for m in metas)
        expected_keys = {"variable_name", "var_type", "long_name", "shape", "set_names"}
        assert all(expected_keys <= m.keys() for m in metas)

    def test_names_only_mode(self, harpy_sl4_file):
        names = harpyx.list_sl4_variables(harpy_sl4_file, include_metadata=False)
        assert all(isinstance(n, str) for n in names)
        assert "p_PC" in names

    def test_shapes_reported(self, harpy_sl4_file):
        metas = harpyx.list_sl4_variables(harpy_sl4_file)
        p_pc = next(m for m in metas if m["variable_name"] == "p_PC")
        assert p_pc["shape"] == (2, 3)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            harpyx.list_sl4_variables("nonexistent.sl4")


# ====================================================================================
# Error Handling
# ====================================================================================


@pytest.mark.unit
class TestSL4ErrorHandling:
    """Test error messages and exception handling."""

    def test_file_not_found_sets(self):
        with pytest.raises(FileNotFoundError):
            harpyx.read_sl4_sets("nonexistent.sl4")

    def test_file_not_found_dataarray(self):
        with pytest.raises(FileNotFoundError):
            harpyx.read_sl4_to_dataarray("nonexistent.sl4", "p_PC")

    def test_unknown_variable_raises(self, harpy_sl4_file):
        with pytest.raises(KeyError, match="BADVAR"):
            harpyx.read_sl4_to_dataarray(harpy_sl4_file, "BADVAR")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
