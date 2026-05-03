"""
Test suite for harpyx I/O functions.

Tests cover read/write of HAR files as xarray DataArrays and Datasets.

Test organisation:
- TestInfrastructure    : Verify test setup works
- TestSetReading        : Read 1C set headers
- TestDimensionNaming   : Handle duplicate dimension names
- TestScalarHandling    : 0-dimensional arrays
- TestDataTypes         : RE, 2I, RL, 2R type support
- TestWriting           : Write DataArrays to HAR
- TestRoundTrip         : Read → Write → Read preservation
- TestDataset           : Full Dataset operations
- TestValidation        : Metadata validation
- TestErrorHandling     : Error messages and exceptions
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from harpy import HarFileObj
from harpy.har_file_io import HarFileIO

import harpyx


# ====================================================================================
# Infrastructure
# ====================================================================================


class TestInfrastructure:
    """Test that the test infrastructure itself is working."""

    def test_fixture_data_loads(self, test_data_dir):
        assert test_data_dir.exists()
        assert test_data_dir.is_dir()

    def test_harpy_test_file_exists(self, harpy_test_file):
        assert Path(harpy_test_file).exists()

    def test_harpy_test_file_readable(self, harpy_test_file):
        hf = HarFileObj(harpy_test_file)
        headers = hf.getHeaderArrayNames()
        assert len(headers) > 0

    def test_xarray_importable(self):
        import xarray as xr

        assert hasattr(xr, "DataArray")
        assert hasattr(xr, "Dataset")

    def test_harpyx_importable(self):
        assert hasattr(harpyx, "read_har_sets")
        assert hasattr(harpyx, "read_har_to_dataarray")


# ====================================================================================
# Set Reading
# ====================================================================================


@pytest.mark.unit
class TestSetReading:
    """Test reading 1C set headers from HAR files."""

    def test_read_all_sets(self, harpyx_test_file):
        sets = harpyx.read_har_sets(harpyx_test_file)

        assert "COM" in sets
        assert "REG" in sets
        assert "IND" in sets
        assert len(sets) == 3

    def test_identify_sets_by_longname(self, harpyx_test_file):
        hfi = HarFileIO.readHarFileInfo(harpyx_test_file)
        for header_id, ha_info in hfi.items():
            if ha_info.data_type == "1C":
                if ha_info.long_name.strip().startswith("Set "):
                    sets = harpyx.read_har_sets(harpyx_test_file)
                    set_name = ha_info.long_name.strip().split()[1]
                    assert set_name in sets

    def test_extract_set_names(self, harpyx_test_file):
        sets = harpyx.read_har_sets(harpyx_test_file)
        assert "COM" in sets
        assert "REG" in sets
        assert "IND" in sets

    def test_set_elements_preserved(self, harpyx_test_file):
        sets = harpyx.read_har_sets(harpyx_test_file)

        com = sets["COM"]
        assert len(com) == 3
        assert "Agriculture" in com.values
        assert "Manufacture" in com.values
        assert "Services" in com.values

        reg = sets["REG"]
        assert len(reg) == 4
        assert "USA" in reg.values
        assert "Japan" in reg.values

    def test_set_returns_dataarray(self, harpyx_test_file):
        sets = harpyx.read_har_sets(harpyx_test_file)
        for set_name, set_da in sets.items():
            assert isinstance(set_da, xr.DataArray)
            assert set_da.ndim == 1
            assert set_da.dims[0] == set_name

    def test_read_sets_filtered_by_header_id(self, harpyx_test_file):
        """set_ids filters by HAR header ID (e.g. 'SCOM'), not set name."""
        sets = harpyx.read_har_sets(harpyx_test_file, set_ids=["SCOM"])
        assert "COM" in sets
        assert "REG" not in sets
        assert "IND" not in sets


# ====================================================================================
# Dimension Naming
# ====================================================================================


@pytest.mark.unit
class TestDimensionNaming:
    """Test handling of duplicate dimension names."""

    def test_unique_dims_unchanged(self, harpyx_test_file):
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "PROD")
        assert da.dims == ("COM", "IND")

    def test_duplicate_dims_numbered_all(self, harpyx_test_file):
        """Both occurrences of a duplicate dim get __1, __2 suffixes."""
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "TRAD")
        assert da.dims == ("COM", "REG__1", "REG__2")

    def test_triple_duplicate_numbered(self, harpyx_test_file):
        """Three occurrences → __1, __2, __3."""
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "TRIP")
        assert da.dims == ("REG__1", "REG__2", "REG__3")

    def test_mixed_unique_duplicate(self, harpyx_test_file):
        """COM, REG, REG → COM, REG__1, REG__2."""
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "TRAD")
        assert "COM" in da.dims
        assert "REG__1" in da.dims
        assert "REG__2" in da.dims
        assert "REG" not in da.dims

    def test_dimension_sets_attr_preserved(self, harpyx_test_file):
        """Original HAR set names stored in attrs['dimension_sets']."""
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "TRAD")
        assert "dimension_sets" in da.attrs
        assert da.attrs["dimension_sets"] == ["COM", "REG", "REG"]

    def test_coordinates_match_for_duplicates(self, harpyx_test_file):
        """REG__1 and REG__2 share the same coordinate values."""
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "TRAD")
        np.testing.assert_array_equal(
            da.coords["REG__1"].values,
            da.coords["REG__2"].values,
        )


# ====================================================================================
# Scalar Handling
# ====================================================================================


@pytest.mark.unit
class TestScalarHandling:
    """Test handling of 0-dimensional scalar arrays."""

    def test_read_scalar_as_0d(self, harpyx_test_file):
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "GDP")
        assert da.ndim == 0
        assert da.shape == ()

    def test_scalar_value_correct(self, harpyx_test_file):
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "GDP")
        assert abs(float(da.values) - 1234.5) < 0.01

    def test_scalar_has_no_dimensions(self, harpyx_test_file):
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "GDP")
        assert len(da.dims) == 0
        assert len(da.coords) == 0

    def test_write_scalar_to_har(self, tmp_path, scalar_dataarray):
        output_file = tmp_path / "scalar_test.har"
        harpyx.write_dataarray_to_har(scalar_dataarray, str(output_file))

        assert output_file.exists()
        hf = HarFileObj(str(output_file))
        hao = hf["GDP"]
        assert hao.array.size == 1

    @pytest.mark.roundtrip
    def test_scalar_roundtrip(self, tmp_path, harpyx_test_file):
        da_original = harpyx.read_har_to_dataarray(harpyx_test_file, "GDP")

        output_file = tmp_path / "scalar_roundtrip.har"
        harpyx.write_dataarray_to_har(da_original, str(output_file))

        da_roundtrip = harpyx.read_har_to_dataarray(str(output_file), "GDP")

        assert da_original.ndim == da_roundtrip.ndim
        assert abs(float(da_original.values) - float(da_roundtrip.values)) < 1e-5


# ====================================================================================
# Data Type Support
# ====================================================================================


@pytest.mark.unit
class TestDataTypes:
    """Test support for different HAR data types."""

    def test_read_re_type(self, harpyx_test_file):
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "PROD")
        assert da.dtype in (np.dtype("float32"), np.dtype("float64"))
        assert da.attrs.get("har_type") == "RE"

    def test_read_2i_labeled(self, harpyx_test_file):
        # HARPY does not preserve set labels for 2I headers on write/read;
        # the array data and integer dtype should still be correct.
        da = harpyx.read_har_to_dataarray(harpyx_test_file, "CNTS")
        assert np.issubdtype(da.dtype, np.integer)

    def test_reject_rl_type(self, harpy_test_file):
        """RL type cannot be written by HARPY (_writeHeader7D always writes 'RE')."""
        pytest.skip("RL is a legacy read-only format; HARPY cannot write it")

    def test_reject_2r_type(self, harpyx_test_file):
        """2R type (unlabeled) raises TypeError when reject_unlabeled=True."""
        with pytest.raises(TypeError, match="2R"):
            harpyx.read_har_to_dataarray(harpyx_test_file, "UNLD")

    def test_read_2r_with_reject_unlabeled_false(self, harpyx_test_file):
        """reject_unlabeled=False allows reading a 2R header as a raw DataArray."""
        da = harpyx.read_har_to_dataarray(
            harpyx_test_file, "UNLD", reject_unlabeled=False
        )
        assert da.ndim == 2
        assert da.shape == (3, 4)
        assert np.issubdtype(da.dtype, np.floating)

    def test_error_message_clear(self, harpyx_test_file):
        """Error message for unsupported types includes type name and guidance."""
        with pytest.raises(TypeError) as exc_info:
            harpyx.read_har_to_dataarray(harpyx_test_file, "UNLD")
        msg = str(exc_info.value)
        assert "2R" in msg
        assert "reject_unlabeled" in msg


# ====================================================================================
# Writing
# ====================================================================================


@pytest.mark.unit
class TestWriting:
    """Test writing xarray DataArrays to HAR files."""

    def test_write_requires_coordinates(self, tmp_path, unlabeled_dataarray):
        output_file = tmp_path / "test_write.har"
        with pytest.raises(ValueError, match="coordinates"):
            harpyx.write_dataarray_to_har(unlabeled_dataarray, str(output_file))

    def test_write_with_coords_succeeds(self, tmp_path, simple_dataarray):
        output_file = tmp_path / "test_write.har"
        harpyx.write_dataarray_to_har(simple_dataarray, str(output_file))
        assert output_file.exists()

    def test_write_sets_created(self, tmp_path, simple_dataarray):
        output_file = tmp_path / "test_write.har"
        harpyx.write_dataarray_to_har(simple_dataarray, str(output_file), write_sets=True)

        hf = HarFileObj(str(output_file))
        headers = hf.getHeaderArrayNames()
        assert "PROD" in headers
        assert len(headers) > 1

    def test_write_duplicate_dims(self, tmp_path, duplicate_dim_dataarray):
        """REG__1, REG__2 dims are written back as REG, REG in the HAR file."""
        output_file = tmp_path / "test_duplicate_write.har"
        harpyx.write_dataarray_to_har(duplicate_dim_dataarray, str(output_file))

        hf = HarFileObj(str(output_file))
        hao = hf["TRAD"]
        assert hao.setNames == ["COM", "REG", "REG"]

    def test_write_dimension_sets_attr(self, tmp_path, duplicate_dim_dataarray):
        """attrs['dimension_sets'] is used when writing duplicate dims."""
        output_file = tmp_path / "test_dim_sets_attr.har"
        harpyx.write_dataarray_to_har(duplicate_dim_dataarray, str(output_file))

        hf = HarFileObj(str(output_file))
        hao = hf["TRAD"]
        expected = duplicate_dim_dataarray.attrs["dimension_sets"]
        assert hao.setNames == expected

    def test_write_integer_array(self, tmp_path, integer_dataarray):
        output_file = tmp_path / "test_integer.har"
        harpyx.write_dataarray_to_har(integer_dataarray, str(output_file))

        hfi = HarFileIO.readHarFileInfo(str(output_file))
        ha_info = hfi.getHeaderArrayInfo("CNTS")
        assert ha_info.data_type in ["2I", "RE"]

    def test_append_mode_preserves_existing_headers(
        self, tmp_path, simple_dataarray, scalar_dataarray
    ):
        """mode='a' adds to an existing file without overwriting prior headers."""
        output_file = tmp_path / "append_test.har"
        harpyx.write_dataarray_to_har(simple_dataarray, str(output_file), mode="w")
        harpyx.write_dataarray_to_har(
            scalar_dataarray, str(output_file), mode="a", write_sets=False
        )

        hf = HarFileObj(str(output_file))
        headers = hf.getHeaderArrayNames()
        assert "PROD" in headers
        assert "GDP" in headers


# ====================================================================================
# Round-Trip
# ====================================================================================


@pytest.mark.roundtrip
class TestRoundTrip:
    """Test that data is preserved through read → write → read cycles."""

    def test_simple_array_roundtrip(self, tmp_path, harpyx_test_file):
        da_original = harpyx.read_har_to_dataarray(harpyx_test_file, "PROD")

        output_file = tmp_path / "roundtrip_simple.har"
        harpyx.write_dataarray_to_har(da_original, str(output_file))

        da_roundtrip = harpyx.read_har_to_dataarray(str(output_file), "PROD")
        xr.testing.assert_allclose(da_original, da_roundtrip)

    def test_duplicate_dims_roundtrip(self, tmp_path, harpyx_test_file):
        da_original = harpyx.read_har_to_dataarray(harpyx_test_file, "TRAD")

        output_file = tmp_path / "roundtrip_duplicate.har"
        harpyx.write_dataarray_to_har(da_original, str(output_file))

        da_roundtrip = harpyx.read_har_to_dataarray(str(output_file), "TRAD")
        assert da_original.dims == da_roundtrip.dims
        xr.testing.assert_allclose(da_original, da_roundtrip)

    def test_metadata_roundtrip(self, tmp_path, harpyx_test_file):
        da_original = harpyx.read_har_to_dataarray(harpyx_test_file, "PROD")

        output_file = tmp_path / "roundtrip_metadata.har"
        harpyx.write_dataarray_to_har(da_original, str(output_file))

        da_roundtrip = harpyx.read_har_to_dataarray(str(output_file), "PROD")
        for key in ["header_id", "long_name", "coefficient_name"]:
            if key in da_original.attrs:
                assert da_roundtrip.attrs[key] == da_original.attrs[key]


# ====================================================================================
# Dataset Operations
# ====================================================================================


@pytest.mark.unit
class TestDataset:
    """Test reading/writing complete Datasets."""

    def test_read_full_file(self, harpyx_test_file):
        ds = harpyx.read_har_to_dataset(harpyx_test_file)
        assert len(ds.data_vars) > 0
        assert len(ds.coords) > 0

    def test_shared_coords_merged(self, harpyx_test_file):
        """Variables sharing a dimension use the same coordinate object."""
        ds = harpyx.read_har_to_dataset(harpyx_test_file)
        if "PROD" in ds and "CONS" in ds:
            assert "COM" in ds["PROD"].dims
            assert "COM" in ds["CONS"].dims
            assert ds["PROD"].coords["COM"].equals(ds["CONS"].coords["COM"])

    def test_write_dataset(self, tmp_path, sample_dataset):
        output_file = tmp_path / "dataset_write.har"
        harpyx.write_dataset_to_har(sample_dataset, str(output_file))
        assert output_file.exists()


# ====================================================================================
# Validation
# ====================================================================================


@pytest.mark.unit
class TestValidation:
    """Test metadata validation functions."""

    def test_validate_accepts_good_metadata(self, simple_dataarray):
        errors = harpyx.validate_har_metadata(simple_dataarray)
        assert len(errors) == 0

    def test_validate_rejects_missing_header_id(self, simple_dataarray):
        da = simple_dataarray.copy()
        del da.attrs["header_id"]

        errors = harpyx.validate_har_metadata(da)
        assert len(errors) > 0
        assert any("header_id" in err for err in errors)

    def test_validate_rejects_missing_coords(self, unlabeled_dataarray):
        errors = harpyx.validate_har_metadata(unlabeled_dataarray)
        assert len(errors) > 0
        assert any("coordinate" in err.lower() for err in errors)


# ====================================================================================
# List Headers
# ====================================================================================


@pytest.mark.unit
class TestListHeaders:
    """Test list_har_headers function."""

    def test_returns_metadata_dicts_by_default(self, harpyx_test_file):
        headers = harpyx.list_har_headers(harpyx_test_file)
        assert len(headers) > 0
        assert all(isinstance(h, dict) for h in headers)
        assert all({"header_id", "data_type", "long_name", "shape"} <= h.keys() for h in headers)

    def test_ids_only_mode(self, harpyx_test_file):
        headers = harpyx.list_har_headers(harpyx_test_file, include_metadata=False)
        assert all(isinstance(h, str) for h in headers)
        assert "PROD" in headers

    def test_set_filter(self, harpyx_test_file):
        headers = harpyx.list_har_headers(harpyx_test_file, header_type="set")
        assert len(headers) > 0
        assert all(h["data_type"] == "1C" for h in headers)

    def test_data_filter(self, harpyx_test_file):
        headers = harpyx.list_har_headers(harpyx_test_file, header_type="data")
        assert all(h["data_type"] != "1C" for h in headers)
        assert any(h["header_id"] == "PROD" for h in headers)

    def test_shape_reported(self, harpyx_test_file):
        headers = harpyx.list_har_headers(harpyx_test_file)
        prod = next(h for h in headers if h["header_id"] == "PROD")
        assert prod["shape"][:2] == (3, 5)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            harpyx.list_har_headers("nonexistent.har")


# ====================================================================================
# Error Handling
# ====================================================================================


@pytest.mark.unit
class TestErrorHandling:
    """Test error messages and exception handling."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            harpyx.read_har_sets("nonexistent.har")

    def test_header_not_found(self, harpyx_test_file):
        with pytest.raises((KeyError, ValueError), match="BADHEADER"):
            harpyx.read_har_to_dataarray(harpyx_test_file, "BADHEADER")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
