import numpy
import pytest

from sgnts.base.offset import Offset, TimeUnits
from sgnts.base.slice_tools import TIME_MAX, TIME_MIN, TSSlice, TSSlices


class TestSlice:
    """Test group for TSSlice class"""

    def test_init(self):
        """Test creating a TSSlice object"""
        slc = TSSlice(1, 2)
        assert slc[0] == 1
        assert slc[1] == 2

    def test_err_valid_min_time(self):
        """Test error from start below TIME_MIN"""
        with pytest.raises(ValueError):
            TSSlice(TIME_MIN - 1, 0)

    def test_err_valid_max_time(self):
        """Test error from stop above TIME_MAX"""
        with pytest.raises(ValueError):
            TSSlice(0, TIME_MAX + 1)

    def test_err_valid_null(self):
        """Test validity of null case, e.g. both or neither can be None"""
        with pytest.raises(ValueError):
            TSSlice(0, None)
        with pytest.raises(ValueError):
            TSSlice(None, 0)

    def test_err_valid_dtype(self):
        """Test validity"""
        with pytest.raises(ValueError):
            TSSlice(1.0, 1.0)

    def test_err_valid_ordering(self):
        """Test validity"""
        with pytest.raises(ValueError):
            TSSlice(1, 0)

    def test_slice(self):
        """Test coercion to builtin slice"""
        assert TSSlice(1, 2).slice == slice(1, 2, 1)
        assert TSSlice(None, None).slice == slice(-1, -1, 1)

    def test_comparison(self):
        """Test comparing two slices"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(2, 3)
        assert slc2 >= slc
        assert slc2 > slc
        assert slc <= slc2
        assert slc < slc2

    def test_comparison_null(self):
        """Test that null slices are handled in comparisons"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(None, None)
        assert not slc2 > slc
        assert not slc2 < slc
        assert not slc > slc2
        assert not slc < slc2

    def test_subtraction(self):
        """Test subtraction method"""
        slc = TSSlice(1, 3)
        slc2 = TSSlice(2, 3)
        s = slc2 - slc
        assert s == [TSSlice(1, 2)]

    def test_subtraction_disjoint(self):
        """Test subtraction method"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(3, 4)
        s = slc2 - slc
        assert s == []

    def test_union(self):
        """Test union method"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(2, 3)
        u = slc | slc2
        assert u == TSSlice(1, 3)

    def test_union_null(self):
        """Test that a null slice is propagated in union"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(None, None)
        u = slc | slc2
        assert u == slc2

    def test_isfinite(self):
        """Test isfinite method"""
        slc = TSSlice(1, 2)
        assert slc.isfinite()
        slc = TSSlice(None, None)
        assert not slc.isfinite()

    def test_bool(self):
        """Test boolean coercion"""
        assert not TSSlice(None, None).isfinite()
        assert TSSlice(1, 2)

    def test_index_numpy(self):
        """Test compatibility indexing numpy arrays with a TSSlice object"""
        data = numpy.array([1, 2, 3, 4, 5])
        slc = TSSlice(1, 3)
        res = data[slc.slice]
        numpy.testing.assert_almost_equal(res, numpy.array([2, 3]))


class TestTSSlices:
    """Test group for TSSlices class"""

    def test_valid_slices(self):
        slcs = TSSlices([TSSlice(1, 2), TSSlice(3, 4)])
        assert slcs.invert(TSSlice(0, 5)) == TSSlices(
            slices=[
                TSSlice(start=0, stop=1),
                TSSlice(start=2, stop=3),
                TSSlice(start=4, stop=5),
            ]
        )

    def test_search(self):
        """Test search method for TSSlices"""
        slcs = TSSlices([TSSlice(1, 2), TSSlice(3, 4)])
        assert slcs.search(TSSlice(0, 5)) == TSSlices(
            slices=[TSSlice(start=1, stop=2), TSSlice(start=3, stop=4)]
        )
        assert slcs.search(TSSlice(-1, 0)) == TSSlices(slices=[])

    def test_search_null_slice(self):
        """Test search method for TSSlices"""
        slcs = TSSlices([TSSlice(TIME_MIN, TIME_MAX)])
        assert slcs.search(TSSlice(0, 5)) == TSSlices(slices=[TSSlice(start=0, stop=5)])

    @pytest.mark.skip(reason="Shouldn't use capsys")
    def test_slices(self, capsys):

        for A, B in [
            (TSSlice(0, 3), TSSlice(2, 5)),
            (TSSlice(0, 3), TSSlice(4, 6)),
            (TSSlice(0, 3), TSSlice(1, 2)),
            (TSSlice(0, 3), TSSlice(1, 3)),
            (TSSlice(0, 3), TSSlice(None, None)),
        ]:
            print("\nA: %s\nB: %s\n" % (A, B))
            print("1.\tTrue if A else False:", True if A else False)
            print("2.\tTrue if B else False:", True if B else False)
            print("3.\tA>B:", A > B)
            print("4.\tB>A:", B > A)
            print("5.\tA&B:", A & B)
            print("6.\tB&A:", B & A)
            print("7.\tA|B:", A | B)
            print("8.\tB|A:", B | A)
            print("9.\tA+B:", A + B)
            print("10.\tB+A:", B + A)
            print("11.\tA-B:", A - B)
            print("12.\tB-A:", B - A)

        for slices in [
            TSSlices(
                [
                    TSSlice(0, 4),
                    TSSlice(2, 6),
                    TSSlice(1, 3),
                ]
            ),
            TSSlices([TSSlice(0, 4), TSSlice(2, 6), TSSlice(1, 3), TSSlice(8, 10)]),
        ]:
            print("\nslices = %s\n" % (slices,))
            print("1.\tslices.simplify() = %s" % slices.simplify())
            print("2.\tslices.intersection() = %s" % slices.intersection())
            print(
                "3.\tslices.search(TSSlice(2,4), align=True) = %s"
                % slices.search(TSSlice(2, 4), align=True)
            )
            print(
                "4.\tslices.search(TSSlice(2,4), align=False) = %s"
                % slices.search(TSSlice(2, 4), align=False)
            )
            print("5.\tslices.invert(TSSlice(2,4)) = %s" % slices.invert(TSSlice(2, 4)))

    def test_search_unaligned(self):
        """Test searching for a slice in slices"""
        slc = TSSlice(1, 2)
        slcs = TSSlices([TSSlice(1, 3), TSSlice(3, 4)])
        res = slcs.search(slc, align=False)
        assert isinstance(res, TSSlices)
        assert res == TSSlices([TSSlice(1, 3)])

    def test_intersection_of_multiple_edge_cases(self):
        """Test intersection_of_multiple with edge cases for 100% coverage"""
        # Test empty list (covers line 453)
        result = TSSlices.intersection_of_multiple([])
        assert result == TSSlices([])

        # Test single element list (covers line 456)
        single = TSSlices([TSSlice(10, 20)])
        result = TSSlices.intersection_of_multiple([single])
        assert result == single

    def test_iter(self):
        """Test iterating over TSSlices to cover __iter__ method"""
        slcs = TSSlices([TSSlice(1, 2), TSSlice(3, 4), TSSlice(5, 6)])
        result = list(slcs)
        assert result == [TSSlice(1, 2), TSSlice(3, 4), TSSlice(5, 6)]

    def test_slice_property(self):
        """Test .slice property to cover line 332"""
        slcs = TSSlices([TSSlice(1, 2), TSSlice(3, 4), TSSlice(5, 6)])
        assert slcs.slice == TSSlice(1, 6)

    def test_align_to_rate_basic(self):
        """Test align_to_rate with basic downsampling from 4Hz to 2Hz"""
        # Setup: MAX_RATE = 16384
        # At 4 Hz: offset_per_sample = 16384 / 4 = 4096
        # At 2 Hz: offset_per_sample = 16384 / 2 = 8192

        # Create slices at 4Hz boundaries
        # (0, 4096) = 1 sample at 4Hz = [0, 0.5) samples at 2Hz
        # -> rounds to empty
        # (8192, 16384) = 2 samples at 4Hz = [1, 2) samples at 2Hz
        # -> keeps as is
        slices = TSSlices([TSSlice(0, 4096), TSSlice(8192, 16384)])
        result = slices.align_to_rate(2)

        # First slice should be eliminated, second should remain
        assert result == TSSlices([TSSlice(8192, 16384)])

    def test_align_to_rate_gap_expansion(self):
        """Test that gaps expand when downsampling"""
        # At 8 Hz: offset_per_sample = 16384 / 8 = 2048
        # Slice (2048, 6144) = samples [1, 3) at 8Hz = 2 samples
        # At 4 Hz: offset_per_sample = 16384 / 4 = 4096
        # In 4Hz samples: [0.5, 1.5)
        # -> ceil(0.5)=1, floor(1.5)=1 -> becomes (4096, 4096) = empty

        slices = TSSlices([TSSlice(2048, 6144)])
        result = slices.align_to_rate(4)

        # Slice should be eliminated due to gap expansion
        assert result == TSSlices([])

    def test_align_to_rate_multiple_slices(self):
        """Test align_to_rate with multiple slices"""
        # At 4 Hz: offset_per_sample = 4096
        # Create several slices:
        # (0, 8192) = 2 samples at 4Hz = [0, 2) at 2Hz -> stays
        # (16384, 24576) = 2 samples at 4Hz = [2, 3) at 2Hz -> stays
        # (32768, 36864) = 1 sample at 4Hz = [4, 4.5) at 2Hz -> eliminated

        slices = TSSlices(
            [
                TSSlice(0, 8192),  # [0, 2) at 2Hz -> stays
                TSSlice(16384, 24576),  # [2, 3) at 2Hz -> stays
                TSSlice(32768, 36864),  # [4, 4.5) at 2Hz -> eliminated
            ]
        )
        result = slices.align_to_rate(2)

        expected = TSSlices([TSSlice(0, 8192), TSSlice(16384, 24576)])
        assert result == expected

    def test_align_to_rate_already_aligned(self):
        """Test align_to_rate when slices are already aligned"""
        # Slices at 2Hz boundaries stay the same when aligning to 2Hz
        slices = TSSlices([TSSlice(0, 8192), TSSlice(16384, 24576)])
        result = slices.align_to_rate(2)

        assert result == slices

    def test_align_to_rate_empty(self):
        """Test align_to_rate with empty TSSlices"""
        slices = TSSlices([])
        result = slices.align_to_rate(4)

        assert result == TSSlices([])

    def test_align_to_rate_null_slice(self):
        """Test align_to_rate with null slices"""
        slices = TSSlices([TSSlice(None, None)])
        result = slices.align_to_rate(4)

        assert result == TSSlices([])

    def test_align_to_rate_invalid_rate(self):
        """Test align_to_rate with invalid target rate"""
        slices = TSSlices([TSSlice(0, 4096)])

        # Should raise assertion error for rate not in ALLOWED_RATES
        with pytest.raises(AssertionError):
            slices.align_to_rate(3)  # Not a power of 2

    def test_align_to_rate_boundary_rounding(self):
        """Test that boundaries round correctly (start up, stop down)"""
        # At 16 Hz: offset_per_sample = 16384 / 16 = 1024
        # Create slice (1024, 7168) = samples [1, 7) at 16Hz
        # At 4 Hz: offset_per_sample = 4096
        # In 4Hz samples: [0.25, 1.75)
        # -> ceil(0.25)=1, floor(1.75)=1
        # Should become (4096, 4096) = empty

        slices = TSSlices([TSSlice(1024, 7168)])
        result = slices.align_to_rate(4)

        # Should be eliminated
        assert result == TSSlices([])

    def test_align_to_rate_large_slice(self):
        """Test align_to_rate with a large slice spanning many samples"""
        # At 8 Hz: offset_per_sample = 2048
        # Create slice spanning 0 to 20480 (10 samples at 8Hz)
        # At 2 Hz: offset_per_sample = 8192
        # In 2Hz samples: [0, 2.5) -> ceil(0)=0, floor(2.5)=2
        # Should become (0, 16384)

        slices = TSSlices([TSSlice(0, 20480)])
        result = slices.align_to_rate(2)

        expected = TSSlices([TSSlice(0, 16384)])
        assert result == expected


class TestTSSliceUnits:
    """
    Tests specific to TSSlice unit handling, validation, and API integration.
    Relies on TestOffsetConversion for verification of the arithmetic correctness.
    """

    @pytest.fixture(autouse=True)
    def setup_offset_constants(self):
        Offset.set_max_rate(16384)
        yield

    def test_initialization_validation(self):
        """Verify strict type checking enforced by __post_init__ based on units."""
        # 1. Floating point values are strictly forbidden for integer-based units
        with pytest.raises(ValueError, match="must be integers"):
            TSSlice(1.5, 2.5, units=TimeUnits.OFFSETS)

        with pytest.raises(ValueError, match="must be integers"):
            TSSlice(1.5, 2.5, units=TimeUnits.NANOSECONDS)

        # 2. SAMPLES unit is valid (no rate required at init anymore)
        s = TSSlice(10, 20, units=TimeUnits.SAMPLES)
        assert s.units == TimeUnits.SAMPLES

        # 3. SECONDS unit allows floats
        s = TSSlice(1.0, 2.0, units=TimeUnits.SECONDS)
        assert s.units == TimeUnits.SECONDS

    def test_conversion_api_integration(self):
        """Verify the .convert() method returns correct TSSlice objects."""
        # Setup: 1 second = 16384 offsets
        s_sec = TSSlice(1.0, 2.0, units=TimeUnits.SECONDS)

        # Convert to Offsets
        s_off = s_sec.convert(TimeUnits.OFFSETS)

        assert isinstance(s_off, TSSlice)
        assert s_off.units == TimeUnits.OFFSETS
        assert s_off.start == 16384
        assert s_off.stop == 32768

        # Check immutability
        assert s_sec.units == TimeUnits.SECONDS

    def test_sample_conversion_flow(self):
        """Verify flow converting to/from SAMPLES requires passing
        rates to convert()."""
        s_off = TSSlice(16384, 32768, units=TimeUnits.OFFSETS)  # 1s to 2s

        # Convert to SAMPLES at 4Hz
        # 1s at 4Hz = sample 4. 2s at 4Hz = sample 8.
        s_samp = s_off.convert(TimeUnits.SAMPLES, to_sample_rate=4)

        assert s_samp.units == TimeUnits.SAMPLES
        assert s_samp.start == 4
        assert s_samp.stop == 8

        # Convert back to OFFSETS (Must provide from_sample_rate)
        s_rec = s_samp.convert(TimeUnits.OFFSETS, from_sample_rate=4)

        assert s_rec.start == 16384
        assert s_rec.stop == 32768

        # Converting SAMPLES -> OFFSETS without rate fails
        # (via Offset.convert validation)
        # Note: Offset.convert raises ValueError if rate is missing
        with pytest.raises(ValueError, match="from_sample_rate required"):
            s_samp.convert(TimeUnits.OFFSETS)

    def test_infinite_slice_conversion(self):
        """Verify converting an infinite slice preserves None bounds
        but updates units."""
        s_inf = TSSlice(None, None, units=TimeUnits.SECONDS)

        # Convert to Nanoseconds
        s_ns = s_inf.convert(TimeUnits.NANOSECONDS)

        assert s_ns.start is None
        assert s_ns.stop is None
        assert s_ns.units == TimeUnits.NANOSECONDS

    def test_mixed_unit_operations_fail(self):
        """
        Verify that boolean operations (intersection, union, subtraction)
        block interaction between different units.
        """
        s_off = TSSlice(0, 16384, units=TimeUnits.OFFSETS)
        s_sec = TSSlice(0.0, 1.0, units=TimeUnits.SECONDS)

        # Intersection
        with pytest.raises(ValueError, match="Cannot operate on mixed units"):
            _ = s_off & s_sec

        # Union
        with pytest.raises(ValueError, match="Cannot operate on mixed units"):
            _ = s_off | s_sec

        # Subtraction
        with pytest.raises(ValueError, match="Cannot operate on mixed units"):
            _ = s_off - s_sec

    def test_python_slice_generation(self):
        """Verify behavior of the .slice property."""
        # Valid for integer units
        s_off = TSSlice(10, 20, units=TimeUnits.OFFSETS)
        p_slice = s_off.slice
        assert p_slice.start == 10
        assert p_slice.stop == 20
        assert p_slice.step == 1

        # Invalid for float units (SECONDS)
        s_sec = TSSlice(1.0, 2.0, units=TimeUnits.SECONDS)
        with pytest.raises(TypeError, match="Cannot create python slice"):
            _ = s_sec.slice

    def test_init_err_seconds_dtype(self):
        """Test that initializing with SECONDS unit requires floats"""
        with pytest.raises(
            ValueError, match="start and stop must be numeric for SECONDS"
        ):
            TSSlice("a", "b", units=TimeUnits.SECONDS)

    def test_init_err_non_seconds_dtype_float(self):
        """Test that initializing with SECONDS unit requires floats"""
        with pytest.raises(
            ValueError, match="start and stop must be integers for unit"
        ):
            TSSlice(0.0, 1.0, units=TimeUnits.OFFSETS)


class TestTSSlicesUnits:
    """Tests for unit-related behavior in TSSlices class"""

    def test_init_err_mixed_units(self):
        """Test that initializing TSSlices with mixed units raises an error"""
        with pytest.raises(
            ValueError, match="All slices in TSSlices " "must have the same units."
        ):
            TSSlices(
                [
                    TSSlice(0, 10, units=TimeUnits.OFFSETS),
                    TSSlice(0.0, 1.0, units=TimeUnits.SECONDS),
                ]
            )

    def test_convert(self):
        """Test converting TSSlices to a different unit"""
        slices = TSSlices(
            [
                TSSlice(0, 16384, units=TimeUnits.OFFSETS),
                TSSlice(16384, 32768, units=TimeUnits.OFFSETS),
            ]
        )
        converted = slices.convert(TimeUnits.SECONDS)
        assert all(s.units == TimeUnits.SECONDS for s in converted.slices)
        assert converted.slices[0].start == 0.0
        assert converted.slices[0].stop == 1.0
        assert converted.slices[1].start == 1.0
        assert converted.slices[1].stop == 2.0

    def test_search_err_mixed_units(self):
        """Test that searching with a slice of different units raises an error"""
        slices = TSSlices(
            [
                TSSlice(0, 10, units=TimeUnits.OFFSETS),
                TSSlice(10, 20, units=TimeUnits.OFFSETS),
            ]
        )
        search_slice = TSSlice(0.0, 1.0, units=TimeUnits.SECONDS)
        with pytest.raises(ValueError, match="Search slice units "):
            slices.search(search_slice)

    def test_invert_err_mixed_units(self):
        """Test that inverting with a slice of different units raises an error"""
        slices = TSSlices(
            [
                TSSlice(0, 10, units=TimeUnits.OFFSETS),
                TSSlice(10, 20, units=TimeUnits.OFFSETS),
            ]
        )
        invert_slice = TSSlice(0.0, 1.0, units=TimeUnits.SECONDS)
        with pytest.raises(ValueError, match="Boundary slice units "):
            slices.invert(invert_slice)

    def test_align_to_rate_err_non_offset_units_missing_sample_rate(self):
        """Test that align_to_rate raises an error if slices are not in OFFSETS"""
        slices = TSSlices(
            [
                TSSlice(0, 16384, units=TimeUnits.SAMPLES),
                TSSlice(16384, 32768, units=TimeUnits.SAMPLES),
            ]
        )
        with pytest.raises(ValueError, match="Cannot auto-align from SAMPLES units"):
            slices.align_to_rate(4)

    def test_align_to_rate_non_offsets_acceptable(self):
        """Test that align_to_rate works if slices are not in OFFSETS but
        also not in SAMPLES"""
        slices = TSSlices(
            [
                TSSlice(0.0, 1.0, units=TimeUnits.SECONDS),
                TSSlice(1.0, 2.0, units=TimeUnits.SECONDS),
            ]
        )
        # Should not raise an error since SECONDS is not SAMPLES
        aligned = slices.align_to_rate(4)
        # Make the expected result manually (manually converted to offsets)
        expected = TSSlices(
            [
                TSSlice(0, 16384, units=TimeUnits.OFFSETS),
                TSSlice(16384, 32768, units=TimeUnits.OFFSETS),
            ]
        )
        assert aligned == expected
