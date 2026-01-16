#!/usr/bin/env python3
import pytest

from sgnts.base import Time
from sgnts.base.offset import Offset, TimeUnits


def test_offset():
    OLD = Offset.MAX_RATE
    Offset.set_max_rate(OLD * 2)
    Offset.set_max_rate(OLD)

    GPSNS = 122287885562500000  # corresponds to a whole number of 1/16th second buffers
    OFFSET = Offset.fromns(GPSNS)
    assert (GPSNS - Offset.tons(OFFSET)) == 0

    GPSNS = (
        1422287885562500000  # corresponds to a whole number of 1/16th second buffers
    )
    OFFSET = Offset.fromns(GPSNS)
    assert (GPSNS - Offset.tons(OFFSET)) == 0


class TestOffsetConversion:
    """
    Detailed test suite for Offset.convert and associated static methods.
    Verifies unit transformation, sample rate handling, and boundary conditions.
    """

    @pytest.fixture(autouse=True)
    def setup_offset_constants(self):
        """
        Ensure Offset class is in a known state (MAX_RATE=16384) before every test.
        This guarantees arithmetic determinism.
        """
        Offset.set_max_rate(16384)
        yield

    def test_identity_optimization(self):
        """Verify convert returns value as-is when units match,
        bypassing calculations."""
        val = 12345
        # OFFSETS -> OFFSETS
        assert Offset.convert(val, TimeUnits.OFFSETS, TimeUnits.OFFSETS) == val

        # SECONDS -> SECONDS
        assert Offset.convert(1.5, TimeUnits.SECONDS, TimeUnits.SECONDS) == 1.5

        # SAMPLES -> SAMPLES (Same Rate)
        assert (
            Offset.convert(
                100,
                TimeUnits.SAMPLES,
                TimeUnits.SAMPLES,
                from_sample_rate=4096,
                to_sample_rate=4096,
            )
            == 100
        )

    def test_offsets_to_seconds_roundtrip(self):
        """Verify conversion between OFFSETS and SECONDS."""
        # 1 Second = 16384 Offsets
        offsets = 16384
        seconds = 1.0

        # To Seconds
        assert Offset.convert(offsets, TimeUnits.OFFSETS, TimeUnits.SECONDS) == seconds
        # From Seconds
        assert Offset.convert(seconds, TimeUnits.SECONDS, TimeUnits.OFFSETS) == offsets

        # Fractional check
        # 0.5 seconds = 8192 offsets
        assert Offset.convert(0.5, TimeUnits.SECONDS, TimeUnits.OFFSETS) == 8192
        assert Offset.convert(8192, TimeUnits.OFFSETS, TimeUnits.SECONDS) == 0.5

    def test_offsets_to_nanoseconds_roundtrip(self):
        """Verify conversion between OFFSETS and NANOSECONDS."""
        # 1 Second in ns
        one_sec_ns = int(Time.SECONDS)  # Usually 1e9
        max_rate = 16384

        # 1.0s -> 16384 offsets
        assert (
            Offset.convert(max_rate, TimeUnits.OFFSETS, TimeUnits.NANOSECONDS)
            == one_sec_ns
        )
        assert (
            Offset.convert(one_sec_ns, TimeUnits.NANOSECONDS, TimeUnits.OFFSETS)
            == max_rate
        )

        # Small value check (precision/rounding)
        # 1 offset = 1/16384 sec ~= 61035.15 ns. Should round.
        expected_ns = round(1e9 / 16384)
        assert (
            Offset.convert(1, TimeUnits.OFFSETS, TimeUnits.NANOSECONDS) == expected_ns
        )

    def test_large_nanosecond_handling(self):
        """Test the specific logic branch in tons() for large offsets > 1e17."""
        # Force a large offset
        large_offset = 200_000_000_000_000  # Enough to trigger 1e17 check logic

        ns = Offset.convert(large_offset, TimeUnits.OFFSETS, TimeUnits.NANOSECONDS)

        # Logic in tons: offset * Time.SECONDS // Offset.MAX_RATE
        expected = large_offset * int(Time.SECONDS) // 16384
        assert ns == expected
        assert isinstance(ns, int)  # Must remain integer type

    def test_offsets_to_samples_logic(self):
        """Verify conversion between OFFSETS and SAMPLES at specific rates."""
        # At 16384 Hz (MAX_RATE), 1 sample = 1 offset
        assert (
            Offset.convert(
                1, TimeUnits.SAMPLES, TimeUnits.OFFSETS, from_sample_rate=16384
            )
            == 1
        )
        assert (
            Offset.convert(
                1, TimeUnits.OFFSETS, TimeUnits.SAMPLES, to_sample_rate=16384
            )
            == 1
        )

        # At 4 Hz:
        # Stride = 16384 / 4 = 4096 offsets per sample
        assert (
            Offset.convert(1, TimeUnits.SAMPLES, TimeUnits.OFFSETS, from_sample_rate=4)
            == 4096
        )
        assert (
            Offset.convert(4096, TimeUnits.OFFSETS, TimeUnits.SAMPLES, to_sample_rate=4)
            == 1
        )

        # Misalignment check
        # Offset 4097 does not land on a 4Hz boundary (multiples of 4096)
        # FIXED: Updated regex to match "sample points"
        with pytest.raises(
            AssertionError, match="does not map to integer sample points"
        ):
            Offset.convert(4097, TimeUnits.OFFSETS, TimeUnits.SAMPLES, to_sample_rate=4)

    def test_sample_resampling(self):
        """Test converting SAMPLES directly to SAMPLES at different rates."""
        # 4Hz to 8Hz (Up-sampling grid)
        # 1 sample at 4Hz = 0.25s = 2 samples at 8Hz
        val = Offset.convert(
            1,
            TimeUnits.SAMPLES,
            TimeUnits.SAMPLES,
            from_sample_rate=4,
            to_sample_rate=8,
        )
        assert val == 2

        # 8Hz to 4Hz (Down-sampling grid)
        # 2 samples at 8Hz = 0.25s = 1 sample at 4Hz
        val = Offset.convert(
            2,
            TimeUnits.SAMPLES,
            TimeUnits.SAMPLES,
            from_sample_rate=8,
            to_sample_rate=4,
        )
        assert val == 1

        # Invalid Down-sampling (Grid misalignment)
        # 1 sample at 8Hz = 0.125s. At 4Hz, sample points are at 0, 0.25...
        # 0.125s is offset 2048. 4Hz stride is 4096. 2048 % 4096 != 0.
        with pytest.raises(AssertionError):
            Offset.convert(
                1,
                TimeUnits.SAMPLES,
                TimeUnits.SAMPLES,
                from_sample_rate=8,
                to_sample_rate=4,
            )

    def test_seconds_to_samples(self):
        """Verify conversion from SECONDS to SAMPLES."""
        # 0.5 seconds at 4Hz should be 2 samples
        # 0.5s -> 8192 offsets. 4Hz stride is 4096. 8192/4096 = 2.
        assert (
            Offset.convert(0.5, TimeUnits.SECONDS, TimeUnits.SAMPLES, to_sample_rate=4)
            == 2
        )

        # 0.25 seconds at 4Hz should be 1 sample
        assert (
            Offset.convert(0.25, TimeUnits.SECONDS, TimeUnits.SAMPLES, to_sample_rate=4)
            == 1
        )

    def test_nanoseconds_alignment_logic(self):
        """Test the 'from_sample_rate' argument when converting FROM Nanoseconds."""
        # 1 second exact
        ns_val = int(Time.SECONDS)

        # Convert ns to offsets, aligned to 4Hz
        val = Offset.convert(
            ns_val, TimeUnits.NANOSECONDS, TimeUnits.OFFSETS, from_sample_rate=4
        )
        assert val == 16384

        # Jittered timestamp (e.g., 1ns past 1 second)
        ns_jittered = ns_val + 1

        # Without alignment (Raw conversion) -> 16384
        raw = Offset.convert(ns_jittered, TimeUnits.NANOSECONDS, TimeUnits.OFFSETS)
        assert raw == 16384

        # Half-way logic (0.125s) -> Rounds to 0 samples at 4Hz
        ns_half = 125_000_000
        val = Offset.convert(
            ns_half, TimeUnits.NANOSECONDS, TimeUnits.OFFSETS, from_sample_rate=4
        )
        assert val == 0

        # Closer to next sample (0.20s) -> Rounds to 1 sample (4096 offset) at 4Hz
        ns_close = 200_000_000
        val = Offset.convert(
            ns_close, TimeUnits.NANOSECONDS, TimeUnits.OFFSETS, from_sample_rate=4
        )
        assert val == 4096

    def test_missing_arguments_errors(self):
        """Verify proper errors are raised when required sample rates are missing."""

        # Converting TO Samples requires to_sample_rate
        with pytest.raises(ValueError, match="to_sample_rate required"):
            Offset.convert(100, TimeUnits.OFFSETS, TimeUnits.SAMPLES)

        # Converting FROM Samples requires from_sample_rate
        with pytest.raises(ValueError, match="from_sample_rate required"):
            Offset.convert(100, TimeUnits.SAMPLES, TimeUnits.OFFSETS)

        # SAMPLES -> SAMPLES requires both
        with pytest.raises(ValueError, match="from_sample_rate required"):
            Offset.convert(100, TimeUnits.SAMPLES, TimeUnits.SAMPLES, to_sample_rate=4)

    def test_invalid_rates(self):
        """Verify assertions/errors for rates not in ALLOWED_RATES."""
        # 3Hz is not a power of 2, unlikely to be in ALLOWED_RATES for MAX_RATE=16384
        bad_rate = 3

        # FIXED: Updated regex to match "not in ALLOWED_RATES"
        # Used by tosamples
        with pytest.raises(AssertionError, match="not in ALLOWED_RATES"):
            Offset.convert(
                1, TimeUnits.SECONDS, TimeUnits.SAMPLES, to_sample_rate=bad_rate
            )

        # Used by fromsamples
        with pytest.raises(AssertionError, match="not in ALLOWED_RATES"):
            Offset.convert(
                1, TimeUnits.SAMPLES, TimeUnits.SECONDS, from_sample_rate=bad_rate
            )

    def test_cross_unit_conversion(self):
        """Verify conversions skipping the explicit intermediate step."""
        # Nanoseconds -> Seconds
        ns = 500_000_000  # 0.5s
        sec = Offset.convert(ns, TimeUnits.NANOSECONDS, TimeUnits.SECONDS)
        assert sec == 0.5

        # Seconds -> Nanoseconds
        assert (
            Offset.convert(0.25, TimeUnits.SECONDS, TimeUnits.NANOSECONDS)
            == 250_000_000
        )

    def test_convert_err_invalid_unit(self):
        """Verify error is raised for invalid unit types."""
        with pytest.raises(ValueError, match="Unknown from_unit"):
            Offset.convert(100, "INVALID_UNIT", TimeUnits.SECONDS)

        with pytest.raises(ValueError, match="Unknown to_unit"):
            Offset.convert(100, TimeUnits.SECONDS, "INVALID_UNIT")
