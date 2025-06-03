"""Test SegmentSource with custom values functionality"""

import numpy as np
import pytest

from sgnts.sources import SegmentSource


def test_segment_source_with_values():
    """Test SegmentSource with custom values for each segment"""
    src = SegmentSource(
        name="src",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        end=20.0,
        segments=((1e9, 3e9), (5e9, 7e9), (10e9, 12e9)),
        values=(2, 3, 4),
    )

    pad = src.srcs["data"]
    found_values = set()

    # Check frames until we've seen all expected values
    for _ in range(20):  # Reduced from 100
        frame = src.new(pad)

        for buf in frame:
            if not buf.is_gap and buf.data is not None and len(buf.data) > 0:
                found_values.add(buf.data[0])

        if frame.EOS or found_values == {2, 3, 4}:
            break

    assert found_values == {2, 3, 4}, f"Expected values {2, 3, 4}, found {found_values}"


def test_segment_source_with_array_values():
    """Test SegmentSource with array values"""
    src = SegmentSource(
        name="src",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        end=10.0,
        segments=((1e9, 3e9), (5e9, 7e9)),
        values=(0, 1),  # Will create zeros and ones arrays
    )

    pad = src.srcs["data"]
    found_zeros = False
    found_ones = False

    for _ in range(10):  # Reduced from 50
        frame = src.new(pad)

        for buf in frame:
            if not buf.is_gap and buf.data is not None and len(buf.data) > 0:
                if np.all(buf.data == 0):
                    found_zeros = True
                elif np.all(buf.data == 1):
                    found_ones = True

        if frame.EOS or (found_zeros and found_ones):
            break

    assert found_zeros and found_ones, "Should have found both zeros and ones"


def test_segment_source_no_values():
    """Test SegmentSource without values (default behavior)"""
    src = SegmentSource(
        name="src",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        end=10.0,
        segments=((1e9, 3e9), (5e9, 7e9)),
    )

    pad = src.srcs["data"]

    # Check a few frames to verify default value
    for _ in range(5):
        frame = src.new(pad)

        for buf in frame:
            if not buf.is_gap and buf.data is not None and len(buf.data) > 0:
                assert np.all(buf.data == 1), "Default value should be 1"

        if frame.EOS:
            break


def test_segment_source_overlapping_segments_error():
    """Test that overlapping segments raise an assertion error"""
    with pytest.raises(AssertionError, match="Input segments must be non-overlapping"):
        SegmentSource(
            name="src",
            source_pad_names=("data",),
            rate=256,
            t0=0.0,
            end=10.0,
            segments=((1e9, 3e9), (2e9, 4e9)),  # Overlapping segments
        )


def test_segment_source_values_length_mismatch():
    """Test that mismatched values length raises an assertion error"""
    with pytest.raises(AssertionError, match="Length of values .* must match"):
        SegmentSource(
            name="src",
            source_pad_names=("data",),
            rate=256,
            t0=0.0,
            end=10.0,
            segments=((1e9, 3e9), (5e9, 7e9)),
            values=(1, 2, 3),  # Too many values
        )


def test_segment_source_adjacent_segments():
    """Test that adjacent (but not overlapping) segments work correctly"""
    src = SegmentSource(
        name="src",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        end=10.0,
        segments=((1e9, 3e9), (3e9, 5e9)),  # Adjacent but not overlapping
        values=(10, 20),
    )

    assert len(src.segment_data) == 2


def test_segment_source_gap_only_buffers():
    """Test SegmentSource when buffers fall entirely outside segments"""
    src = SegmentSource(
        name="src",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        end=2.0,
        segments=((5e9, 6e9), (10e9, 11e9)),  # Outside t0-end range
        values=(42, 43),
    )

    pad = src.srcs["data"]

    # Verify no segments are in range
    assert len(src.segment_data) == 0
    assert len(src.segment_slices.slices) == 0

    # All buffers should be gaps
    frame = src.new(pad)
    for buf in frame:
        assert buf.is_gap


def test_segment_source_complex_values():
    """Test SegmentSource with complex number values"""
    # Create segments with complex values
    complex_values = (1 + 2j, 3 - 4j, 5 + 0j)

    src = SegmentSource(
        name="src",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        end=10.0,
        segments=((1e9, 3e9), (4e9, 6e9), (7e9, 9e9)),
        values=complex_values,
    )

    pad = src.srcs["data"]
    found_values = set()

    # Collect values from a few frames
    for _ in range(10):
        frame = src.new(pad)

        for buf in frame:
            if not buf.is_gap and buf.data is not None and len(buf.data) > 0:
                # Check that the data is already complex
                assert np.iscomplexobj(buf.data), "Buffer data should be complex"
                # Get the value directly - it should already be complex
                val = buf.data[0]
                found_values.add(val)

        if frame.EOS:
            break

    # Verify we found our complex values
    assert len(found_values) > 0, "Should have found some complex values"
    for val in found_values:
        assert val in complex_values, f"Found unexpected value {val}"
        # Also verify each value is actually complex type
        assert isinstance(
            val, (complex, np.complexfloating)
        ), f"Value {val} should be complex type"


def test_segment_source_rounding():
    """Test SegmentSource automatic rounding to nearest sample boundary"""
    from sgnts.base import Offset

    # Test at 256 Hz sample rate
    rate = 256
    offset_factor = Offset.MAX_RATE // rate  # Should be 64 with MAX_RATE=16384
    sample_period_ns = int(1e9 / rate)  # 3906250 ns

    # Helper function to create a time from samples
    def samples_to_ns(samples, sample_rate):
        """Convert samples at given rate to nanoseconds"""
        offset = Offset.fromsamples(samples, sample_rate)
        return Offset.tons(offset)

    # Create segments with times that need rounding
    segments = (
        # Segment 1: Exact sample boundaries (no rounding needed)
        (samples_to_ns(1, rate), samples_to_ns(4, rate)),
        # Segment 2: Small offset that should round to nearest sample
        # 10 samples + 100ns should round back to 10 samples
        # 20 samples - 100ns should round back to 20 samples
        (samples_to_ns(10, rate) + 100, samples_to_ns(20, rate) - 100),
        # Segment 3: Larger offset to test rounding
        # 30 samples + 1ms: Will round to nearest sample
        # 40 samples + 2ms: Will round to nearest sample
        (samples_to_ns(30, rate) + 1000000, samples_to_ns(40, rate) + 2000000),
        # Segment 4: Test rounding at half-sample boundary
        # Add exactly half a sample period - should round up
        (
            samples_to_ns(50, rate) + sample_period_ns // 2,
            samples_to_ns(60, rate) + sample_period_ns // 2,
        ),
    )

    values = (100, 200, 300, 400)

    # Calculate expected rounding for verification
    # (keeping calculations for test validation without debug output)

    # Create the source - it should automatically round segment times
    src = SegmentSource(
        name="src",
        source_pad_names=("data",),
        rate=rate,
        t0=0.0,
        end=1.0,  # 1 second = 256 samples
        segments=segments,
        values=values,
    )

    # Check the segment data that was created
    assert len(src.segment_data) == 4, "All segments should be in range"

    # Verify the automatic rounding worked
    for _i, (seg_slice, _orig_idx) in enumerate(src.segment_data):
        # ALL stored offsets should now be valid for the sample rate
        assert (
            seg_slice.start % offset_factor == 0
        ), f"Start offset {seg_slice.start} should be rounded to valid value"
        assert (
            seg_slice.stop % offset_factor == 0
        ), f"Stop offset {seg_slice.stop} should be rounded to valid value"

    # Now test that we can get data without errors
    pad = src.srcs["data"]
    found_values = set()

    # This should work without errors now!
    for _ in range(10):
        frame = src.new(pad)

        for buf in frame:
            if not buf.is_gap and buf.data is not None and len(buf.data) > 0:
                found_values.add(buf.data[0])
                # Verify all buffer offsets are valid
                assert buf.offset % offset_factor == 0
                assert buf.end_offset % offset_factor == 0

        if frame.EOS:
            break

    # We should find all values
    assert found_values == {
        100,
        200,
        300,
        400,
    }, f"Expected all values, found {found_values}"


def test_segment_source_rounding_edge_cases():
    """Test SegmentSource rounding with edge cases at different sample rates"""
    from sgnts.base import Offset

    # Test at multiple sample rates
    test_cases = [
        # (rate, test_name)
        (1024, "1024 Hz"),
        (512, "512 Hz"),
        (2048, "2048 Hz"),
    ]

    for rate, _test_name in test_cases:
        offset_factor = Offset.MAX_RATE // rate

        # Create segments that test rounding at this rate
        # We'll create times that are slightly off from exact sample boundaries

        # Helper to create time that's slightly off from a sample boundary
        def create_test_time(target_samples, offset_ns, sample_rate):
            """Create a time that's target_samples + offset_ns"""
            exact_offset = Offset.fromsamples(target_samples, sample_rate)
            exact_ns = Offset.tons(exact_offset)
            return exact_ns + offset_ns

        segments = (
            # Segment 1: Very small offset (should round to nearest sample)
            (create_test_time(1, 50, rate), create_test_time(5, -50, rate)),
            # Segment 2: Medium offset (tests rounding)
            (create_test_time(10, 10000, rate), create_test_time(15, -10000, rate)),
            # Segment 3: Large offset that should round
            (create_test_time(20, 500000, rate), create_test_time(25, -500000, rate)),
        )

        values = (10, 20, 30)

        # Test will verify that automatic rounding handles these cases

        src = SegmentSource(
            name="src",
            source_pad_names=("data",),
            rate=rate,
            t0=0.0,
            end=0.1,  # 100ms
            segments=segments,
            values=values,
        )

        # Verify the stored segments are properly aligned
        for _i, (seg_slice, _) in enumerate(src.segment_data):
            # Ensure all segments are aligned to sample boundaries
            assert seg_slice.start % offset_factor == 0
            assert seg_slice.stop % offset_factor == 0
