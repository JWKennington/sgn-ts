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


def test_segment_source_gaps_between_segments():
    """Test that SegmentSource correctly creates gap buffers between segments."""
    from sgnts.base import Offset

    # Create source with clear gaps between segments
    # Segments at 0.1-0.2s and 0.4-0.5s (gap from 0.2-0.4s)
    segments = (
        (int(0.1 * 1e9), int(0.2 * 1e9)),
        (int(0.4 * 1e9), int(0.5 * 1e9)),
    )
    values = (10, 20)

    src = SegmentSource(
        name="test",
        source_pad_names=("data",),
        rate=64,  # Low rate for simpler numbers
        t0=0.0,
        duration=1.0,
        segments=segments,
        values=values,
    )

    pad = src.srcs["data"]

    # Get frames until we cover the gap region
    # Gap is from ~3277 to ~6553 in offsets
    gap_start_offset = Offset.fromns(int(0.2 * 1e9))
    gap_end_offset = Offset.fromns(int(0.4 * 1e9))

    found_gap = False
    found_data_before = False
    found_data_after = False

    for _ in range(20):
        frame = src.new(pad)

        for buf in frame:
            # Check if buffer overlaps with the gap region
            if buf.offset < gap_end_offset and buf.end_offset > gap_start_offset:
                # If this buffer is entirely within the gap region, it must be a gap
                if buf.offset >= gap_start_offset and buf.end_offset <= gap_end_offset:
                    assert (
                        buf.is_gap
                    ), f"Buffer {buf.offset}-{buf.end_offset} should be gap but isn't"
                    assert buf.data is None, "Gap buffer should have data=None"
                    found_gap = True
                # If buffer spans the gap region, check if it's a gap
                elif buf.is_gap:
                    found_gap = True

            # Check data before gap (may overlap slightly)
            if buf.offset < gap_start_offset and not buf.is_gap:
                assert buf.data is not None
                if all(buf.data == 10):  # All values should be from first segment
                    found_data_before = True

            # Check data after gap (may overlap slightly)
            if buf.end_offset > gap_end_offset and not buf.is_gap:
                assert buf.data is not None
                if all(buf.data == 20):  # All values should be from second segment
                    found_data_after = True

        if frame.EOS:
            break

    assert found_gap, "Should have found gap buffer(s) between segments"
    assert found_data_before, "Should have found data before gap"
    assert found_data_after, "Should have found data after gap"


def test_segment_source_boundary_gaps():
    """Test gap buffers at exact segment boundaries."""
    from sgnts.base import Offset

    # Create segments that touch at boundaries
    # This is where the bug was - boundary touches were treated as overlaps
    segments = (
        (0, int(0.3 * 1e9)),  # 0 to 0.3s
        (int(0.3 * 1e9), int(0.6 * 1e9)),  # 0.3 to 0.6s (touches previous)
        (int(0.8 * 1e9), int(1.0 * 1e9)),  # 0.8 to 1.0s (gap from 0.6-0.8s)
    )
    values = (100, 200, 300)

    src = SegmentSource(
        name="test",
        source_pad_names=("data",),
        rate=128,
        t0=0.0,
        duration=1.0,
        segments=segments,
        values=values,
    )

    pad = src.srcs["data"]

    # Get all buffers in first frame
    frame = src.new(pad)

    # Track what we find
    gap_offsets = []
    data_values = {}

    for buf in frame:
        if buf.is_gap:
            gap_offsets.append((buf.offset, buf.end_offset))
        else:
            # Record the value at this offset range
            if buf.data is not None and len(buf.data) > 0:
                data_values[(buf.offset, buf.end_offset)] = buf.data[0]

    # There should be a gap from 0.6s to 0.8s
    gap_start = Offset.fromns(int(0.6 * 1e9))
    gap_end = Offset.fromns(int(0.8 * 1e9))

    # Check that we have gap buffer(s) in the gap region
    found_gap_in_region = False
    for start, end in gap_offsets:
        if start >= gap_start and end <= gap_end:
            found_gap_in_region = True
            break

    assert (
        found_gap_in_region
    ), f"Should have gap between 0.6-0.8s, but gaps are at: {gap_offsets}"

    # Verify no gaps where segments touch (at 0.3s boundary)
    boundary_offset = Offset.fromns(int(0.3 * 1e9))
    for start, end in gap_offsets:
        # Make sure no gap spans across the boundary where segments touch
        assert not (
            start < boundary_offset < end
        ), f"Should not have gap at segment boundary 0.3s, but found gap {start}-{end}"


def test_segment_source_multiple_gaps():
    """Test multiple gaps in various positions."""
    from sgnts.base import Offset

    # Multiple gaps: at start, middle, and end
    segments = (
        (int(0.2 * 1e9), int(0.3 * 1e9)),  # Gap before (0-0.2s)
        (int(0.5 * 1e9), int(0.6 * 1e9)),  # Gap before and after (0.3-0.5s, 0.6-0.8s)
        (int(0.8 * 1e9), int(0.9 * 1e9)),  # Gap after (0.9-1.0s)
    )
    values = (11, 22, 33)

    src = SegmentSource(
        name="test",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        duration=1.0,
        segments=segments,
        values=values,
    )

    pad = src.srcs["data"]

    # Collect all buffers
    all_buffers = []
    for _ in range(10):
        frame = src.new(pad)
        all_buffers.extend(list(frame.buffers))
        if frame.EOS:
            break

    # Count gaps and data buffers
    gap_count = sum(1 for buf in all_buffers if buf.is_gap)
    data_count = sum(1 for buf in all_buffers if not buf.is_gap)

    assert gap_count > 0, "Should have gap buffers"
    assert data_count > 0, "Should have data buffers"

    # Verify gap regions exist
    gap_regions = [
        (0, int(0.2 * 1e9)),  # Start gap
        (int(0.3 * 1e9), int(0.5 * 1e9)),  # Middle gap 1
        (int(0.6 * 1e9), int(0.8 * 1e9)),  # Middle gap 2
        (int(0.9 * 1e9), int(1.0 * 1e9)),  # End gap
    ]

    for gap_start_ns, gap_end_ns in gap_regions:
        gap_start = Offset.fromns(gap_start_ns)
        gap_end = Offset.fromns(gap_end_ns)

        # Find buffers in this gap region (may not be entirely within)
        gap_found = any(
            buf.is_gap and buf.offset < gap_end and buf.end_offset > gap_start
            for buf in all_buffers
        )

        assert gap_found, (
            f"Should have gap buffer(s) in region "
            f"{gap_start_ns/1e9:.1f}-{gap_end_ns/1e9:.1f}s"
        )
