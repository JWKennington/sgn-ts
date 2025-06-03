"""Test SegmentSource with partially overlapping segments"""

from sgnts.sources import SegmentSource
from sgnts.base import Offset


def test_segment_partial_overlap_at_end():
    """Test segment that extends past the end time"""

    # Segment goes from 4s to 6s, but source ends at 5s
    segments = ((4_000_000_000, 6_000_000_000),)
    values = (42,)

    source = SegmentSource(
        name="test",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        end=5.0,  # Ends at 5s
        segments=segments,
        values=values,
    )

    # Should have 1 segment, clipped to end at 5s
    assert len(source.segment_data) == 1

    seg_slice, orig_idx = source.segment_data[0]
    assert orig_idx == 0  # Original index preserved

    # Check clipping
    assert seg_slice.start == Offset.fromns(4_000_000_000)
    assert seg_slice.stop == Offset.fromns(5_000_000_000)  # Clipped to 5s

    # Get a frame and verify data exists
    pad = source.srcs["data"]

    # Collect data over multiple frames
    found_data = False
    for _ in range(20):
        frame = source.new(pad)
        for buf in frame:
            if not buf.is_gap and buf.data is not None and len(buf.data) > 0:
                assert buf.data[0] == 42
                found_data = True
        if frame.EOS:
            break

    assert found_data, "Should have found data in the partial segment"


def test_segment_partial_overlap_at_start():
    """Test segment that starts before t0"""

    # Segment goes from -1s to 1s, but source starts at 0s
    segments = ((-1_000_000_000, 1_000_000_000),)
    values = (33,)

    source = SegmentSource(
        name="test",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,  # Starts at 0s
        end=2.0,
        segments=segments,
        values=values,
    )

    # Should have 1 segment, clipped to start at 0s
    assert len(source.segment_data) == 1

    seg_slice, orig_idx = source.segment_data[0]
    assert orig_idx == 0

    # Check clipping
    assert seg_slice.start == Offset.fromns(0)  # Clipped to 0s
    assert seg_slice.stop == Offset.fromns(1_000_000_000)


def test_segment_fully_outside():
    """Test segments completely outside the time range are excluded"""

    segments = (
        (10_000_000_000, 11_000_000_000),  # 10-11s, fully after end
        (-3_000_000_000, -1_000_000_000),  # -3 to -1s, fully before start
    )
    values = (100, 200)

    source = SegmentSource(
        name="test",
        source_pad_names=("data",),
        rate=256,
        t0=0.0,
        end=5.0,
        segments=segments,
        values=values,
    )

    # No segments should be kept
    assert len(source.segment_data) == 0


def test_multiple_segments_with_partial_overlap():
    """Test mix of fully contained and partially overlapping segments"""

    segments = (
        (500_000_000, 1_500_000_000),  # 0.5-1.5s: fully contained
        (2_000_000_000, 3_000_000_000),  # 2-3s: fully contained
        (4_000_000_000, 6_000_000_000),  # 4-6s: extends past end
        (7_000_000_000, 8_000_000_000),  # 7-8s: fully outside
    )
    values = (10, 20, 30, 40)

    source = SegmentSource(
        name="test",
        source_pad_names=("data",),
        rate=64,  # Lower rate for easier testing
        t0=0.0,
        end=5.0,
        segments=segments,
        values=values,
    )

    # Should keep 3 segments (first 3)
    assert len(source.segment_data) == 3

    # Check each kept segment
    expected = [
        (0, 500_000_000, 1_500_000_000),  # Fully contained
        (1, 2_000_000_000, 3_000_000_000),  # Fully contained
        (2, 4_000_000_000, 5_000_000_000),  # Clipped at end
    ]

    for i, (seg_slice, orig_idx) in enumerate(source.segment_data):
        exp_idx, exp_start_ns, exp_end_ns = expected[i]
        assert orig_idx == exp_idx
        assert seg_slice.start == Offset.fromns(exp_start_ns)
        assert seg_slice.stop == Offset.fromns(exp_end_ns)


def test_tiny_overlap():
    """Test segment with tiny overlap at boundary"""

    # Segment that just barely overlaps
    segments = ((4_999_000_000, 5_001_000_000),)  # 4.999s to 5.001s
    values = (77,)

    source = SegmentSource(
        name="test",
        source_pad_names=("data",),
        rate=1024,  # High rate to catch small overlap
        t0=0.0,
        end=5.0,
        segments=segments,
        values=values,
    )

    # Should keep the segment
    assert len(source.segment_data) == 1

    # Clipped to exactly 5s
    seg_slice, _ = source.segment_data[0]
    assert seg_slice.stop == Offset.fromns(5_000_000_000)


if __name__ == "__main__":
    test_segment_partial_overlap_at_end()
    test_segment_partial_overlap_at_start()
    test_segment_fully_outside()
    test_multiple_segments_with_partial_overlap()
    test_tiny_overlap()
