from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from sgn.base import SourcePad

from sgnts.base import Offset, TSFrame, TSSlice, TSSlices, TSSource
from sgnts.base.array_ops import Array


@dataclass
class SegmentSource(TSSource):
    """Produce non-gap buffers for segments, and gap buffers otherwise.

    Args:
        rate:
            int, the sample rate of the data
        segments:
            tuple[tuple[int, int], ...], a tuple of segment tuples corresponding to
            time in ns
        values:
            Optional[tuple[Union[int, Array], ...]], optional tuple of values to set
            for each segment's non-gap buffers. Must be same length as segments.
            If None, defaults to 1 for all non-gap buffers.
    """

    rate: int = 2048
    segments: Optional[tuple[tuple[int, int], ...]] = None
    values: Optional[tuple[Union[int, Array], ...]] = None

    def __post_init__(self):
        assert (
            self.segments is not None
        ), "Segments must be provided during initialization"

        # Assert that segments are non-overlapping (but adjacent is OK)
        # Check by ensuring no two segments have overlapping interiors
        for i in range(len(self.segments)):
            for j in range(i + 1, len(self.segments)):
                seg1_start, seg1_end = self.segments[i]
                seg2_start, seg2_end = self.segments[j]
                # Check if segments overlap (not just touch at boundary)
                if seg1_start < seg2_end and seg2_start < seg1_end:
                    # They overlap if seg1 starts before seg2 ends AND seg2
                    # starts before seg1 ends
                    # But we need to exclude the case where they just touch at
                    # a boundary
                    if not (seg1_end == seg2_start or seg2_end == seg1_start):
                        raise AssertionError(
                            f"Input segments must be non-overlapping. "
                            f"Segments {i} ({seg1_start}, {seg1_end}) and "
                            f"{j} ({seg2_start}, {seg2_end}) overlap."
                        )

        # Validate values if provided
        if self.values is not None:
            assert len(self.values) == len(self.segments), (
                f"Length of values ({len(self.values)}) must match "
                f"length of segments ({len(self.segments)})"
            )

        super().__post_init__()
        assert (
            len(self.source_pads) == 1
        ), f"SegmentSource requires exactly one source pad, got {len(self.source_pads)}"

        # Filter segments that overlap with the source time range and track
        # their indices
        self.segment_data = []  # List of (slice, original_index) tuples
        t0_ns = self.t0 * 1e9
        end_ns = self.end * 1e9

        for i, s in enumerate(self.segments):
            # Include segments that have any overlap with the time range
            if s[0] < end_ns and s[1] > t0_ns:
                # Clip segment to the source time range
                seg_start = max(s[0], t0_ns)
                seg_end = min(s[1], end_ns)
                slice_obj = TSSlice(
                    Offset.fromns(seg_start, sample_rate=self.rate),
                    Offset.fromns(seg_end, sample_rate=self.rate),
                )
                self.segment_data.append((slice_obj, i))

        # Create TSSlices from just the slices
        self.segment_slices = TSSlices([sd[0] for sd in self.segment_data])

        for pad in self.source_pads:
            self.set_pad_buffer_params(pad=pad, sample_shape=(), rate=self.rate)

    def new(self, pad: SourcePad) -> TSFrame:
        """New TSFrames are created on "pad" with stride matching the stride specified
        in Offset.SAMPLE_STRIDE_AT_MAX_RATE. EOS is set if we have reach the requested
        "end" time. Non-gap buffers will be produced when they are within the segments
        provided, and gap buffers will be produced otherwise.

        Args:
            pad:
                SourcePad, the pad for which to produce a new TSFrame

        Returns:
            TSFrame, the TSFrame with non-gap buffers within segments and gap buffers
            outside segments.
        """
        # FIXME: Find a better way to set EOS
        # Create frame with default data=None (gap buffers)
        frame = self.prepare_frame(pad, data=None)

        bufs = []
        for buf in frame:
            # Find which segments overlap with this buffer
            nongap_slices = self.segment_slices.search(buf.slice)

            if nongap_slices and nongap_slices.slices:
                # Split the buffer based on gap/non-gap regions
                split_bufs = buf.split(nongap_slices, contiguous=True)

                # For each split buffer, determine if it's gap or non-gap
                for split_buf in split_bufs:
                    # Check if this buffer overlaps with any segment
                    for slice_obj, orig_idx in self.segment_data:
                        overlap = split_buf.slice & slice_obj
                        # Only consider finite overlaps (not just boundary touches)
                        if overlap and overlap.isfinite():  # Has finite overlap
                            # Set the appropriate value for this non-gap buffer
                            if self.values is not None:
                                split_buf.set_data(self.values[orig_idx])
                            else:
                                split_buf.set_data(1)  # Default to 1
                            break

                    # Gap buffers keep data=None (already set)
                    bufs.append(split_buf)
            else:
                # No overlap with any segment, keep as gap buffer
                bufs.append(buf)

        frame.set_buffers(bufs)

        return frame
