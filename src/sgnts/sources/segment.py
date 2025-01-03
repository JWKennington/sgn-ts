from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sgn.base import SourcePad

from sgnts.base import Offset, TSFrame, TSSlice, TSSlices, TSSource


@dataclass
class SegmentSource(TSSource):
    """Produce non-gap buffers for segments, and gap buffers otherwise.

    Args:
        rate:
            int, the sample rate of the data
        segments:
            tuple[tuple[int, int], ...], a tuple of segment tuples corresponding to
            time in ns
    """

    rate: int = 2048
    segments: Optional[tuple[tuple[int, int], ...]] = None

    def __post_init__(self):
        assert self.segments is not None
        super().__post_init__()
        assert len(self.source_pads) == 1
        # FIXME
        self.segment_slices = TSSlices(
            TSSlice(Offset.fromns(s[0]), Offset.fromns(s[1]))
            for s in self.segments
            if (s[0] >= self.t0 * 1e9 and s[1] <= self.end * 1e9)
        ).simplify()

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
        frame = self.prepare_frame(pad, data=1)

        bufs = []
        for buf in frame:
            nongap_slices = self.segment_slices.search(buf.slice)
            bufs.extend(buf.split(nongap_slices, contiguous=True))

        frame.set_buffers(bufs)

        return frame
