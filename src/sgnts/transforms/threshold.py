from dataclasses import dataclass

import numpy
from sgn.base import SourcePad

from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSlice, TSSlices, TSTransform


@dataclass
class Threshold(TSTransform):
    """Only allow data above or below a threshold to pass. data will otherwise be marked
    as gap.

    Args:
        threshold:
            float, the absolute value threshold above which to allow data to pass
        invert:
            bool, If False, only data above a threshold will pass. If True: only data
            below a threshold will pass
        startwn:
            int, the number of samples ahead of the crossing to allow data to pass
        stopwn:
            int, the number of samples after the crossing to allow data to pass
    """

    threshold: float = float("+inf")
    invert: bool = False
    startwn: int = 0
    stopwn: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert len(self.sink_pads) == 1
        assert len(self.source_pads) == 1
        self.sinkpad = self.sink_pads[0]
        self.nongap_slices = TSSlices([])

    # Modified from: https://stackoverflow.com/questions/43258896/
    # extract-subarrays-of-numpy-array-whose-values-are-above-a-threshold
    def __split_above_threshold(
        self,
        buffer: SeriesBuffer,
        threshold: float,
        start_window: int = 0,
        stop_window: int = 0,
    ) -> list[TSSlice]:
        """Find subslices in buffer whose data are above threshold, along with
        start_window samples ahead of and stop_window samples after the crossing.

        Args:
            buffer:
                SeriesBuffer, the buffer from which to extract subslices
            threshold:
                float, the crossing threshold
            start_window:
                int, the number of samples ahead of the crossing to allow data to pass
            stop_window:
                int, the number of samples after the crossing to allow data to pass

        Returns:
            list[TSSlice], a list of TSSlices whose data value crossed a threshold,
            along with a window around the crossing
        """
        signal = buffer.data
        sample_rate = buffer.sample_rate
        off0 = buffer.offset
        mask = numpy.concatenate(([False], numpy.abs(signal) >= threshold, [False]))
        idx = numpy.flatnonzero(mask[1:] != mask[:-1])
        return [
            TSSlice(
                off0 + Offset.fromsamples(int(idx[i] - start_window), sample_rate),
                off0 + Offset.fromsamples(int(idx[i + 1] + stop_window), sample_rate),
            )
            for i in range(0, len(idx), 2)
        ]

    def transform(self, pad: SourcePad) -> TSFrame:
        """Split buffers into subbuffers of gaps and nongaps depending on whether the
        data passed a threshold.

        Args:
            pad:
                SourcePad, the source pad to output the transformed frame

        Returns:
            TSFrame, the output TSFrame
        """
        frame = self.preparedframes[self.sinkpad]
        boundary_offsets = TSSlice(
            frame[0].offset,
            frame[-1].end_offset,
        )
        self.nongap_slices += TSSlices(
            [
                j
                for sub in [
                    self.__split_above_threshold(
                        b,
                        self.threshold,
                        self.startwn,
                        self.stopwn,
                    )
                    for b in frame
                    if b
                ]
                for j in sub
            ]
        )
        self.nongap_slices = self.nongap_slices.simplify()

        # restrict to slices that are new enough to matter
        self.nongap_slices = TSSlices(
            [
                s
                for s in self.nongap_slices.slices
                if not s.stop <= boundary_offsets.start
            ]
        )

        aligned_nongap_slices = self.nongap_slices.search(boundary_offsets, align=True)
        if self.invert:
            aligned_nongap_slices = aligned_nongap_slices.invert(boundary_offsets)

        out = sorted(
            [
                b
                for bs in [
                    buf.split(aligned_nongap_slices.search(buf.slice), contiguous=True)
                    for buf in frame
                ]
                for b in bs
            ]
        )

        # sanity check that buffers don't overlap
        o0 = out[0]
        for o in out[1:]:
            if o.offset != o0.end_offset:
                raise ValueError(f"overlapping offsets {o0.slice, o.slice}")
            o0 = o

        return TSFrame(buffers=out, EOS=self.at_EOS, metadata=frame.metadata)
