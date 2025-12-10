from dataclasses import dataclass

import numpy
from sgn import validator

from sgnts.base import (
    Offset,
    SeriesBuffer,
    TSCollectFrame,
    TSFrame,
    TSSlice,
    TSSlices,
    TSTransform,
)
from sgnts.decorators import transform


# FIXME: only supports numpy and not pytorch
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

    def configure(self) -> None:
        self.nongap_slices = TSSlices([])

    @validator.one_to_one
    def validate(self) -> None:
        pass

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
        signal = numpy.array(buffer.data)
        sample_rate = buffer.sample_rate
        off0 = buffer.offset
        # NOTE the tuple casting is here because of mypy. Numpy typing seems a
        # bit broken in a few places.
        mask: numpy.ndarray = numpy.concatenate(
            ((False,), tuple(numpy.abs(signal) >= threshold), (False,))
        )
        idx = numpy.flatnonzero(mask[1:] != mask[:-1])
        return [
            TSSlice(
                off0 + Offset.fromsamples(int(idx[i] - start_window), sample_rate),
                off0 + Offset.fromsamples(int(idx[i + 1] + stop_window), sample_rate),
            )
            for i in range(0, len(idx), 2)
        ]

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: TSCollectFrame) -> None:
        """Process frame to threshold data based on absolute value."""

        boundary_offsets = TSSlice(
            input_frame[0].offset,
            input_frame[-1].end_offset,
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
                    for b in input_frame
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
                    for buf in input_frame
                ]
                for b in bs
            ]
        )

        # sanity check that buffers don't overlap
        o0 = out[0]
        for o in out[1:]:
            assert o.offset == o0.end_offset, (
                f"Buffer overlap detected: buffer offset {o.offset} != "
                f"previous buffer end_offset {o0.end_offset}"
            )
            o0 = o

        output_frame.extend(out)
