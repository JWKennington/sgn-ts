import numpy
from dataclasses import dataclass
from ..base import TSTransform, TSSlices, TSSlice, TSFrame, Offset


@dataclass
class Threshold(TSTransform):
    """
    Only allow data above or below a threshold to pass. data will otherwise be marked as gap.

    Parameters:
    -----------
    threshold: float
        The absolute value threshold above which to allow data to pass
    invert: bool
    startwn: int
        The number of samples ahead of the crossing to allow data to pass
    stopwn: int
        The number of samples ahead of the crossing to allow data to pass
    """

    threshold: float = None
    invert: bool = False
    startwn: int = None
    stopwn: int = None

    def __post_init__(self):
        super().__post_init__()
        assert len(self.sink_pads) == 1
        assert len(self.source_pads) == 1
        assert self.threshold is not None
        assert self.startwn is not None
        assert self.stopwn is not None
        self.sinkpad = self.sink_pads[0]
        self.nongap_slices = TSSlices([])

    # Modified from: https://stackoverflow.com/questions/43258896/extract-subarrays-of-numpy-array-whose-values-are-above-a-threshold
    def __split_above_threshold(
        self, buffer, threshold, start_window=0, stop_window=0, invert=False
    ):
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

    def transform(self, pad):
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
                        self.invert,
                    )
                    for b in frame
                    if b
                ]
                for j in sub
            ]
        ).simplify()
        # restrict to slices that are new enough to matter
        self.nongap_slices = TSSlices(
            [s for s in self.nongap_slices.slices if not s.stop <= boundary_offsets.start]
        )

        aligned_nongap_slices = self.nongap_slices.search(boundary_offsets, align=True)
        if self.invert:
            aligned_nongap_slices = aligned_nongap_slices.invert(boundary_offsets)

        out = sorted(
            [
                b
                for bs in [
                    buf.split(aligned_nongap_slices, contiguous=True)
                    for buf in frame
                ]
                for b in bs
            ]
        )
        return TSFrame(buffers=out, EOS=self.at_EOS, metadata=frame.metadata)
