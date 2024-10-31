from dataclasses import dataclass

from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSlice, TSSlices, TSSource


@dataclass
class SegmentSrc(TSSource):
    """

    Parameters
    ----------
    rate: int
        the sample rate of the data
    segments: tuple
        A tuple of segment tuples corresponding to time in ns
    end: int
        The time at which to stop producing buffers
    """

    rate: int = 2048
    segments: tuple = None
    end: float = None

    def __post_init__(self):
        assert isinstance(self.end, float)
        assert self.segments is not None
        super().__post_init__()
        # FIXME
        self.segments = TSSlices(
            TSSlice(Offset.fromns(s[0]), Offset.fromns(s[1]))
            for s in self.segments
            if (s[0] >= self.t0 * 1e9 and s[1] <= self.end * 1e9)
        ).simplify()
        self.num_samples = Offset.sample_stride(self.rate)

    def new(self, pad):
        """ """
        frame_slice = TSSlice(
            self.offset[pad],
            self.offset[pad] + Offset.fromsamples(self.num_samples, self.rate),
        )
        nongap_slices = self.segments.search(frame_slice)
        gap_slices = nongap_slices.invert(frame_slice)
        outbufs = [
            SeriesBuffer.fromoffsetslice(s, self.rate) for s in gap_slices.slices
        ]
        outbufs.extend(
            [
                SeriesBuffer.fromoffsetslice(s, self.rate, data=1)
                for s in nongap_slices.slices
            ]
        )
        outbufs = sorted(outbufs)

        self.offset[pad] = frame_slice.stop
        return TSFrame(buffers=outbufs, EOS=outbufs[-1].end >= self.end * 1e9)  # FIXME
